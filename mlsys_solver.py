#!/usr/bin/env python3
"""
MLSys Graph Scheduler using techniques inspired by VLIW instruction scheduling.

Key optimizations adapted from VLIW:
1. Dependency-based list scheduling with critical path priority
2. Resource-aware packing (memory capacity instead of execution slots)
3. Adaptive granularity adjustment (similar to opportunistic demotion)
4. Tensor lifetime analysis for memory reuse
"""

import json
import sys
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional
import math


@dataclass
class Tensor:
    """Represents a tensor with dimensions."""
    idx: int
    width: int
    height: int
    
    @property
    def size(self) -> int:
        """Memory footprint in elements."""
        return self.width * self.height


@dataclass
class Operation:
    """Represents a computation operation."""
    idx: int
    op_type: str
    inputs: List[int]
    outputs: List[int]
    base_cost: int
    
    def compute_cost(self, w: int, h: int, k: int, native_w: int, native_h: int) -> float:
        """
        Compute cost for this operation at given granularity.
        Pads to native granularity if smaller.
        """
        actual_w = max(w, native_w)
        actual_h = max(h, native_h)
        
        if self.op_type == "MatMul":
            # MatMul cost scales with output tiles and reduction steps
            return self.base_cost * (actual_w / native_w) * (actual_h / native_h)
        else:  # Pointwise
            # Pointwise ignores k, just spatial tiles
            return self.base_cost * (actual_w / native_w) * (actual_h / native_h)


@dataclass
class Problem:
    """The MLSys scheduling problem."""
    tensors: List[Tensor]
    ops: List[Operation]
    fast_memory_capacity: int
    slow_memory_bandwidth: int
    native_granularity: Tuple[int, int]
    
    @staticmethod
    def from_json(filename: str) -> 'Problem':
        """Load problem from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        tensors = [Tensor(i, w, h) for i, (w, h) in enumerate(zip(data['widths'], data['heights']))]
        ops = [
            Operation(i, op_type, inputs, outputs, base_cost)
            for i, (op_type, inputs, outputs, base_cost) in enumerate(zip(
                data['op_types'], data['inputs'], data['outputs'], data['base_costs']
            ))
        ]
        
        return Problem(
            tensors=tensors,
            ops=ops,
            fast_memory_capacity=data['fast_memory_capacity'],
            slow_memory_bandwidth=data['slow_memory_bandwidth'],
            native_granularity=tuple(data['native_granularity'])
        )


@dataclass
class Subgraph:
    """A scheduled subgraph with execution granularity."""
    ops: List[int]
    tensors_to_retain: List[int]
    granularity: Tuple[int, int, int]  # (w, h, k)
    
    def to_json(self) -> dict:
        """Convert to JSON format for output."""
        return {
            "ops": self.ops,
            "tensors_to_retain": self.tensors_to_retain,
            "granularity": list(self.granularity)
        }


class DependencyAnalyzer:
    """
    Analyzes operation dependencies (inspired by VLIW dependency tracking).
    Builds dependency graph and computes critical path depths.
    """
    
    def __init__(self, problem: Problem):
        self.problem = problem
        self.deps = self._build_deps()
        self.depth = self._compute_depth()
        self.graph_inputs = self._find_graph_inputs()
        self.graph_outputs = self._find_graph_outputs()
    
    def _build_deps(self) -> List[Set[int]]:
        """Build dependency graph: deps[i] = set of ops that must run before op i."""
        # First, find which op produces each tensor
        tensor_producer: Dict[int, int] = {}
        for op in self.problem.ops:
            for t_idx in op.outputs:
                tensor_producer[t_idx] = op.idx
        
        # Now build dependencies
        deps: List[Set[int]] = [set() for _ in self.problem.ops]
        for op in self.problem.ops:
            for t_idx in op.inputs:
                if t_idx in tensor_producer:
                    deps[op.idx].add(tensor_producer[t_idx])
        
        return deps
    
    def _compute_depth(self) -> List[int]:
        """Compute critical path depth (longest dependency chain to each op)."""
        depth = [0] * len(self.problem.ops)
        for i in range(len(self.problem.ops)):
            if self.deps[i]:
                depth[i] = 1 + max(depth[d] for d in self.deps[i])
        return depth
    
    def _find_graph_inputs(self) -> Set[int]:
        """Find tensors that are graph inputs (not produced by any op)."""
        produced = set()
        for op in self.problem.ops:
            produced.update(op.outputs)
        return set(range(len(self.problem.tensors))) - produced
    
    def _find_graph_outputs(self) -> Set[int]:
        """Find tensors that are graph outputs (not consumed by any op)."""
        consumed = set()
        for op in self.problem.ops:
            consumed.update(op.inputs)
        all_tensors = set(range(len(self.problem.tensors)))
        return all_tensors - consumed


class GranularityOptimizer:
    """
    Optimizes execution granularity (inspired by VLIW opportunistic demotion).
    Adaptively adjusts granularity to fit within memory constraints.
    """
    
    def __init__(self, problem: Problem):
        self.problem = problem
        self.native_w, self.native_h = problem.native_granularity
    
    def find_valid_granularity(self, op: Operation, required_tensors: Set[int]) -> Tuple[int, int, int]:
        """
        Find a valid granularity that fits memory constraints.
        Starts with native and reduces if needed (like opportunistic demotion in VLIW).
        """
        # Get output tensor dimensions to determine max granularity
        output_tensor = self.problem.tensors[op.outputs[0]]
        max_w = output_tensor.width
        max_h = output_tensor.height
        
        # Try different granularities, starting from native
        # Use powers of 2 for efficient tiling
        w_options = [min(w, max_w) for w in [self.native_w, self.native_w // 2, 64, 32, 16] if w <= max_w]
        h_options = [min(h, max_h) for h in [self.native_h, self.native_h // 2, 64, 32, 16] if h <= max_h]
        
        if not w_options: w_options = [max_w]
        if not h_options: h_options = [max_h]
        
        for w in w_options:
            for h in h_options:
                # For MatMul, also try different k values
                if op.op_type == "MatMul":
                    # Get K dimension from input tensors
                    # For MatMul: A is (M, K), B is (K, N), output is (M, N)
                    # inputs[0] is A, inputs[1] is B
                    input_a = self.problem.tensors[op.inputs[0]]
                    max_k = input_a.width  # K dimension
                    k_values = [min(k, max_k) for k in [128, 64, 32, 16] if k <= max_k]
                    if not k_values: k_values = [max_k]
                else:
                    k_values = [1]
                
                for k in k_values:
                    # Calculate working set for this granularity
                    working_set = self._calculate_working_set(op, w, h, k)
                    
                    # Check if it fits in fast memory
                    if working_set <= self.problem.fast_memory_capacity:
                        return (w, h, k)
        
        # Fallback to minimum possible
        return (min(16, max_w), min(16, max_h), 16 if op.op_type == "MatMul" else 1)
    
    def _calculate_working_set(self, op: Operation, w: int, h: int, k: int) -> int:
        """
        Calculate the working set size for executing op at this granularity.
        This is the memory needed for input/output slices during execution.
        """
        if op.op_type == "MatMul":
            # MatMul: C[w,h] = A[w,k] @ B[k,h]
            # Working set = output slice + input A slice + input B slice
            # Output slice: w * h (accumulator)
            # Input A slice: w * k
            # Input B slice: k * h
            return w * h + w * k + k * h
        else:  # Pointwise
            # Pointwise operations: output = f(inputs...)
            # Working set = sum of all input slices + output slice
            # Each slice is w * h
            return w * h * (len(op.inputs) + len(op.outputs))


class ListScheduler:
    """
    List scheduler inspired by VLIW packing.
    Schedules operations respecting dependencies and memory constraints.
    """
    
    def __init__(self, problem: Problem):
        self.problem = problem
        self.analyzer = DependencyAnalyzer(problem)
        self.optimizer = GranularityOptimizer(problem)
    
    def schedule(self) -> List[Subgraph]:
        """
        Create a schedule using list scheduling with priority ordering.
        Similar to VLIWPacker but for graph operations.
        """
        subgraphs = []
        scheduled = [False] * len(self.problem.ops)
        
        # Track which tensors are currently in fast memory
        in_fast_memory = set(self.analyzer.graph_inputs)
        
        while not all(scheduled):
            # Find ready operations (all dependencies satisfied)
            ready = [
                i for i in range(len(self.problem.ops))
                if not scheduled[i] and all(scheduled[d] for d in self.analyzer.deps[i])
            ]
            
            if not ready:
                raise RuntimeError("Scheduling deadlock detected")
            
            # Sort by priority (depth-based, like VLIW)
            ready.sort(key=lambda i: -self.analyzer.depth[i])
            
            # Try to pack operations into a subgraph
            # For simplicity, start with one op per subgraph
            # A more sophisticated approach would fuse compatible ops
            for op_idx in ready:
                op = self.problem.ops[op_idx]
                
                # Determine which tensors need to be in fast memory
                required_tensors = set(op.inputs)
                
                # Find valid granularity
                granularity = self.optimizer.find_valid_granularity(op, required_tensors)
                
                # Determine which tensors to retain after this subgraph
                # Keep outputs if they're used by future ops or are graph outputs
                tensors_to_retain = []
                for t_idx in op.outputs:
                    if t_idx in self.analyzer.graph_outputs:
                        tensors_to_retain.append(t_idx)
                    else:
                        # Check if used by unscheduled ops
                        for future_op in self.problem.ops:
                            if not scheduled[future_op.idx] and t_idx in future_op.inputs:
                                tensors_to_retain.append(t_idx)
                                break
                
                # Create subgraph
                subgraph = Subgraph(
                    ops=[op_idx],
                    tensors_to_retain=tensors_to_retain,
                    granularity=granularity
                )
                
                subgraphs.append(subgraph)
                scheduled[op_idx] = True
                
                # Update fast memory state
                in_fast_memory.update(op.outputs)
                
                # Only process one ready op at a time for now
                break
        
        return subgraphs


class MLSysSolver:
    """Main solver orchestrating the scheduling process."""
    
    def __init__(self, problem: Problem):
        self.problem = problem
        self.scheduler = ListScheduler(problem)
    
    def solve(self) -> dict:
        """Generate a solution."""
        subgraphs = self.scheduler.schedule()
        
        return {
            "subgraphs": [sg.to_json() for sg in subgraphs]
        }


def main():
    if len(sys.argv) != 3:
        print("Usage: python mlsys_solver.py <input.json> <output.json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load problem
    problem = Problem.from_json(input_file)
    
    # Solve
    solver = MLSysSolver(problem)
    solution = solver.solve()
    
    # Write solution
    with open(output_file, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"Solution written to {output_file}")


if __name__ == "__main__":
    main()
