# MLSys Graph Scheduler

This solver applies optimization techniques inspired by VLIW (Very Long Instruction Word) instruction scheduling to the MLSys graph scheduling problem.

## VLIW-Inspired Optimizations

The solution adapts key techniques from the tinygrad VLIW backend (see the Anthropic challenge implementation):

### 1. **Dependency-Based List Scheduling**
- **VLIW Analogy**: Schedule instructions based on dependency chains and critical path depth
- **MLSys Application**: Build dependency graph of operations, compute critical path depth, and schedule operations respecting data dependencies
- **Implementation**: `DependencyAnalyzer` class tracks producer-consumer relationships and computes scheduling priorities

### 2. **Resource-Aware Packing**
- **VLIW Analogy**: Pack instructions into bundles respecting execution unit slot limits (alu: 12, valu: 6, load: 2, store: 2, flow: 1)
- **MLSys Application**: Pack operations into subgraphs respecting fast memory capacity constraints
- **Implementation**: `ListScheduler._check_fusion_memory()` ensures working sets fit in fast memory

### 3. **Adaptive Granularity Optimization**
- **VLIW Analogy**: Opportunistic demotion - when VALU is full, demote vector ops to ALU as scalar ops
- **MLSys Application**: Adjust execution granularity when memory is tight - start with native granularity and reduce if needed
- **Implementation**: `GranularityOptimizer.find_valid_granularity()` tries granularities from large to small

### 4. **Register Allocation Principles**
- **VLIW Analogy**: Linear scan register allocator tracks register lifetimes and reuses freed registers
- **MLSys Application**: Track tensor lifetimes to determine which tensors to retain in fast memory
- **Implementation**: `ListScheduler._find_tensors_to_retain()` keeps only tensors needed by future operations

### 5. **Operation Fusion**
- **VLIW Analogy**: Bundle multiple instructions into a single VLIW word for parallel execution
- **MLSys Application**: Fuse producer-consumer operation chains into single subgraphs (ephemeral intermediates don't consume fast memory)
- **Implementation**: `ListScheduler.can_fuse()` and greedy fusion in `schedule()`

## Key Insights

1. **Ephemeral Data**: Intermediate tensors produced and consumed within a subgraph (like ephemeral registers in VLIW bundles) don't need fast memory storage - they flow directly from producer to consumer.

2. **Working Set vs Full Tensors**: Unlike naive approaches that try to fit entire tensors in memory, we only need to fit the working set at the execution granularity (slices/tiles).

3. **Priority-Based Scheduling**: Operations are scheduled in critical path order (highest depth first), ensuring dependencies are satisfied and minimizing pipeline stalls.

## Usage

```bash
python3 mlsys_solver.py <input.json> <output.json>
```

Example:
```bash
python3 mlsys_solver.py benchmarks/mlsys-2026-1.json solution.json
```

## Architecture

- `Problem`: Loads and represents the input problem
- `DependencyAnalyzer`: Builds dependency graph and computes critical paths  
- `GranularityOptimizer`: Finds valid execution granularities respecting memory constraints
- `ListScheduler`: Main scheduling algorithm with fusion logic
- `MLSysSolver`: Orchestrates the solution generation

## Future Improvements

1. **Better Fusion**: Adjust granularity dynamically when fusing to enable more fusion opportunities
2. **Batch Staggering**: Implement the batch offset optimization from VLIW for pipelined execution
3. **Cost-Based Optimization**: Use latency calculations to guide scheduling decisions
4. **Multi-Operation Subgraphs**: Extend beyond simple chains to more complex subgraph patterns
