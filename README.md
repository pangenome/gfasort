# GFASort

A Rust library for sorting bidirected pangenome graphs using the Ygs algorithm (path-guided SGD + grooming + topological sort), exactly matching ODGI's behavior.

## Features

- **Path-guided SGD**: Positions nodes to minimize discrepancy between path distances and layout distances
- **Grooming**: Ensures consistent node orientations along paths
- **Topological Sort**: Produces valid linearization respecting edge directions
- **Deterministic**: Same input always produces same output
- **Multi-threaded**: Parallel SGD with configurable thread count
- **ODGI-compatible**: Exact port of ODGI's sorting algorithms

## Background

This library implements the Ygs sorting pipeline from [ODGI](https://github.com/pangenome/odgi), which consists of three stages:

1. **Y** - Path-guided Stochastic Gradient Descent (PG-SGD)
   - Uses path information to guide node placement
   - Minimizes error between path distances and graph layout distances
   - Multi-threaded with lock-free atomic updates

2. **g** - Grooming
   - Removes spurious inverting links
   - Ensures nodes are oriented consistently along paths
   - Uses BFS traversal to propagate orientation decisions

3. **s** - Topological Sort
   - Produces valid node ordering respecting edge directions
   - Modified Kahn's algorithm for bidirected graphs with cycles
   - Preserves SGD layout order when breaking cycles

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
gfasort = "0.1"
```

Or use from local path:

```toml
[dependencies]
gfasort = { path = "../gfasort" }
```

## Quick Start

```rust
use gfasort::{BidirectedGraph, YgsParams, ygs_sort, Handle};

// Build graph
let mut graph = BidirectedGraph::new();
graph.add_node(1, b"ACGT".to_vec());
graph.add_node(2, b"TGCA".to_vec());
graph.add_node(3, b"AAAA".to_vec());

// Add edges
graph.add_edge(Handle::forward(1), Handle::forward(2));
graph.add_edge(Handle::forward(2), Handle::forward(3));

// Add a path
let mut path = gfasort::BiPath::new("path1".to_string());
path.add_step(Handle::forward(1));
path.add_step(Handle::forward(2));
path.add_step(Handle::forward(3));
graph.paths.push(path);

// Sort with default parameters (calculated from graph)
let params = YgsParams::from_graph(&graph, false, 4); // 4 threads
ygs_sort(&mut graph, &params);

// Graph is now sorted!
```

## API

### Main Entry Points

```rust
// Full Ygs pipeline
pub fn ygs_sort(graph: &mut BidirectedGraph, params: &YgsParams)

// Individual stages
pub fn sgd_sort_only(graph: &mut BidirectedGraph, params: PathSGDParams, verbose: bool)
pub fn groom_only(graph: &mut BidirectedGraph, verbose: bool)
pub fn topological_sort_only(graph: &mut BidirectedGraph, verbose: bool)
```

### Configuration

```rust
pub struct YgsParams {
    pub path_sgd: PathSGDParams,
    pub verbose: bool,
}

pub struct PathSGDParams {
    pub iter_max: u64,              // Iterations (default: 100)
    pub min_term_updates: u64,      // Updates per iteration (auto-calculated)
    pub eta_max: f64,               // Learning rate (auto-calculated)
    pub theta: f64,                 // Zipfian exponent (default: 0.99)
    pub space: u64,                 // Jump distance (auto-calculated)
    pub cooling_start: f64,         // When to switch to local (default: 0.5)
    pub nthreads: usize,            // Thread count
    pub progress: bool,             // Print progress
    // ... (other fields with defaults)
}

// Convenience constructor (recommended)
impl YgsParams {
    pub fn from_graph(graph: &BidirectedGraph, verbose: bool, nthreads: usize) -> Self
}
```

## Algorithm Details

### Path-Guided SGD

The SGD algorithm optimizes node positions to match distances along paths:

1. Initialize positions based on current graph order
2. Calculate learning rate schedule (exponential decay)
3. Multi-threaded optimization:
   - Sample random pairs of nodes along paths
   - Use Zipfian distribution for neighbor selection (favors nearby nodes)
   - Update positions to minimize `|layout_distance - path_distance|`
   - Adaptive cooling: switches to local optimization after 50% of iterations

**Time Complexity**: O(iterations × total_path_length)
**Space Complexity**: O(V + total_path_length)

### Grooming

Grooming ensures consistent node orientations:

1. Analyze path-based orientation preferences (count forward vs reverse traversals)
2. Find head nodes (no incoming edges) as traversal seeds
3. BFS traversal:
   - Visit each node via edges
   - If reached via reverse orientation, mark for flipping
4. Apply flips:
   - Reverse complement sequences of flipped nodes
   - Update edge and path orientations

**Time Complexity**: O(V + E)
**Space Complexity**: O(V)

### Topological Sort

Modified Kahn's algorithm for bidirected graphs:

1. Start with head nodes (no incoming edges)
2. Process nodes in priority order
3. Break cycles by choosing lowest node ID (preserves SGD layout)
4. Emit nodes in topological order

**Time Complexity**: O(V + E)
**Space Complexity**: O(V)

## Performance

On typical HLA-Zoo graphs (30-2000 nodes):
- **SGD**: 0.1-5 seconds (4 threads)
- **Grooming**: <0.1 seconds
- **Topological sort**: <0.1 seconds
- **Total**: <10 seconds for most graphs

For larger graphs (>10k nodes), consider reducing iterations or increasing threads.

## Determinism

The algorithm is fully deterministic:
- RNG seeds are derived from fixed values (`9399220 + thread_id`)
- Graph traversals use sorted iteration
- Stable data structures (Vec for nodes, BTreeSet for ready queue)

Same input graph + same parameters = same output, always.

## Citation

This library is based on the sorting algorithms from ODGI:

> Garrison, E., Guarracino, A., Heumos, S. et al. Building pangenome graphs. *bioRxiv* (2022).
> [https://doi.org/10.1101/2022.02.14.480413](https://doi.org/10.1101/2022.02.14.480413)

Original ODGI implementation:
> [https://github.com/pangenome/odgi](https://github.com/pangenome/odgi)

## License

MIT License - see LICENSE file for details.

## Credits

Extracted from [SeqRush](https://github.com/KristopherKubicki/seqrush) by Andrea Guarracino.

Original algorithms from [ODGI](https://github.com/pangenome/odgi) by Erik Garrison and the Pangenome Graph team.

## Development

```bash
# Clone repository
git clone https://github.com/pangenome/gfasort
cd gfasort

# Build
cargo build --release

# Run tests
cargo test

# Build documentation
cargo doc --open
```

## Examples

See the [SeqRush integration](https://github.com/KristopherKubicki/seqrush) for real-world usage examples.

## Contributing

Contributions welcome! Please open issues or pull requests on GitHub.

## See Also

- [ODGI](https://github.com/pangenome/odgi) - Original implementation
- [SeqRush](https://github.com/KristopherKubicki/seqrush) - Pangenome graph construction tool
- [Variation Graph Toolkit](https://github.com/vgteam/vg) - Variation graph tools
