# GFASort

A Rust library and CLI tool for sorting bidirected pangenome graphs using configurable pipelines (path-guided SGD, grooming, topological sort), compatible with ODGI's behavior.

## Features

- **Path-guided SGD (Y)**: Positions nodes to minimize discrepancy between path distances and layout distances
- **Grooming (g)**: Ensures consistent node orientations along paths
- **Topological Sort (s)**: Produces valid linearization respecting edge directions
- **Unchop (u)**: Merges linear chains of nodes into single nodes
- **nD Layout (L)**: Compute 2D, 3D, or nD coordinates for graph visualization
- **Flexible Pipeline**: Run any combination of steps in any order (`-p Ygs`, `-p Ygsu`, `-p L`, etc.)
- **Multi-threaded**: Parallel SGD and layout with configurable thread count

## Installation

```bash
# Clone and build
git clone https://github.com/pangenome/gfasort
cd gfasort
cargo build --release

# Binary will be at target/release/gfasort
```

## CLI Usage

```bash
# Full pipeline (default): SGD -> grooming -> topological sort
gfasort -i input.gfa -o output.gfa -p Ygs

# Full pipeline with unchop at the end
gfasort -i input.gfa -o output.gfa -p Ygsu

# Topological sort only
gfasort -i input.gfa -o output.gfa -p s

# Unchop only (merge linear chains)
gfasort -i input.gfa -o output.gfa -p u

# Grooming then SGD
gfasort -i input.gfa -o output.gfa -p gY

# With options
gfasort -i input.gfa -o output.gfa -p Ygs -t 4 --iter-max 200 -v 2

# 2D layout (outputs TSV coordinates)
gfasort -i input.gfa -o output.gfa -p L --layout-out layout.tsv

# 3D layout
gfasort -i input.gfa -o output.gfa -p L --layout-out layout.tsv --dimensions 3

# Sort then compute layout
gfasort -i input.gfa -o output.gfa -p sYgL --layout-out layout.tsv

# Layout with more iterations
gfasort -i input.gfa -o output.gfa -p L --layout-out layout.tsv --layout-iter 50 -t 4
```

### Pipeline Characters

| Char | Step | Description |
|------|------|-------------|
| `Y` | Path-guided SGD | Stochastic gradient descent using path distances (1D) |
| `g` | Grooming | Orient nodes consistently along paths |
| `s` | Topological sort | Linearize graph respecting edge directions |
| `S` | Priority topo sort | Topological sort preserving SGD layout |
| `u` | Unchop | Merge linear chains of nodes |
| `L` | nD Layout | Compute nD coordinates for visualization |

Steps are executed left-to-right in the order specified.

### Options

```
-i, --input <FILE>       Input GFA file
-o, --output <FILE>      Output GFA file
-p, --pipeline <STR>     Pipeline steps (default: sYgs)
-t, --threads <N>        Number of threads for SGD/layout (default: 1)
    --iter-max <N>       SGD iterations (default: 100)
-v, --verbose <N>        Verbosity level: 0=none, 1=basic (default), 2=detailed

Layout options (only used if L is in pipeline):
    --layout-out <FILE>  Output TSV file for layout coordinates
    --dimensions <N>     Number of dimensions: 2, 3, etc. (default: 2)
    --layout-iter <N>    Layout iterations (default: 30)
```

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
# With CLI binaries (default)
gfasort = { git = "https://github.com/pangenome/gfasort" }

# Library only (no clap dependency)
gfasort = { git = "https://github.com/pangenome/gfasort", default-features = false }
```

### Example

```rust
use gfasort::{BidirectedGraph, YgsParams, ygs_sort, Handle, BiPath};
use gfasort::ygs::{sgd_sort_only, groom_only, topological_sort_only, unchop_only};

// Build graph
let mut graph = BidirectedGraph::new();
graph.add_node(1, b"ACGT".to_vec());
graph.add_node(2, b"TGCA".to_vec());
graph.add_node(3, b"AAAA".to_vec());

// Add edges
graph.add_edge(Handle::forward(1), Handle::forward(2));
graph.add_edge(Handle::forward(2), Handle::forward(3));

// Add a path
let mut path = BiPath::new("path1".to_string());
path.add_step(Handle::forward(1));
path.add_step(Handle::forward(2));
path.add_step(Handle::forward(3));
graph.paths.push(path);

// Option 1: Full Ygs pipeline
let params = YgsParams::from_graph(&graph, 0, 4);  // verbose=0
ygs_sort(&mut graph, &params);

// Option 2: Individual steps (verbose: 0=none, 1=basic, 2=detailed)
groom_only(&mut graph, 0);
sgd_sort_only(&mut graph, params.path_sgd.clone(), 0);
topological_sort_only(&mut graph, 0);
unchop_only(&mut graph, 0);
```

### Layout Example

```rust
use gfasort::{BidirectedGraph, LayoutSGDParams, path_linear_sgd_layout, Layout};
use std::sync::Arc;

// Build or load graph...
let graph = Arc::new(graph);

// Compute 2D layout
let params = LayoutSGDParams::from_graph(&graph, 2, 4);  // 2D, 4 threads
let layout: Layout = path_linear_sgd_layout(graph.clone(), params);

// Access coordinates
for node in 0..layout.num_nodes {
    let (x_plus, y_plus) = (layout.x_plus(node), layout.y_plus(node));
    let (x_minus, y_minus) = (layout.x_minus(node), layout.y_minus(node));
    println!("Node {}: +({}, {}) -({}, {})", node, x_plus, y_plus, x_minus, y_minus);
}

// Write to TSV
let mut file = std::fs::File::create("layout.tsv").unwrap();
layout.write_tsv(&mut file).unwrap();
```

## Algorithm Details

### Path-Guided SGD (Y)

Optimizes node positions to match distances along paths:

1. Initialize positions based on current graph order
2. Calculate learning rate schedule (exponential decay)
3. Multi-threaded optimization:
   - Sample random pairs of nodes along paths
   - Use Zipfian distribution for neighbor selection (favors nearby nodes)
   - Update positions to minimize `|layout_distance - path_distance|`
   - Adaptive cooling: switches to local optimization after 50% of iterations

### Grooming (g)

Ensures consistent node orientations:

1. Analyze path-based orientation preferences
2. BFS traversal from head nodes
3. Flip nodes reached via reverse orientation

### Topological Sort (s)

Modified Kahn's algorithm for bidirected graphs:

1. Start with head nodes (no incoming edges)
2. Process nodes in priority order
3. Break cycles by choosing lowest node ID

### nD Layout (L)

Computes 2D, 3D, or arbitrary nD coordinates using path-guided SGD (port of ODGI's `odgi layout`):

1. Initialize: X dimension from node rank, other dimensions from Gaussian noise
2. Track both + and - ends of each node (for proper edge rendering)
3. Sample random node pairs along paths using Zipfian distribution
4. Update positions to minimize `|layout_distance - path_distance|` using Euclidean distance
5. Learning rate schedule with cooling phase for local refinement

Output TSV format:
```
idx     x+      y+      x-      y-
0       100.5   200.3   101.2   199.8
1       150.2   180.1   151.0   179.5
...
```

Each row contains coordinates for both the + and - ends of a node, enabling proper edge rendering in visualization tools.

## Additional Tools

```bash
# Analyze SGD behavior
sgd_diagnostics input.gfa

# Measure layout quality
measure_layout_quality input.gfa

# Compare gfasort and ODGI layouts
compare_layouts input.gfa gfasort_layout.tsv odgi_layout.tsv
```

## Determinism

The algorithm is fully deterministic:
- RNG seeds derived from fixed values
- Graph traversals use sorted iteration
- Same input + same parameters = same output

## Citation

Based on sorting algorithms from [ODGI](https://github.com/pangenome/odgi):

> Guarracino A, Heumos S, Naber F, Panber P, Kelleher J, Garrison E. ODGI: understanding pangenome graphs. *Bioinformatics*. 2022;38(13):3319-3326. https://doi.org/10.1093/bioinformatics/btac308

## License

MIT License
