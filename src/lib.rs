//! # GFASort
//!
//! A Rust library for sorting bidirected pangenome graphs using the Ygs algorithm
//! (path-guided SGD + grooming + topological sort), exactly matching ODGI's behavior.
//!
//! ## Features
//!
//! - **Path-guided SGD**: Positions nodes to minimize discrepancy between path
//!   distances and layout distances
//! - **Grooming**: Ensures consistent node orientations along paths
//! - **Topological Sort**: Produces valid linearization respecting edge directions
//! - **Deterministic**: Same input always produces same output
//! - **Multi-threaded**: Parallel SGD with configurable thread count
//! - **ODGI-compatible**: Exact port of ODGI's sorting algorithms
//!
//! ## Quick Start
//!
//! ```rust
//! use gfasort::{BidirectedGraph, YgsParams, ygs_sort};
//!
//! // Build graph
//! let mut graph = BidirectedGraph::new();
//! graph.add_node(1, b"ACGT".to_vec());
//! // ... add more nodes, edges, paths ...
//!
//! // Sort with default parameters
//! let params = YgsParams::from_graph(&graph, false, 4);
//! ygs_sort(&mut graph, &params);
//! ```

// Core modules
mod graph;
mod graph_ops;
mod legacy_graph_ops;
mod ygs;
mod sgd;
mod groom;

// Public API - Graph structures
pub use graph::{Handle, BiNode, BiEdge, BiPath, reverse_complement};
pub use graph_ops::BidirectedGraph;

// Public API - Sorting algorithms
pub use ygs::{YgsParams, ygs_sort, sgd_sort_only, groom_only, topological_sort_only};
pub use sgd::{PathSGDParams, PathIndex, path_sgd_sort, path_linear_sgd};
