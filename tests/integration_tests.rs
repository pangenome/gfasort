use gfasort::*;
use std::path::Path;

#[test]
fn test_load_simple_gfa() {
    let test_file = "tests/data/simple.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let graph = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");

    println!("Loaded graph: {} nodes, {} edges, {} paths",
             graph.node_count(), graph.edges.len(), graph.paths.len());

    assert!(graph.node_count() > 0, "Graph should have nodes");
    assert!(!graph.edges.is_empty(), "Graph should have edges");
}

#[test]
fn test_ygs_sort_simple() {
    let test_file = "tests/data/simple.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let mut graph = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");

    let original_node_count = graph.node_count();
    let original_edge_count = graph.edges.len();

    println!("Before sorting: {} nodes, {} edges",
             original_node_count, original_edge_count);

    // Sort with Ygs pipeline
    let params = YgsParams::from_graph(&graph, 2, 2);
    ygs_sort(&mut graph, &params);

    println!("After sorting: {} nodes, {} edges",
             graph.node_count(), graph.edges.len());

    // Graph structure should be preserved
    assert_eq!(graph.node_count(), original_node_count,
               "Node count should not change");
    assert_eq!(graph.edges.len(), original_edge_count,
               "Edge count should not change");
}

#[test]
fn test_ygs_determinism() {
    let test_file = "tests/data/simple.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let graph1 = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");
    let graph2 = graph1.clone();

    let mut sorted1 = graph1.clone();
    let mut sorted2 = graph2.clone();

    let params = YgsParams::from_graph(&sorted1, 0, 2);

    ygs_sort(&mut sorted1, &params);
    ygs_sort(&mut sorted2, &params);

    // Compare node orders
    for (i, (n1, n2)) in sorted1.nodes.iter().zip(sorted2.nodes.iter()).enumerate() {
        match (n1, n2) {
            (Some(node1), Some(node2)) => {
                assert_eq!(node1.id, node2.id,
                          "Node IDs should match at position {}", i);
                assert_eq!(node1.sequence, node2.sequence,
                          "Sequences should match at position {}", i);
            }
            (None, None) => {}
            _ => panic!("Node presence mismatch at position {}", i),
        }
    }

    println!("✓ Determinism test passed - both runs produced identical results");
}

#[test]
fn test_sgd_only() {
    let test_file = "tests/data/simple.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let mut graph = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");

    let params = YgsParams::from_graph(&graph, 2, 2);

    // Run only SGD
    sgd_sort_only(&mut graph, params.path_sgd, 2);

    assert!(graph.node_count() > 0, "Graph should still have nodes after SGD");
    println!("✓ SGD completed successfully");
}

#[test]
fn test_groom_only() {
    let test_file = "tests/data/simple.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let mut graph = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");

    // Run only grooming
    groom_only(&mut graph, 2);

    assert!(graph.node_count() > 0, "Graph should still have nodes after grooming");
    println!("✓ Grooming completed successfully");
}

#[test]
fn test_topological_sort_only() {
    let test_file = "tests/data/simple.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let mut graph = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");

    // Run only topological sort
    topological_sort_only(&mut graph, 2);

    assert!(graph.node_count() > 0, "Graph should still have nodes after topo sort");
    println!("✓ Topological sort completed successfully");
}

#[test]
fn test_drb1_graph() {
    let test_file = "tests/data/DRB1-3123.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let mut graph = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");

    println!("DRB1 graph: {} nodes, {} edges, {} paths",
             graph.node_count(), graph.edges.len(), graph.paths.len());

    let original_nodes = graph.node_count();

    // Sort with reduced iterations for faster testing
    let mut params = YgsParams::from_graph(&graph, 2, 4);
    params.path_sgd.iter_max = 10; // Reduce iterations for testing

    ygs_sort(&mut graph, &params);

    assert_eq!(graph.node_count(), original_nodes,
               "Node count should be preserved");

    println!("✓ DRB1 graph sorted successfully");
}

#[test]
fn test_write_and_reload() {
    use tempfile::NamedTempFile;

    let test_file = "tests/data/simple.gfa";
    if !Path::new(test_file).exists() {
        eprintln!("Skipping test - file not found: {}", test_file);
        return;
    }

    let mut graph = gfa_parser::load_gfa(Path::new(test_file))
        .expect("Failed to load GFA");

    // Sort the graph
    let params = YgsParams::from_graph(&graph, 0, 2);
    ygs_sort(&mut graph, &params);

    // Write to temporary file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    gfa_parser::write_gfa(&graph, temp_file.path())
        .expect("Failed to write GFA");

    // Reload and compare
    let reloaded = gfa_parser::load_gfa(temp_file.path())
        .expect("Failed to reload GFA");

    assert_eq!(graph.node_count(), reloaded.node_count(),
               "Node counts should match after reload");
    assert_eq!(graph.edges.len(), reloaded.edges.len(),
               "Edge counts should match after reload");

    println!("✓ Write and reload test passed");
}
