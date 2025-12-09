/// Compare stress/distortion between gfasort and ODGI layouts
///
/// Usage: compare_layouts <gfa_file> <gfasort_layout.tsv> <odgi_layout.tsv>

use gfasort::graph_ops::BidirectedGraph;
use gfasort::graph::{Handle, BiPath};
use gfasort::PathIndex;
use rand::distr::{Distribution, Uniform};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::fs::File;

fn parse_gfa(content: &str) -> Result<BidirectedGraph, String> {
    let mut graph = BidirectedGraph::new();

    for line in content.lines() {
        if line.starts_with('S') {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let node_id: usize = parts[1].parse()
                    .map_err(|e| format!("Failed to parse node ID: {}", e))?;
                let sequence = parts[2].as_bytes().to_vec();
                graph.add_node(node_id, sequence);
            }
        }
    }

    for line in content.lines() {
        if line.starts_with('L') {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 5 {
                let from_id: usize = parts[1].parse()
                    .map_err(|e| format!("Failed to parse from ID: {}", e))?;
                let from_orient = parts[2];
                let to_id: usize = parts[3].parse()
                    .map_err(|e| format!("Failed to parse to ID: {}", e))?;
                let to_orient = parts[4];

                let from_handle = if from_orient == "+" {
                    Handle::forward(from_id)
                } else {
                    Handle::reverse(from_id)
                };

                let to_handle = if to_orient == "+" {
                    Handle::forward(to_id)
                } else {
                    Handle::reverse(to_id)
                };

                graph.add_edge(from_handle, to_handle);
            }
        }
    }

    for line in content.lines() {
        if line.starts_with('P') {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let path_name = parts[1].to_string();
                let mut path = BiPath::new(path_name);

                for step_str in parts[2].split(',') {
                    let step = step_str.trim();
                    if step.is_empty() {
                        continue;
                    }

                    let orient = step.chars().last().unwrap();
                    let node_id: usize = step[..step.len()-1].parse()
                        .map_err(|e| format!("Failed to parse path node ID: {}", e))?;

                    let handle = if orient == '+' {
                        Handle::forward(node_id)
                    } else {
                        Handle::reverse(node_id)
                    };

                    path.steps.push(handle);
                }

                graph.paths.push(path);
            }
        }
    }

    Ok(graph)
}

/// Load gfasort layout (idx, x+, y+, x-, y- format)
fn load_gfasort_layout(path: &str) -> Result<HashMap<usize, (f64, f64)>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);
    let mut layout = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        if i == 0 || line.trim().is_empty() {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            let idx: usize = parts[0].parse()
                .map_err(|e| format!("Failed to parse index: {}", e))?;
            let x: f64 = parts[1].parse()
                .map_err(|e| format!("Failed to parse x: {}", e))?;
            let y: f64 = parts[2].parse()
                .map_err(|e| format!("Failed to parse y: {}", e))?;
            layout.insert(idx, (x, y));
        }
    }

    Ok(layout)
}

/// Load ODGI layout (idx, X, Y, component format)
/// ODGI outputs 2 rows per node: row 2*i is + end, row 2*i+1 is - end
/// We only use the + end (even rows) for stress comparison
fn load_odgi_layout(path: &str) -> Result<HashMap<usize, (f64, f64)>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);
    let mut layout = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        if i == 0 || line.trim().is_empty() {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            let idx: usize = parts[0].parse()
                .map_err(|e| format!("Failed to parse index: {}", e))?;

            // ODGI outputs 2 rows per node (+ and - ends)
            // Row idx is for: node = idx/2, end = idx%2 (0=+, 1=-)
            // We only use + end (even rows)
            if idx % 2 == 0 {
                let node_idx = idx / 2;
                let x: f64 = parts[1].parse()
                    .map_err(|e| format!("Failed to parse x: {}", e))?;
                let y: f64 = parts[2].parse()
                    .map_err(|e| format!("Failed to parse y: {}", e))?;
                layout.insert(node_idx, (x, y));
            }
        }
    }

    Ok(layout)
}

/// Calculate layout stress using path distances
fn calculate_stress(
    graph: &BidirectedGraph,
    layout: &HashMap<usize, (f64, f64)>,
    sample_count: usize,
) -> f64 {
    let path_index = PathIndex::from_graph(graph);

    // Build node ID to index mapping (same as gfasort's SGD)
    let node_ids: Vec<usize> = if !graph.node_order.is_empty() {
        graph.node_order.clone()
    } else {
        let mut ids: Vec<_> = graph.nodes.iter().enumerate()
            .filter_map(|(id, n)| if n.is_some() { Some(id) } else { None })
            .collect();
        ids.sort();
        ids
    };

    let mut handle_to_idx: HashMap<Handle, usize> = HashMap::new();
    for (idx, node_id) in node_ids.iter().enumerate() {
        handle_to_idx.insert(Handle::forward(*node_id), idx);
    }

    let mut rng = Xoshiro256Plus::seed_from_u64(12345);
    let total_steps = path_index.get_total_steps();
    if total_steps < 2 {
        return 0.0;
    }

    let step_dist = Uniform::new(0, total_steps).unwrap();

    let mut stress_sum = 0.0;
    let mut count = 0u64;

    for _ in 0..sample_count {
        let step_a = step_dist.sample(&mut rng);
        let path_idx = path_index.get_path_of_step(step_a);
        let path_step_count = path_index.get_path_step_count(path_idx);

        if path_step_count < 2 {
            continue;
        }

        let rank_a = path_index.get_rank_of_step(step_a);
        let rank_dist = Uniform::new(0, path_step_count).unwrap();
        let rank_b = rank_dist.sample(&mut rng);

        if rank_a == rank_b {
            continue;
        }

        let step_a_idx = path_index.get_step_at_path_position(path_idx, rank_a);
        let step_b_idx = path_index.get_step_at_path_position(path_idx, rank_b);

        let handle_a = path_index.get_handle_of_step(step_a_idx);
        let handle_b = path_index.get_handle_of_step(step_b_idx);

        let pos_a = path_index.get_position_of_step(step_a_idx) as f64;
        let pos_b = path_index.get_position_of_step(step_b_idx) as f64;
        let path_dist = (pos_a - pos_b).abs();

        if path_dist == 0.0 {
            continue;
        }

        let idx_a = match handle_to_idx.get(&Handle::forward(handle_a.node_id())).copied() {
            Some(idx) => idx,
            None => continue,
        };
        let idx_b = match handle_to_idx.get(&Handle::forward(handle_b.node_id())).copied() {
            Some(idx) => idx,
            None => continue,
        };

        // Get layout coordinates
        let (xa, ya) = match layout.get(&idx_a) {
            Some(&coords) => coords,
            None => continue,
        };
        let (xb, yb) = match layout.get(&idx_b) {
            Some(&coords) => coords,
            None => continue,
        };

        let dx = xa - xb;
        let dy = ya - yb;
        let layout_dist = (dx * dx + dy * dy).sqrt();

        // Normalized stress: (d_layout - d_path)^2 / d_path^2
        let error = layout_dist - path_dist;
        stress_sum += (error * error) / (path_dist * path_dist);
        count += 1;
    }

    if count > 0 {
        (stress_sum / count as f64).sqrt()
    } else {
        0.0
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <gfa_file> <gfasort_layout.tsv> <odgi_layout.tsv>", args[0]);
        std::process::exit(1);
    }

    let gfa_path = &args[1];
    let gfasort_layout_path = &args[2];
    let odgi_layout_path = &args[3];

    // Load graph
    let content = std::fs::read_to_string(gfa_path)
        .expect("Failed to read GFA file");
    let graph = parse_gfa(&content).expect("Failed to parse GFA");

    eprintln!("Loaded graph: {} nodes, {} edges, {} paths",
             graph.node_count(), graph.edges.len(), graph.paths.len());

    // Load layouts
    let gfasort_layout = load_gfasort_layout(gfasort_layout_path)
        .expect("Failed to load gfasort layout");
    let odgi_layout = load_odgi_layout(odgi_layout_path)
        .expect("Failed to load ODGI layout");

    eprintln!("Loaded gfasort layout: {} nodes", gfasort_layout.len());
    eprintln!("Loaded ODGI layout: {} nodes", odgi_layout.len());

    // Calculate stress for both
    let sample_count = 100000;
    let gfasort_stress = calculate_stress(&graph, &gfasort_layout, sample_count);
    let odgi_stress = calculate_stress(&graph, &odgi_layout, sample_count);

    println!("Stress comparison ({} samples):", sample_count);
    println!("  gfasort: {:.6}", gfasort_stress);
    println!("  ODGI:    {:.6}", odgi_stress);
    println!("  ratio (gfasort/ODGI): {:.4}", gfasort_stress / odgi_stress);
}
