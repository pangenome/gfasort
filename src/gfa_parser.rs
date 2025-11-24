/// Simple GFA parser for testing
use crate::graph::{BiPath, Handle};
use crate::graph_ops::BidirectedGraph;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn load_gfa(path: &Path) -> Result<BidirectedGraph, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    let mut graph = BidirectedGraph::new();
    let mut node_id_map: HashMap<String, usize> = HashMap::new();
    let mut next_id = 1usize;

    // Store links and paths for second pass
    let mut pending_links: Vec<(String, String, String, String)> = Vec::new();
    let mut pending_paths: Vec<(String, String)> = Vec::new();

    // First pass: read all segments
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('H') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.is_empty() {
            continue;
        }

        match fields[0] {
            "S" => {
                // Segment: S <id> <sequence>
                if fields.len() < 3 {
                    continue;
                }
                let name = fields[1].to_string();
                let sequence = fields[2].as_bytes().to_vec();

                let node_id = *node_id_map.entry(name).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });

                graph.add_node(node_id, sequence);
            }
            "L" => {
                // Link: L <from> <from_orient> <to> <to_orient> <overlap>
                if fields.len() < 5 {
                    continue;
                }
                pending_links.push((
                    fields[1].to_string(),
                    fields[2].to_string(),
                    fields[3].to_string(),
                    fields[4].to_string(),
                ));
            }
            "P" => {
                // Path: P <name> <node_list> <overlaps>
                if fields.len() < 3 {
                    continue;
                }
                pending_paths.push((fields[1].to_string(), fields[2].to_string()));
            }
            _ => {
                // Ignore other line types
            }
        }
    }

    // Second pass: add links
    for (from_name, from_orient, to_name, to_orient) in pending_links {
        let from_id = node_id_map.get(&from_name)
            .ok_or_else(|| format!("Unknown node in link: {}", from_name))?;
        let to_id = node_id_map.get(&to_name)
            .ok_or_else(|| format!("Unknown node in link: {}", to_name))?;

        let from_handle = if from_orient == "+" {
            Handle::forward(*from_id)
        } else {
            Handle::reverse(*from_id)
        };

        let to_handle = if to_orient == "+" {
            Handle::forward(*to_id)
        } else {
            Handle::reverse(*to_id)
        };

        graph.add_edge(from_handle, to_handle);
    }

    // Third pass: add paths
    for (path_name, node_list) in pending_paths {
        let mut path = BiPath::new(path_name);

        for step in node_list.split(',') {
            let step = step.trim();
            if step.is_empty() {
                continue;
            }

            let (node_name, orient) = if step.ends_with('+') {
                (&step[..step.len()-1], '+')
            } else if step.ends_with('-') {
                (&step[..step.len()-1], '-')
            } else {
                continue;
            };

            if let Some(&node_id) = node_id_map.get(node_name) {
                let handle = if orient == '+' {
                    Handle::forward(node_id)
                } else {
                    Handle::reverse(node_id)
                };
                path.add_step(handle);
            }
        }

        if !path.steps.is_empty() {
            graph.paths.push(path);
        }
    }

    Ok(graph)
}

pub fn write_gfa(graph: &BidirectedGraph, path: &Path) -> Result<(), String> {
    use std::io::Write;

    let mut file = File::create(path)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    // Write header
    writeln!(file, "H\tVN:Z:1.0")
        .map_err(|e| format!("Write error: {}", e))?;

    // Write segments (nodes)
    for (id, node_opt) in graph.nodes.iter().enumerate() {
        if let Some(node) = node_opt {
            let seq = String::from_utf8_lossy(&node.sequence);
            writeln!(file, "S\t{}\t{}", id, seq)
                .map_err(|e| format!("Write error: {}", e))?;
        }
    }

    // Write links (edges)
    let mut edges_vec: Vec<_> = graph.edges.iter().collect();
    edges_vec.sort();
    for edge in edges_vec {
        let from_id = edge.from.node_id();
        let from_orient = if edge.from.is_reverse() { '-' } else { '+' };
        let to_id = edge.to.node_id();
        let to_orient = if edge.to.is_reverse() { '-' } else { '+' };

        writeln!(file, "L\t{}\t{}\t{}\t{}\t0M",
                 from_id, from_orient, to_id, to_orient)
            .map_err(|e| format!("Write error: {}", e))?;
    }

    // Write paths
    for path in &graph.paths {
        let steps: Vec<String> = path.steps.iter()
            .map(|h| format!("{}{}", h.node_id(), if h.is_reverse() { '-' } else { '+' }))
            .collect();
        let overlaps = vec!["0M"; path.steps.len().saturating_sub(1)].join(",");

        writeln!(file, "P\t{}\t{}\t{}",
                 path.name,
                 steps.join(","),
                 overlaps)
            .map_err(|e| format!("Write error: {}", e))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_simple_gfa() {
        let test_gfa = "tests/data/simple.gfa";
        if !Path::new(test_gfa).exists() {
            eprintln!("Test file not found: {}", test_gfa);
            return;
        }

        let graph = load_gfa(Path::new(test_gfa)).expect("Failed to load GFA");

        // Should have nodes
        assert!(graph.node_count() > 0);

        // Should have edges
        assert!(!graph.edges.is_empty());

        eprintln!("Loaded graph: {} nodes, {} edges",
                 graph.node_count(), graph.edges.len());
    }
}
