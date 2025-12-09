/// GFASort - Bidirected graph sorting tool
///
/// Supports flexible pipeline configuration using -p option:
///   Y = Path-guided SGD (positions nodes based on path distances)
///   g = Grooming (orients nodes consistently along paths)
///   s = Topological sort (linearizes graph respecting edge directions)
///   S = Priority topological sort (preserves SGD layout while respecting edges)
///   u = Unchop (merge linear chains of nodes)
///   L = 2D Layout (compute 2D coordinates using path-guided SGD)
///
/// Examples:
///   -p Ygs   = Full pipeline: SGD → grooming → topological sort (default)
///   -p YgS   = Full pipeline with priority topo sort (may preserve better layout)
///   -p Ygsu  = Full pipeline with unchop at the end
///   -p s     = Topological sort only
///   -p gY    = Grooming then SGD
///   -p Ys    = SGD then topological sort
///   -p g     = Grooming only
///   -p u     = Unchop only
///   -p L     = 2D layout only (outputs TSV with --layout-out)

use gfasort::graph_ops::BidirectedGraph;
use gfasort::graph::{Handle, BiPath};
use gfasort::ygs::{YgsParams, sgd_sort_only, groom_only, topological_sort_only, priority_topological_sort_only, unchop_only};
use gfasort::sgd::{LayoutSGDParams, path_linear_sgd_layout, calculate_layout_stress};
use gfasort::layout::Layout;
use clap::Parser;
use std::process;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "gfasort")]
#[command(about = "Sort a GFA file using configurable pipeline steps")]
#[command(long_about = "Sort a GFA file using configurable pipeline steps.\n\n\
Pipeline characters:\n  \
  Y = Path-guided SGD (stochastic gradient descent)\n  \
  g = Grooming (orient nodes consistently)\n  \
  s = Topological sort (currently hurts layout quality)\n  \
  S = Priority topological sort (preserves SGD layout)\n  \
  u = Unchop (merge linear chains)\n  \
  L = 2D Layout (compute nD coordinates)\n\n\
Examples:\n  \
  gfasort -i in.gfa -o out.gfa -p Yg    # SGD + groom (default, best quality)\n  \
  gfasort -i in.gfa -o out.gfa -p Ygs   # With topological sort\n  \
  gfasort -i in.gfa -o out.gfa -p Ygu   # With unchop\n  \
  gfasort -i in.gfa -o out.gfa -p s     # Topological sort only\n  \
  gfasort -i in.gfa -o out.gfa -p u     # Unchop only\n  \
  gfasort -i in.gfa -o out.gfa -p L --layout-out layout.tsv  # 2D layout")]
struct Args {
    /// Input GFA file
    #[arg(short = 'i', long)]
    input: String,

    /// Output GFA file
    #[arg(short = 'o', long)]
    output: String,

    /// Pipeline to run. Characters: Y=SGD, g=groom, s=topo-sort, S=priority-topo-sort, u=unchop, L=layout.
    /// Executed left-to-right. Default: sYgs (matches ODGI's standard workflow)
    #[arg(short = 'p', long, default_value = "sYgs")]
    pipeline: String,

    /// Number of SGD iterations (only used if Y is in pipeline)
    #[arg(long, default_value = "100")]
    iter_max: usize,

    /// Number of threads (only used if Y or L is in pipeline)
    #[arg(short = 't', long, default_value = "1")]
    threads: usize,

    /// Verbosity level (0=none, 1=basic, 2=detailed)
    #[arg(short = 'v', long, default_value = "1")]
    verbose: u8,

    /// Number of dimensions for layout (only used if L is in pipeline)
    #[arg(long, default_value = "2")]
    dimensions: usize,

    /// Output file for layout coordinates in TSV format (only used if L is in pipeline)
    #[arg(long)]
    layout_out: Option<String>,

    /// Number of layout iterations (only used if L is in pipeline)
    #[arg(long, default_value = "30")]
    layout_iter: usize,
}

fn parse_gfa(content: &str) -> Result<BidirectedGraph, String> {
    let mut graph = BidirectedGraph::new();

    // Parse S lines (segments/nodes)
    // IMPORTANT: Use add_node to properly populate node_order with GFA file order
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

    // Parse L lines (links/edges)
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

    // Parse P lines (paths)
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

fn validate_pipeline(pipeline: &str) -> Result<(), String> {
    for c in pipeline.chars() {
        match c {
            'Y' | 'g' | 's' | 'S' | 'u' | 'L' => {}
            _ => return Err(format!("Unknown pipeline character '{}'. Valid: Y (SGD), g (groom), s (topo-sort), S (priority-topo-sort), u (unchop), L (layout)", c)),
        }
    }
    if pipeline.is_empty() {
        return Err("Pipeline cannot be empty".to_string());
    }
    Ok(())
}

fn main() {
    let args = Args::parse();

    // Validate pipeline string
    if let Err(e) = validate_pipeline(&args.pipeline) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }

    if args.verbose >= 1 {
        eprintln!("[gfasort] reading {}", args.input);
    }

    // Read and parse the GFA file
    let content = match std::fs::read_to_string(&args.input) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            process::exit(1);
        }
    };

    let mut graph = match parse_gfa(&content) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing GFA: {}", e);
            process::exit(1);
        }
    };

    if args.verbose >= 1 {
        eprintln!("[gfasort] loaded {} nodes, {} edges, {} paths",
                 graph.node_count(), graph.edges.len(), graph.paths.len());
    }

    if args.verbose >= 2 {
        eprintln!("[gfasort] pipeline: {}", args.pipeline);
    }

    // Build SGD params once (in case Y is used)
    let ygs_params = YgsParams::from_graph(&graph, args.verbose, args.threads);
    let mut sgd_params = ygs_params.path_sgd.clone();
    sgd_params.iter_max = args.iter_max as u64;

    // Build layout params once (in case L is used)
    let mut layout_params = LayoutSGDParams::from_graph(&graph, args.dimensions, args.threads);
    layout_params.iter_max = args.layout_iter as u64;
    layout_params.progress = args.verbose >= 2;

    // Store layout if computed
    let mut layout: Option<Layout> = None;

    // Execute pipeline steps in order
    for (step_num, c) in args.pipeline.chars().enumerate() {
        if args.verbose >= 1 {
            let step_name = match c {
                'Y' => "SGD",
                'g' => "groom",
                's' => "topo-sort",
                'S' => "priority-topo-sort",
                'u' => "unchop",
                'L' => &format!("{}D layout", args.dimensions),
                _ => "?",
            };
            eprintln!("[gfasort] [{}/{}] {}", step_num + 1, args.pipeline.len(), step_name);
        }

        match c {
            'Y' => {
                sgd_sort_only(&mut graph, sgd_params.clone(), args.verbose);
            }
            'g' => {
                groom_only(&mut graph, args.verbose);
            }
            's' => {
                topological_sort_only(&mut graph, args.verbose);
            }
            'S' => {
                priority_topological_sort_only(&mut graph, args.verbose);
            }
            'u' => {
                unchop_only(&mut graph, args.verbose);
            }
            'L' => {
                let graph_arc = Arc::new(graph.clone());
                layout = Some(path_linear_sgd_layout(graph_arc.clone(), layout_params.clone()));

                // Calculate and report stress
                if args.verbose >= 1 {
                    let stress = calculate_layout_stress(&graph, layout.as_ref().unwrap(), 10000);
                    eprintln!("[gfasort] layout stress: {:.6}", stress);
                }
            }
            _ => unreachable!(), // Already validated
        }
    }

    // Write layout if computed and output path specified
    if let Some(ref layout) = layout {
        if let Some(ref layout_path) = args.layout_out {
            if args.verbose >= 1 {
                eprintln!("[gfasort] writing layout to {}", layout_path);
            }
            let mut file = match std::fs::File::create(layout_path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Error creating layout file: {}", e);
                    process::exit(1);
                }
            };
            if let Err(e) = layout.write_tsv(&mut file) {
                eprintln!("Error writing layout: {}", e);
                process::exit(1);
            }
        } else if args.verbose >= 1 {
            eprintln!("[gfasort] warning: layout computed but --layout-out not specified");
        }
    }

    if args.verbose >= 1 {
        eprintln!("[gfasort] writing {}", args.output);
    }

    // Write the sorted graph
    let mut output_buffer = Vec::new();
    if let Err(e) = graph.write_gfa(&mut output_buffer) {
        eprintln!("Error writing GFA: {}", e);
        process::exit(1);
    }

    if let Err(e) = std::fs::write(&args.output, output_buffer) {
        eprintln!("Error writing output file: {}", e);
        process::exit(1);
    }

    if args.verbose >= 1 {
        eprintln!("[gfasort] done");
    }
}
