/// Exact implementation of ODGI's path_linear_sgd from path_sgd.cpp
/// This is the algorithm used by `odgi sort -p Ygs`
use crate::graph_ops::BidirectedGraph;
use crate::graph::Handle;
use rand::distr::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread;
use std::time::Duration;

/// Path index structure - simplified version of ODGI's XP
pub struct PathIndex {
    /// For each step, the handle it refers to
    step_to_handle: Vec<Handle>,
    /// For each step, its position in the path (in bp)
    step_to_position: Vec<usize>,
    /// For each step, which path it belongs to
    step_to_path: Vec<usize>,
    /// For each step, its rank in the path (0-indexed)
    step_to_rank: Vec<usize>,
    paths: Vec<PathInfo>,
}

#[derive(Clone)]
struct PathInfo {
    step_count: usize,
    length: usize,       // in bp
    first_step: usize,   // index in step arrays
}

impl PathIndex {
    pub fn from_graph(graph: &BidirectedGraph) -> Self {
        let mut step_to_handle = Vec::new();
        let mut step_to_position = Vec::new();
        let mut step_to_path = Vec::new();
        let mut step_to_rank = Vec::new();
        let mut paths = Vec::new();

        for (path_idx, path) in graph.paths.iter().enumerate() {
            let first_step = step_to_handle.len();
            let mut position = 0;

            for (rank, &handle) in path.steps.iter().enumerate() {
                step_to_handle.push(handle);
                step_to_position.push(position);
                step_to_path.push(path_idx);
                step_to_rank.push(rank);

                // Add node length to position
                if let Some(node) = graph.nodes.get(handle.node_id()).and_then(|n| n.as_ref()) {
                    position += node.sequence.len();
                }
            }

            paths.push(PathInfo {
                step_count: path.steps.len(),
                length: position,
                first_step,
            });
        }

        PathIndex {
            step_to_handle,
            step_to_position,
            step_to_path,
            step_to_rank,
            paths,
        }
    }

    pub fn get_total_steps(&self) -> usize {
        self.step_to_handle.len()
    }

    pub fn get_handle_of_step(&self, step_idx: usize) -> Handle {
        self.step_to_handle[step_idx]
    }

    pub fn get_position_of_step(&self, step_idx: usize) -> usize {
        self.step_to_position[step_idx]
    }

    pub fn get_path_of_step(&self, step_idx: usize) -> usize {
        self.step_to_path[step_idx]
    }

    pub fn get_rank_of_step(&self, step_idx: usize) -> usize {
        self.step_to_rank[step_idx]
    }

    pub fn get_path_step_count(&self, path_idx: usize) -> usize {
        self.paths[path_idx].step_count
    }

    pub fn get_step_at_path_position(&self, path_idx: usize, rank: usize) -> usize {
        self.paths[path_idx].first_step + rank
    }

    pub fn num_paths(&self) -> usize {
        self.paths.len()
    }

    pub fn get_path_length(&self, path_idx: usize) -> usize {
        self.paths[path_idx].length
    }
}

/// Dirty Zipfian distribution with O(1) sampling
/// This is an exact port of ODGI's dirty_zipfian_int_distribution using the
/// Gray-Menasce-Blakeley method from "Quickly Generating Billion-Record
/// Synthetic Databases", SIGMOD 1994
struct DirtyZipfian {
    min: u64,
    max: u64,
    theta: f64,
    zeta: f64,
    zeta2theta: f64,  // zeta(2, theta) - precomputed for efficiency
}

impl DirtyZipfian {
    fn new(min: u64, max: u64, theta: f64, zeta: f64, zeta2theta: f64) -> Self {
        DirtyZipfian { min, max, theta, zeta, zeta2theta }
    }

    /// O(1) sampling using the Gray-Menasce formula
    fn sample(&self, rng: &mut impl Rng) -> u64 {
        let n = self.max - self.min + 1;

        // Precompute alpha and eta for the formula
        let alpha = 1.0 / (1.0 - self.theta);
        let eta = (1.0 - fast_precise_pow(2.0 / n as f64, 1.0 - self.theta))
                / (1.0 - self.zeta2theta / self.zeta);

        let u: f64 = rng.random();
        let uz = u * self.zeta;

        // Fast path for most common values
        if uz < 1.0 {
            return self.min;
        }
        if uz < 1.0 + fast_precise_pow(0.5, self.theta) {
            return self.min + 1;
        }

        // General case using inverse CDF approximation
        let result = self.min as f64 + (n as f64 * fast_precise_pow(eta * u - eta + 1.0, alpha));
        (result as u64).min(self.max)
    }
}

/// Fast approximate power function - exact port of ODGI's fast_precise_pow
/// Uses union-based bit manipulation on the high 32 bits of an IEEE 754 double
fn fast_precise_pow(a: f64, b: f64) -> f64 {
    // Extract integer part of exponent
    let e = b as i32;

    // Approximate a^(b-e) using bit manipulation on the high 32 bits
    // The magic number 1072632447 = 0x3FF00000 = (1023 << 20)
    // where 1023 is the IEEE 754 exponent bias
    let bits = a.to_bits();
    let high = (bits >> 32) as i32;
    let new_high = ((b - e as f64) * (high - 1072632447) as f64 + 1072632447.0) as i32;
    // Set low 32 bits to 0 (same as ODGI: u.x[0] = 0)
    let frac_bits = (new_high as u64) << 32;
    let frac_result = f64::from_bits(frac_bits);

    // Exponentiation by squaring with the integer part
    let mut base = a;
    let mut exp = e;
    let mut r = 1.0;
    while exp != 0 {
        if exp & 1 != 0 {
            r *= base;
        }
        base *= base;
        exp >>= 1;
    }

    r * frac_result
}

/// Convert f64 to u64 bits for atomic operations
fn f64_to_u64(f: f64) -> u64 {
    f.to_bits()
}

/// Convert u64 bits to f64
fn u64_to_f64(u: u64) -> f64 {
    f64::from_bits(u)
}

/// Path-guided stochastic gradient descent parameters
#[derive(Debug, Clone)]
pub struct PathSGDParams {
    pub iter_max: u64,
    pub iter_with_max_learning_rate: u64,
    pub min_term_updates: u64,
    pub delta: f64,
    pub eps: f64,
    pub eta_max: f64,
    pub theta: f64,
    pub space: u64,
    pub space_max: u64,
    pub space_quantization_step: u64,
    pub cooling_start: f64,
    pub nthreads: usize,
    pub progress: bool,
    /// Random seed for SGD (0 = use time-based seed for non-deterministic behavior)
    pub seed: u64,
}

impl Default for PathSGDParams {
    fn default() -> Self {
        // These are ODGI's default parameters from odgi sort -p Ygs
        PathSGDParams {
            iter_max: 100,  // ODGI default
            iter_with_max_learning_rate: 0,  // ODGI default
            min_term_updates: 100,
            delta: 0.0,
            eps: 0.01,
            eta_max: 100.0,  // ODGI default (w_min = 1/100)
            theta: 0.99,  // ODGI default for Ygs
            space: 100,
            space_max: 100,
            space_quantization_step: 100,  // ODGI default is 100, not 10
            cooling_start: 0.5,  // ODGI default: last 50% of iterations are cooling phase
            nthreads: 1,
            progress: false,
            seed: 9399220,  // ODGI's default fixed seed for reproducibility
        }
    }
}

/// Main path-guided SGD implementation (exact port of ODGI's path_linear_sgd)
pub fn path_linear_sgd(
    graph: Arc<BidirectedGraph>,
    params: PathSGDParams,
) -> HashMap<usize, f64> {
    let num_nodes = graph.nodes.len();
    if num_nodes == 0 {
        return HashMap::new();
    }

    // Build path index
    let path_index = PathIndex::from_graph(&graph);

    // Check if we have any paths with more than one step
    let mut has_valid_paths = false;
    for path in &path_index.paths {
        if path.step_count > 1 {
            has_valid_paths = true;
            break;
        }
    }

    if !has_valid_paths {
        eprintln!("[path_sgd] No paths with multiple steps found");
        return HashMap::new();
    }

    // Initialize positions based on current graph order
    let x: Vec<AtomicU64> = (0..num_nodes)
        .map(|_| AtomicU64::new(0))
        .collect();

    // Seed positions with graph layout using node_order (GFA file order)
    // This preserves the ordering from the input file which may carry information
    // about the desired layout
    let mut len = 0u64;
    let mut handle_to_idx: HashMap<Handle, usize> = HashMap::new();
    let mut idx = 0;

    // Use node_order if available (preserves GFA file order), otherwise use sorted ID order
    let node_ids: Vec<usize> = if !graph.node_order.is_empty() {
        graph.node_order.clone()
    } else {
        let mut ids: Vec<_> = graph.nodes.iter().enumerate()
            .filter_map(|(id, n)| if n.is_some() { Some(id) } else { None })
            .collect();
        ids.sort();
        ids
    };

    for node_id in &node_ids {
        if let Some(Some(node)) = graph.nodes.get(*node_id) {
            let handle = Handle::forward(*node_id);
            handle_to_idx.insert(handle, idx);
            x[idx].store(f64_to_u64(len as f64), Ordering::Relaxed);
            len += node.sequence.len() as u64;
            idx += 1;
        }
    }

    // Calculate first cooling iteration
    let first_cooling_iteration = (params.cooling_start * params.iter_max as f64).floor() as u64;

    // Calculate learning rate schedule
    let w_min = 1.0 / params.eta_max;
    let w_max = 1.0;
    let etas = path_linear_sgd_schedule(
        w_min,
        w_max,
        params.iter_max,
        params.iter_with_max_learning_rate,
        params.eps,
    );

    // Pre-calculate zetas for Zipfian distribution
    let zeta_size = if params.space <= params.space_max {
        params.space as usize
    } else {
        params.space_max as usize + (params.space - params.space_max) as usize / params.space_quantization_step as usize + 1
    } + 1;

    let mut zetas = vec![0.0; zeta_size];
    let mut zeta_tmp = 0.0;
    for i in 1..=params.space {
        // ODGI uses fast_precise_pow for zeta precomputation
        zeta_tmp += fast_precise_pow(1.0 / i as f64, params.theta);
        if i <= params.space_max {
            zetas[i as usize] = zeta_tmp;
        }
        if i >= params.space_max && (i - params.space_max) % params.space_quantization_step == 0 {
            let idx = params.space_max as usize + 1 + ((i - params.space_max) / params.space_quantization_step) as usize;
            if idx < zetas.len() {
                zetas[idx] = zeta_tmp;
            }
        }
    }

    // Shared state for threads
    let x = Arc::new(x);
    let path_index = Arc::new(path_index);
    let handle_to_idx = Arc::new(handle_to_idx);
    let zetas = Arc::new(zetas);
    let etas = Arc::new(etas);

    let term_updates = Arc::new(AtomicU64::new(0));
    let iteration = Arc::new(AtomicU64::new(0));
    let eta = Arc::new(AtomicU64::new(f64_to_u64(etas[0])));
    let adj_theta = Arc::new(AtomicU64::new(f64_to_u64(params.theta)));  // Adaptive theta
    let cooling = Arc::new(AtomicBool::new(false));
    let work_todo = Arc::new(AtomicBool::new(true));
    let delta_max = Arc::new(AtomicU64::new(0));

    // Progress tracking
    if params.progress {
        eprintln!("[path_sgd] Starting with {} iterations, {} term updates per iteration",
                 params.iter_max, params.min_term_updates);
    }

    // Checker thread - monitors progress and updates learning rate
    let checker_handle = {
        let term_updates = Arc::clone(&term_updates);
        let iteration = Arc::clone(&iteration);
        let work_todo = Arc::clone(&work_todo);
        let eta = Arc::clone(&eta);
        let adj_theta = Arc::clone(&adj_theta);
        let cooling = Arc::clone(&cooling);
        let etas = Arc::clone(&etas);
        let min_term_updates = params.min_term_updates;
        let iter_max = params.iter_max;

        thread::spawn(move || {
            let mut last_reported_iter = 0u64;
            while work_todo.load(Ordering::Relaxed) {
                let curr_updates = term_updates.load(Ordering::Relaxed);

                // Check if we've done enough updates for this iteration
                if curr_updates >= min_term_updates {
                    iteration.fetch_add(1, Ordering::Relaxed);
                    let new_iter = iteration.load(Ordering::Relaxed);

                    // Debug: report progress every 10 iterations
                    if new_iter % 10 == 0 && new_iter != last_reported_iter {
                        eprintln!("[path_sgd] iteration {}/{}, updates this iter: {}",
                                 new_iter, iter_max, curr_updates);
                        last_reported_iter = new_iter;
                    }

                    if new_iter > iter_max {
                        eprintln!("[path_sgd] Completed {} iterations", new_iter);
                        work_todo.store(false, Ordering::Relaxed);
                    } else {
                        // Update learning rate
                        if (new_iter as usize) < etas.len() {
                            eta.store(f64_to_u64(etas[new_iter as usize]), Ordering::Relaxed);
                        }

                        // Check if we're in cooling phase
                        if new_iter > first_cooling_iteration {
                            adj_theta.store(f64_to_u64(0.001), Ordering::Relaxed);
                            cooling.store(true, Ordering::Relaxed);
                        }
                    }

                    // Reset term_updates for next iteration
                    term_updates.store(0, Ordering::Relaxed);
                }

                thread::sleep(Duration::from_millis(1));
            }
            eprintln!("[path_sgd] Checker thread exiting, final iteration: {}",
                     iteration.load(Ordering::Relaxed));
        })
    };

    // Worker threads
    let mut handles = vec![];

    for tid in 0..params.nthreads {
        let x = Arc::clone(&x);
        let path_index = Arc::clone(&path_index);
        let handle_to_idx = Arc::clone(&handle_to_idx);
        let zetas = Arc::clone(&zetas);
        let term_updates = Arc::clone(&term_updates);
        let work_todo = Arc::clone(&work_todo);
        let eta = Arc::clone(&eta);
        let adj_theta = Arc::clone(&adj_theta);
        let cooling = Arc::clone(&cooling);
        let delta_max = Arc::clone(&delta_max);
        let space = params.space;
        let space_max = params.space_max;
        let space_quantization_step = params.space_quantization_step;
        let base_seed = params.seed;

        let handle = thread::spawn(move || {
            // Use fixed seed like ODGI for reproducibility (seed + thread_id)
            let seed = base_seed + tid as u64;
            let mut rng = Xoshiro256Plus::seed_from_u64(seed);

            let total_steps = path_index.get_total_steps();
            let step_dist = Uniform::new(0, total_steps).unwrap();
            let flip_dist = Uniform::new(0, 2).unwrap();

            // Track local updates, batch them to global counter
            let mut term_updates_local = 0u64;

            // ODGI workers just check work_todo, no per-thread limit
            while work_todo.load(Ordering::Relaxed) {
                // Sample a random step
                let step_idx = step_dist.sample(&mut rng);
                let path_idx = path_index.get_path_of_step(step_idx);
                let path_step_count = path_index.get_path_step_count(path_idx);

                if path_step_count == 1 {
                    continue;
                }

                let rank_a = path_index.get_rank_of_step(step_idx);
                let mut rank_b = rank_a;

                // Decide how to sample the second step
                if cooling.load(Ordering::Relaxed) || flip_dist.sample(&mut rng) == 1 {
                    // Use Zipfian distribution with adaptive theta
                    let current_theta = u64_to_f64(adj_theta.load(Ordering::Relaxed));

                    if rank_a > 0 && (flip_dist.sample(&mut rng) == 1 || rank_a == path_step_count - 1) {
                        // Go backward
                        let jump_space = space.min(rank_a as u64);
                        let space_idx = if jump_space > space_max {
                            space_max as usize + ((jump_space - space_max) / space_quantization_step) as usize + 1
                        } else {
                            jump_space as usize
                        };

                        let space_idx = space_idx.min(zetas.len() - 1);
                        // Compute zeta2theta = zeta(2, theta) = 1 + 1/2^theta for current theta
                        let zeta2theta = 1.0 + fast_precise_pow(0.5, current_theta);
                        let zipf = DirtyZipfian::new(1, jump_space, current_theta, zetas[space_idx], zeta2theta);
                        let z_i = zipf.sample(&mut rng);
                        rank_b = rank_a.saturating_sub(z_i as usize);
                    } else if rank_a < path_step_count - 1 {
                        // Go forward
                        let jump_space = space.min((path_step_count - rank_a - 1) as u64);
                        let space_idx = if jump_space > space_max {
                            space_max as usize + ((jump_space - space_max) / space_quantization_step) as usize + 1
                        } else {
                            jump_space as usize
                        };

                        let space_idx = space_idx.min(zetas.len() - 1);
                        // Compute zeta2theta = zeta(2, theta) = 1 + 1/2^theta for current theta
                        let zeta2theta = 1.0 + fast_precise_pow(0.5, current_theta);
                        let zipf = DirtyZipfian::new(1, jump_space, current_theta, zetas[space_idx], zeta2theta);
                        let z_i = zipf.sample(&mut rng);
                        rank_b = (rank_a + z_i as usize).min(path_step_count - 1);
                    }
                } else {
                    // Sample randomly across the path
                    let rank_dist = Uniform::new(0, path_step_count).unwrap();
                    rank_b = rank_dist.sample(&mut rng);
                }

                if rank_a == rank_b {
                    continue;
                }

                // Get handles for the terms
                let step_a_idx = path_index.get_step_at_path_position(path_idx, rank_a);
                let step_b_idx = path_index.get_step_at_path_position(path_idx, rank_b);

                let term_i = path_index.get_handle_of_step(step_a_idx);
                let term_j = path_index.get_handle_of_step(step_b_idx);

                // Get positions in path
                let pos_a = path_index.get_position_of_step(step_a_idx) as f64;
                let pos_b = path_index.get_position_of_step(step_b_idx) as f64;

                // Calculate term distance
                let term_dist = (pos_a - pos_b).abs();
                if term_dist == 0.0 {
                    continue;
                }

                let term_weight = 1.0 / term_dist;
                let mu = u64_to_f64(eta.load(Ordering::Relaxed)) * term_weight;
                let mu = mu.min(1.0);

                // Get node indices from our mapping
                // IMPORTANT: Always use forward orientation when looking up indices,
                // since handle_to_idx only contains forward handles
                let i = match handle_to_idx.get(&Handle::forward(term_i.node_id())).copied() {
                    Some(idx) => idx,
                    None => {
                        eprintln!("[path_sgd] WARNING: Handle {} not in handle_to_idx!", term_i.node_id());
                        continue;
                    }
                };
                let j = match handle_to_idx.get(&Handle::forward(term_j.node_id())).copied() {
                    Some(idx) => idx,
                    None => {
                        eprintln!("[path_sgd] WARNING: Handle {} not in handle_to_idx!", term_j.node_id());
                        continue;
                    }
                };

                // Calculate position difference
                let x_i = u64_to_f64(x[i].load(Ordering::Relaxed));
                let x_j = u64_to_f64(x[j].load(Ordering::Relaxed));
                let mut dx = x_i - x_j;

                // ODGI uses epsilon to avoid NaN, not continue
                if dx == 0.0 {
                    dx = 1e-9;
                }

                // Calculate update
                let mag = dx.abs();
                let delta_update = mu * (mag - term_dist) / 2.0;

                // Update delta_max
                let delta_abs = delta_update.abs();
                let mut current = delta_max.load(Ordering::Relaxed);
                while delta_abs > u64_to_f64(current) {
                    match delta_max.compare_exchange_weak(
                        current,
                        f64_to_u64(delta_abs),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => current = x,
                    }
                }

                // Apply update
                let r = delta_update / mag;
                let r_x = r * dx;

                // Update positions - ODGI pattern: X[i].store(X[i].load() - r_x)
                // Re-load current value right before storing to incorporate concurrent updates
                x[i].store(f64_to_u64(u64_to_f64(x[i].load(Ordering::Relaxed)) - r_x), Ordering::Relaxed);
                x[j].store(f64_to_u64(u64_to_f64(x[j].load(Ordering::Relaxed)) + r_x), Ordering::Relaxed);

                // ODGI batches updates to reduce atomic contention
                term_updates_local += 1;
                if term_updates_local >= 1000 {
                    term_updates.fetch_add(term_updates_local, Ordering::Relaxed);
                    term_updates_local = 0;
                }
            }

            // Flush remaining local updates
            if term_updates_local > 0 {
                term_updates.fetch_add(term_updates_local, Ordering::Relaxed);
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    work_todo.store(false, Ordering::Relaxed);
    checker_handle.join().unwrap();

    // Convert atomic positions to final positions
    let mut positions = HashMap::new();
    for (idx, pos) in x.iter().enumerate() {
        positions.insert(idx, u64_to_f64(pos.load(Ordering::Relaxed)));
    }

    if params.progress {
        eprintln!("[path_sgd] Complete: {} term updates", term_updates.load(Ordering::Relaxed));
    }

    positions
}

/// Learning rate schedule (exact port from ODGI)
fn path_linear_sgd_schedule(
    w_min: f64,
    w_max: f64,
    iter_max: u64,
    iter_with_max_learning_rate: u64,
    eps: f64,
) -> Vec<f64> {
    let mut etas = Vec::new();

    // ODGI formula (from path_sgd.cpp lines 478-493)
    let eta_max = 1.0 / w_min;
    let eta_min = eps / w_max;
    let lambda = (eta_max / eta_min).ln() / (iter_max as f64 - 1.0);

    // Note: ODGI loops from 0 to iter_max (inclusive), so iter_max+1 values
    for t in 0..=iter_max {
        let eta = eta_max * (-lambda * ((t as i64 - iter_with_max_learning_rate as i64).abs() as f64)).exp();
        etas.push(eta);
    }

    etas
}

/// Apply path-guided SGD to graph and return sorted handles
pub fn path_sgd_sort(graph: &BidirectedGraph, params: PathSGDParams) -> Vec<Handle> {
    let graph_arc = Arc::new(graph.clone());
    let positions = path_linear_sgd(graph_arc.clone(), params);

    // Create mapping from index to handle
    // CRITICAL: Must use the SAME node ordering as path_linear_sgd!
    // path_linear_sgd uses node_order (GFA file order) if available,
    // otherwise sorted IDs. We must match this exactly.
    let node_ids: Vec<usize> = if !graph_arc.node_order.is_empty() {
        graph_arc.node_order.clone()
    } else {
        let mut ids: Vec<_> = graph_arc.nodes.iter().enumerate()
            .filter_map(|(id, n)| if n.is_some() { Some(id) } else { None })
            .collect();
        ids.sort();
        ids
    };

    let mut idx_to_handle: HashMap<usize, Handle> = HashMap::new();
    for (idx, node_id) in node_ids.iter().enumerate() {
        idx_to_handle.insert(idx, Handle::forward(*node_id));
    }

    // Sort nodes by position
    let mut node_positions: Vec<(usize, f64)> = positions.into_iter().collect();
    node_positions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Map back to handles (filter out indices without valid handles)
    node_positions.into_iter()
        .filter_map(|(idx, _)| idx_to_handle.get(&idx).copied())
        .collect()
}

/// Parameters for nD layout SGD
#[derive(Debug, Clone)]
pub struct LayoutSGDParams {
    /// Number of dimensions (2 for 2D, 3 for 3D, etc.)
    pub dimensions: usize,
    /// Maximum number of iterations
    pub iter_max: u64,
    /// Iteration at which learning rate is maximum
    pub iter_with_max_learning_rate: u64,
    /// Minimum term updates per iteration
    pub min_term_updates: u64,
    /// Delta threshold for early stopping
    pub delta: f64,
    /// Final learning rate (epsilon)
    pub eps: f64,
    /// Maximum learning rate (eta_max)
    pub eta_max: f64,
    /// Zipfian distribution theta parameter
    pub theta: f64,
    /// Maximum jump space for Zipfian sampling
    pub space: u64,
    /// Maximum space before quantization
    pub space_max: u64,
    /// Quantization step size
    pub space_quantization_step: u64,
    /// When to start cooling phase (0.0-1.0)
    pub cooling_start: f64,
    /// Number of threads
    pub nthreads: usize,
    /// Show progress
    pub progress: bool,
    /// Random seed
    pub seed: u64,
}

impl Default for LayoutSGDParams {
    fn default() -> Self {
        LayoutSGDParams {
            dimensions: 2,
            iter_max: 30,  // ODGI layout default
            iter_with_max_learning_rate: 0,
            min_term_updates: 100,
            delta: 0.0,
            eps: 0.01,
            eta_max: 100.0,
            theta: 0.99,
            space: 100,
            space_max: 1000,
            space_quantization_step: 100,
            cooling_start: 0.5,
            nthreads: 1,
            progress: false,
            seed: 9399220,
        }
    }
}

impl LayoutSGDParams {
    /// Create parameters from graph statistics (matches ODGI's layout defaults)
    pub fn from_graph(graph: &crate::graph_ops::BidirectedGraph, dimensions: usize, nthreads: usize) -> Self {
        let path_index = PathIndex::from_graph(graph);

        let mut sum_path_step_count = 0u64;
        let mut max_path_step_count = 0usize;

        for i in 0..path_index.num_paths() {
            let step_count = path_index.get_path_step_count(i);
            sum_path_step_count += step_count as u64;
            max_path_step_count = max_path_step_count.max(step_count);
        }

        LayoutSGDParams {
            dimensions,
            iter_max: 30,  // ODGI layout default
            iter_with_max_learning_rate: 0,
            min_term_updates: 10 * sum_path_step_count,  // ODGI layout default: 10x path steps
            delta: 0.0,
            eps: 0.01,
            eta_max: (max_path_step_count * max_path_step_count) as f64,
            theta: 0.99,
            space: max_path_step_count as u64,
            space_max: 1000,
            space_quantization_step: 100,
            cooling_start: 0.5,
            nthreads,
            progress: false,
            seed: 9399220,
        }
    }
}

/// N-dimensional path-guided SGD layout (exact port of ODGI's path_linear_sgd_layout)
///
/// This function computes an nD layout for graph visualization using path-guided
/// stochastic gradient descent. Unlike the 1D version, this tracks both ends
/// of each node (+ and - orientation) with separate coordinates.
///
/// The algorithm minimizes the stress between layout distances and path distances
/// using the same sampling and update strategy as ODGI.
pub fn path_linear_sgd_layout(
    graph: Arc<crate::graph_ops::BidirectedGraph>,
    params: LayoutSGDParams,
) -> crate::layout::Layout {
    use crate::layout::Layout;

    let num_nodes = graph.node_count();
    if num_nodes == 0 {
        return Layout::new(params.dimensions, 0);
    }

    let path_index = PathIndex::from_graph(&graph);

    // Check if we have any paths with more than one step
    let mut has_valid_paths = false;
    for i in 0..path_index.num_paths() {
        if path_index.get_path_step_count(i) > 1 {
            has_valid_paths = true;
            break;
        }
    }

    if !has_valid_paths {
        eprintln!("[path_sgd_layout] No paths with multiple steps found");
        return Layout::new(params.dimensions, num_nodes);
    }

    // Build node ID to index mapping (same as 1D)
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

    // Initialize positions
    // ODGI uses 2*num_nodes entries per dimension (+ and - end for each node)
    // We store: coords[dim][node * 2 + end]
    let dims = params.dimensions;
    let entries_per_dim = num_nodes * 2;

    // Create atomic coordinate arrays for each dimension
    let coords: Vec<Vec<AtomicU64>> = (0..dims)
        .map(|_| (0..entries_per_dim).map(|_| AtomicU64::new(0)).collect())
        .collect();

    // Initialize with ODGI's default: node rank in X, gaussian noise in Y (for 2D)
    // For nD: first dimension gets node rank, others get gaussian noise
    let mut rng = Xoshiro256Plus::seed_from_u64(params.seed);
    let gaussian = rand_distr::StandardNormal;

    let mut len = 0u64;
    for (idx, node_id) in node_ids.iter().enumerate() {
        if let Some(Some(node)) = graph.nodes.get(*node_id) {
            let node_len = node.sequence.len() as u64;
            let sqrt_n = (num_nodes as f64 * 2.0).sqrt();

            // + end (offset 0)
            coords[0][idx * 2].store(f64_to_u64(len as f64), Ordering::Relaxed);
            for d in 1..dims {
                let noise: f64 = rand_distr::Distribution::sample(&gaussian, &mut rng);
                coords[d][idx * 2].store(f64_to_u64(noise * sqrt_n), Ordering::Relaxed);
            }

            // - end (offset 1) = + end position + node length
            coords[0][idx * 2 + 1].store(f64_to_u64((len + node_len) as f64), Ordering::Relaxed);
            for d in 1..dims {
                let noise: f64 = rand_distr::Distribution::sample(&gaussian, &mut rng);
                coords[d][idx * 2 + 1].store(f64_to_u64(noise * sqrt_n), Ordering::Relaxed);
            }

            len += node_len;
        }
    }

    // Calculate first cooling iteration
    let first_cooling_iteration = (params.cooling_start * params.iter_max as f64).floor() as u64;

    // Calculate learning rate schedule
    let w_min = 1.0 / params.eta_max;
    let w_max = 1.0;
    let etas = path_linear_sgd_schedule(
        w_min,
        w_max,
        params.iter_max,
        params.iter_with_max_learning_rate,
        params.eps,
    );

    // Pre-calculate zetas for Zipfian distribution (same as 1D)
    let zeta_size = if params.space <= params.space_max {
        params.space as usize
    } else {
        params.space_max as usize + (params.space - params.space_max) as usize / params.space_quantization_step as usize + 1
    } + 1;

    let mut zetas = vec![0.0; zeta_size];
    let mut zeta_tmp = 0.0;
    for i in 1..=params.space {
        zeta_tmp += fast_precise_pow(1.0 / i as f64, params.theta);
        if i <= params.space_max {
            zetas[i as usize] = zeta_tmp;
        }
        if i >= params.space_max && (i - params.space_max) % params.space_quantization_step == 0 {
            let idx = params.space_max as usize + 1 + ((i - params.space_max) / params.space_quantization_step) as usize;
            if idx < zetas.len() {
                zetas[idx] = zeta_tmp;
            }
        }
    }

    // Shared state
    let coords = Arc::new(coords);
    let path_index = Arc::new(path_index);
    let handle_to_idx = Arc::new(handle_to_idx);
    let zetas = Arc::new(zetas);
    let etas = Arc::new(etas);

    let term_updates = Arc::new(AtomicU64::new(0));
    let iteration = Arc::new(AtomicU64::new(0));
    let eta = Arc::new(AtomicU64::new(f64_to_u64(etas[0])));
    let adj_theta = Arc::new(AtomicU64::new(f64_to_u64(params.theta)));
    let cooling = Arc::new(AtomicBool::new(false));
    let work_todo = Arc::new(AtomicBool::new(true));
    let delta_max = Arc::new(AtomicU64::new(0));

    if params.progress {
        eprintln!("[path_sgd_layout] Starting {}D layout with {} iterations, {} term updates per iteration",
                 dims, params.iter_max, params.min_term_updates);
    }

    // Checker thread
    let checker_handle = {
        let term_updates = Arc::clone(&term_updates);
        let iteration = Arc::clone(&iteration);
        let work_todo = Arc::clone(&work_todo);
        let eta = Arc::clone(&eta);
        let adj_theta = Arc::clone(&adj_theta);
        let cooling = Arc::clone(&cooling);
        let etas = Arc::clone(&etas);
        let min_term_updates = params.min_term_updates;
        let iter_max = params.iter_max;
        let progress = params.progress;

        thread::spawn(move || {
            while work_todo.load(Ordering::Relaxed) {
                let curr_updates = term_updates.load(Ordering::Relaxed);

                if curr_updates >= min_term_updates {
                    iteration.fetch_add(1, Ordering::Relaxed);
                    let new_iter = iteration.load(Ordering::Relaxed);

                    if progress && new_iter % 5 == 0 {
                        eprintln!("[path_sgd_layout] iteration {}/{}", new_iter, iter_max);
                    }

                    if new_iter > iter_max {
                        work_todo.store(false, Ordering::Relaxed);
                    } else {
                        if (new_iter as usize) < etas.len() {
                            eta.store(f64_to_u64(etas[new_iter as usize]), Ordering::Relaxed);
                        }

                        if new_iter > first_cooling_iteration {
                            adj_theta.store(f64_to_u64(0.001), Ordering::Relaxed);
                            cooling.store(true, Ordering::Relaxed);
                        }
                    }

                    term_updates.store(0, Ordering::Relaxed);
                }

                thread::sleep(Duration::from_millis(1));
            }
        })
    };

    // Worker threads
    let mut handles = vec![];

    for tid in 0..params.nthreads {
        let coords = Arc::clone(&coords);
        let path_index = Arc::clone(&path_index);
        let handle_to_idx = Arc::clone(&handle_to_idx);
        let zetas = Arc::clone(&zetas);
        let term_updates = Arc::clone(&term_updates);
        let work_todo = Arc::clone(&work_todo);
        let eta = Arc::clone(&eta);
        let adj_theta = Arc::clone(&adj_theta);
        let cooling = Arc::clone(&cooling);
        let delta_max = Arc::clone(&delta_max);
        let space = params.space;
        let space_max = params.space_max;
        let space_quantization_step = params.space_quantization_step;
        let base_seed = params.seed;
        let graph = Arc::clone(&graph);

        let handle = thread::spawn(move || {
            let seed = base_seed + tid as u64;
            let mut rng = Xoshiro256Plus::seed_from_u64(seed);

            let total_steps = path_index.get_total_steps();
            let step_dist = Uniform::new(0, total_steps).unwrap();
            let flip_dist = Uniform::new(0, 2).unwrap();

            let mut term_updates_local = 0u64;

            while work_todo.load(Ordering::Relaxed) {
                // Sample a random step
                let step_idx = step_dist.sample(&mut rng);
                let path_idx = path_index.get_path_of_step(step_idx);
                let path_step_count = path_index.get_path_step_count(path_idx);

                if path_step_count == 1 {
                    continue;
                }

                let rank_a = path_index.get_rank_of_step(step_idx);
                let mut rank_b = rank_a;

                // Sample second step (same logic as 1D)
                if cooling.load(Ordering::Relaxed) || flip_dist.sample(&mut rng) == 1 {
                    let current_theta = u64_to_f64(adj_theta.load(Ordering::Relaxed));

                    if rank_a > 0 && (flip_dist.sample(&mut rng) == 1 || rank_a == path_step_count - 1) {
                        let jump_space = space.min(rank_a as u64);
                        let space_idx = if jump_space > space_max {
                            space_max as usize + ((jump_space - space_max) / space_quantization_step) as usize + 1
                        } else {
                            jump_space as usize
                        };
                        let space_idx = space_idx.min(zetas.len() - 1);
                        let zeta2theta = 1.0 + fast_precise_pow(0.5, current_theta);
                        let zipf = DirtyZipfian::new(1, jump_space, current_theta, zetas[space_idx], zeta2theta);
                        let z_i = zipf.sample(&mut rng);
                        rank_b = rank_a.saturating_sub(z_i as usize);
                    } else if rank_a < path_step_count - 1 {
                        let jump_space = space.min((path_step_count - rank_a - 1) as u64);
                        let space_idx = if jump_space > space_max {
                            space_max as usize + ((jump_space - space_max) / space_quantization_step) as usize + 1
                        } else {
                            jump_space as usize
                        };
                        let space_idx = space_idx.min(zetas.len() - 1);
                        let zeta2theta = 1.0 + fast_precise_pow(0.5, current_theta);
                        let zipf = DirtyZipfian::new(1, jump_space, current_theta, zetas[space_idx], zeta2theta);
                        let z_i = zipf.sample(&mut rng);
                        rank_b = (rank_a + z_i as usize).min(path_step_count - 1);
                    }
                } else {
                    let rank_dist = Uniform::new(0, path_step_count).unwrap();
                    rank_b = rank_dist.sample(&mut rng);
                }

                if rank_a == rank_b {
                    continue;
                }

                // Get handles for the terms
                let step_a_idx = path_index.get_step_at_path_position(path_idx, rank_a);
                let step_b_idx = path_index.get_step_at_path_position(path_idx, rank_b);

                let term_i = path_index.get_handle_of_step(step_a_idx);
                let term_j = path_index.get_handle_of_step(step_b_idx);

                // Get positions in path
                let mut pos_a = path_index.get_position_of_step(step_a_idx) as f64;
                let mut pos_b = path_index.get_position_of_step(step_b_idx) as f64;

                // Get node lengths for position adjustment
                let term_i_length = graph.nodes.get(term_i.node_id())
                    .and_then(|n| n.as_ref())
                    .map(|n| n.sequence.len())
                    .unwrap_or(0) as f64;
                let term_j_length = graph.nodes.get(term_j.node_id())
                    .and_then(|n| n.as_ref())
                    .map(|n| n.sequence.len())
                    .unwrap_or(0) as f64;

                // ODGI: randomly choose which end of each node to use
                let term_i_is_rev = term_i.is_reverse();
                let mut use_other_end_a = flip_dist.sample(&mut rng) == 1;
                if use_other_end_a {
                    pos_a += term_i_length;
                    use_other_end_a = !term_i_is_rev;
                } else {
                    use_other_end_a = term_i_is_rev;
                }

                let term_j_is_rev = term_j.is_reverse();
                let mut use_other_end_b = flip_dist.sample(&mut rng) == 1;
                if use_other_end_b {
                    pos_b += term_j_length;
                    use_other_end_b = !term_j_is_rev;
                } else {
                    use_other_end_b = term_j_is_rev;
                }

                // Calculate term distance
                let term_dist = (pos_a - pos_b).abs();
                if term_dist == 0.0 {
                    continue;
                }

                let term_weight = 1.0 / term_dist;
                let mu = (u64_to_f64(eta.load(Ordering::Relaxed)) * term_weight).min(1.0);

                // Get node indices
                let i = match handle_to_idx.get(&Handle::forward(term_i.node_id())).copied() {
                    Some(idx) => idx,
                    None => continue,
                };
                let j = match handle_to_idx.get(&Handle::forward(term_j.node_id())).copied() {
                    Some(idx) => idx,
                    None => continue,
                };

                // Calculate offsets for + or - end
                let offset_i = if use_other_end_a { 1 } else { 0 };
                let offset_j = if use_other_end_b { 1 } else { 0 };

                let idx_i = i * 2 + offset_i;
                let idx_j = j * 2 + offset_j;

                // Calculate deltas and magnitude (Euclidean distance)
                let mut deltas = vec![0.0; dims];
                let mut mag_sq = 0.0;
                for d in 0..dims {
                    let c_i = u64_to_f64(coords[d][idx_i].load(Ordering::Relaxed));
                    let c_j = u64_to_f64(coords[d][idx_j].load(Ordering::Relaxed));
                    deltas[d] = c_i - c_j;
                    mag_sq += deltas[d] * deltas[d];
                }

                // Avoid NaN
                if mag_sq == 0.0 {
                    deltas[0] = 1e-9;
                    mag_sq = 1e-18;
                }

                let mag = mag_sq.sqrt();
                let d_ij = term_dist;

                // Check for early stopping
                let delta_update = mu * (mag - d_ij) / 2.0;
                let delta_abs = delta_update.abs();

                let mut current = delta_max.load(Ordering::Relaxed);
                while delta_abs > u64_to_f64(current) {
                    match delta_max.compare_exchange_weak(
                        current,
                        f64_to_u64(delta_abs),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => current = x,
                    }
                }

                // Apply update to all dimensions
                let r = delta_update / mag;
                for d in 0..dims {
                    let r_d = r * deltas[d];
                    let c_i = u64_to_f64(coords[d][idx_i].load(Ordering::Relaxed));
                    let c_j = u64_to_f64(coords[d][idx_j].load(Ordering::Relaxed));
                    coords[d][idx_i].store(f64_to_u64(c_i - r_d), Ordering::Relaxed);
                    coords[d][idx_j].store(f64_to_u64(c_j + r_d), Ordering::Relaxed);
                }

                term_updates_local += 1;
                if term_updates_local >= 1000 {
                    term_updates.fetch_add(term_updates_local, Ordering::Relaxed);
                    term_updates_local = 0;
                }
            }

            if term_updates_local > 0 {
                term_updates.fetch_add(term_updates_local, Ordering::Relaxed);
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    work_todo.store(false, Ordering::Relaxed);
    checker_handle.join().unwrap();

    if params.progress {
        eprintln!("[path_sgd_layout] Complete");
    }

    // Convert atomic coordinates to Layout
    let coord_vecs: Vec<Vec<f64>> = coords.iter()
        .map(|dim_coords| {
            dim_coords.iter()
                .map(|c| u64_to_f64(c.load(Ordering::Relaxed)))
                .collect()
        })
        .collect();

    Layout::from_vectors(coord_vecs)
}

/// Calculate layout stress/distortion using path distances as targets
///
/// This computes the normalized stress metric:
/// stress = sqrt(sum((d_layout - d_path)^2 / d_path^2) / num_pairs)
///
/// Lower values indicate better layout quality (closer to ideal distances).
pub fn calculate_layout_stress(
    graph: &crate::graph_ops::BidirectedGraph,
    layout: &crate::layout::Layout,
    sample_count: usize,
) -> f64 {
    let path_index = PathIndex::from_graph(graph);

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

        // Use + end (end=0) for stress calculation
        let layout_dist = layout.distance(idx_a, 0, idx_b, 0);

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