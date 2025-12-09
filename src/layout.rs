/// N-dimensional layout for graph visualization
///
/// This module provides coordinate storage and I/O for 2D and nD graph layouts.
/// It follows ODGI's approach where each node has 2 positions (+ and - orientation ends).

use std::io::{self, Write, BufRead};

/// N-dimensional layout storing coordinates for each node end
///
/// For a graph with N nodes and D dimensions, stores 2*N*D coordinates:
/// - Each node has 2 ends (+ and - orientation)
/// - Each end has D coordinate values
///
/// Storage layout: coords[node_idx * 2 * dims + end * dims + dim]
/// where end=0 for +, end=1 for -
#[derive(Clone, Debug)]
pub struct Layout {
    /// Number of dimensions (2 for 2D, 3 for 3D, etc.)
    pub dimensions: usize,
    /// Number of nodes
    pub num_nodes: usize,
    /// Flattened coordinate storage: [node0_plus_x, node0_plus_y, ..., node0_minus_x, node0_minus_y, ...]
    pub coords: Vec<f64>,
}

impl Layout {
    /// Create a new layout with the given dimensions and node count
    pub fn new(dimensions: usize, num_nodes: usize) -> Self {
        let size = num_nodes * 2 * dimensions;
        Layout {
            dimensions,
            num_nodes,
            coords: vec![0.0; size],
        }
    }

    /// Create a layout from flat coordinate vectors (one per dimension)
    /// Each vector should have 2*num_nodes entries (+ and - end for each node)
    pub fn from_vectors(coord_vecs: Vec<Vec<f64>>) -> Self {
        let dimensions = coord_vecs.len();
        assert!(dimensions > 0, "Must have at least 1 dimension");

        let entries_per_dim = coord_vecs[0].len();
        assert!(entries_per_dim % 2 == 0, "Must have even number of entries (2 per node)");
        let num_nodes = entries_per_dim / 2;

        // Verify all dimensions have same size
        for v in &coord_vecs {
            assert_eq!(v.len(), entries_per_dim, "All dimension vectors must have same length");
        }

        // Interleave coordinates: for each node end, store all dimensions together
        let mut coords = vec![0.0; num_nodes * 2 * dimensions];
        for node in 0..num_nodes {
            for end in 0..2 {
                let src_idx = node * 2 + end;
                for dim in 0..dimensions {
                    let dst_idx = node * 2 * dimensions + end * dimensions + dim;
                    coords[dst_idx] = coord_vecs[dim][src_idx];
                }
            }
        }

        Layout {
            dimensions,
            num_nodes,
            coords,
        }
    }

    /// Get the index into coords for a specific node/end/dimension
    #[inline]
    pub fn index(&self, node: usize, end: usize, dim: usize) -> usize {
        debug_assert!(node < self.num_nodes);
        debug_assert!(end < 2);
        debug_assert!(dim < self.dimensions);
        node * 2 * self.dimensions + end * self.dimensions + dim
    }

    /// Get coordinate value for a node end in a specific dimension
    #[inline]
    pub fn get(&self, node: usize, end: usize, dim: usize) -> f64 {
        self.coords[self.index(node, end, dim)]
    }

    /// Set coordinate value for a node end in a specific dimension
    #[inline]
    pub fn set(&mut self, node: usize, end: usize, dim: usize, value: f64) {
        let idx = self.index(node, end, dim);
        self.coords[idx] = value;
    }

    /// Get all coordinates for a node end as a slice
    pub fn get_coords(&self, node: usize, end: usize) -> &[f64] {
        let start = self.index(node, end, 0);
        &self.coords[start..start + self.dimensions]
    }

    /// Get X coordinate (dimension 0) for node's + end
    #[inline]
    pub fn x_plus(&self, node: usize) -> f64 {
        self.get(node, 0, 0)
    }

    /// Get Y coordinate (dimension 1) for node's + end
    #[inline]
    pub fn y_plus(&self, node: usize) -> f64 {
        debug_assert!(self.dimensions >= 2);
        self.get(node, 0, 1)
    }

    /// Get X coordinate (dimension 0) for node's - end
    #[inline]
    pub fn x_minus(&self, node: usize) -> f64 {
        self.get(node, 1, 0)
    }

    /// Get Y coordinate (dimension 1) for node's - end
    #[inline]
    pub fn y_minus(&self, node: usize) -> f64 {
        debug_assert!(self.dimensions >= 2);
        self.get(node, 1, 1)
    }

    /// Calculate Euclidean distance between two node ends
    pub fn distance(&self, node_a: usize, end_a: usize, node_b: usize, end_b: usize) -> f64 {
        let mut sum_sq = 0.0;
        for dim in 0..self.dimensions {
            let delta = self.get(node_a, end_a, dim) - self.get(node_b, end_b, dim);
            sum_sq += delta * delta;
        }
        sum_sq.sqrt()
    }

    /// Write layout to TSV format (ODGI-compatible for 2D)
    /// Format: idx  x+  y+  x-  y-  (for 2D)
    /// For nD: idx  d0+  d1+  ...  dN+  d0-  d1-  ...  dN-
    pub fn write_tsv<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write header
        write!(writer, "idx")?;
        for dim in 0..self.dimensions {
            write!(writer, "\t{}+", dim_name(dim))?;
        }
        for dim in 0..self.dimensions {
            write!(writer, "\t{}-", dim_name(dim))?;
        }
        writeln!(writer)?;

        // Write data
        for node in 0..self.num_nodes {
            write!(writer, "{}", node)?;
            // + end coordinates
            for dim in 0..self.dimensions {
                write!(writer, "\t{}", self.get(node, 0, dim))?;
            }
            // - end coordinates
            for dim in 0..self.dimensions {
                write!(writer, "\t{}", self.get(node, 1, dim))?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }

    /// Read layout from TSV format
    pub fn read_tsv<R: BufRead>(reader: &mut R) -> io::Result<Self> {
        let mut lines = reader.lines();

        // Parse header to determine dimensions
        let header = lines.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty file"))??;
        let cols: Vec<&str> = header.split('\t').collect();

        // Header is: idx, d0+, d1+, ..., dN+, d0-, d1-, ..., dN-
        // So (cols.len() - 1) / 2 = dimensions
        if cols.len() < 3 || (cols.len() - 1) % 2 != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                "Invalid header format"));
        }
        let dimensions = (cols.len() - 1) / 2;

        // Read data
        let mut coords_data: Vec<Vec<f64>> = Vec::new();
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != cols.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("Row has {} columns, expected {}", parts.len(), cols.len())));
            }

            let mut node_coords = Vec::with_capacity(dimensions * 2);
            for i in 1..parts.len() {
                let val: f64 = parts[i].parse()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                        format!("Failed to parse coordinate: {}", e)))?;
                node_coords.push(val);
            }
            coords_data.push(node_coords);
        }

        let num_nodes = coords_data.len();
        let mut layout = Layout::new(dimensions, num_nodes);

        for (node, node_coords) in coords_data.into_iter().enumerate() {
            // First half is + end, second half is - end
            for dim in 0..dimensions {
                layout.set(node, 0, dim, node_coords[dim]);
                layout.set(node, 1, dim, node_coords[dimensions + dim]);
            }
        }

        Ok(layout)
    }

    /// Calculate layout stress/distortion
    /// This measures how well the layout distances match the target distances
    ///
    /// stress = sum((layout_dist - target_dist)^2 * weight) / sum(weight)
    /// where weight = 1/target_dist^2 (standard MDS weighting)
    pub fn calculate_stress(&self, target_distances: &[(usize, usize, usize, usize, f64)]) -> f64 {
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for &(node_a, end_a, node_b, end_b, target_dist) in target_distances {
            if target_dist == 0.0 {
                continue;
            }
            let layout_dist = self.distance(node_a, end_a, node_b, end_b);
            let weight = 1.0 / (target_dist * target_dist);
            let error = layout_dist - target_dist;
            weighted_sum += error * error * weight;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            (weighted_sum / weight_total).sqrt()
        } else {
            0.0
        }
    }
}

/// Get dimension name (x, y, z, w, d4, d5, ...)
fn dim_name(dim: usize) -> &'static str {
    match dim {
        0 => "x",
        1 => "y",
        2 => "z",
        3 => "w",
        _ => "d",  // Will need dynamic formatting for dim >= 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_new() {
        let layout = Layout::new(2, 10);
        assert_eq!(layout.dimensions, 2);
        assert_eq!(layout.num_nodes, 10);
        assert_eq!(layout.coords.len(), 10 * 2 * 2);
    }

    #[test]
    fn test_layout_get_set() {
        let mut layout = Layout::new(2, 5);
        layout.set(2, 0, 0, 100.0);  // node 2, + end, x
        layout.set(2, 0, 1, 200.0);  // node 2, + end, y
        layout.set(2, 1, 0, 150.0);  // node 2, - end, x
        layout.set(2, 1, 1, 250.0);  // node 2, - end, y

        assert_eq!(layout.x_plus(2), 100.0);
        assert_eq!(layout.y_plus(2), 200.0);
        assert_eq!(layout.x_minus(2), 150.0);
        assert_eq!(layout.y_minus(2), 250.0);
    }

    #[test]
    fn test_layout_distance() {
        let mut layout = Layout::new(2, 2);
        layout.set(0, 0, 0, 0.0);
        layout.set(0, 0, 1, 0.0);
        layout.set(1, 0, 0, 3.0);
        layout.set(1, 0, 1, 4.0);

        let dist = layout.distance(0, 0, 1, 0);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_vectors() {
        let x = vec![1.0, 2.0, 3.0, 4.0];  // 2 nodes, + and - for each
        let y = vec![10.0, 20.0, 30.0, 40.0];

        let layout = Layout::from_vectors(vec![x, y]);
        assert_eq!(layout.num_nodes, 2);
        assert_eq!(layout.dimensions, 2);

        assert_eq!(layout.x_plus(0), 1.0);
        assert_eq!(layout.y_plus(0), 10.0);
        assert_eq!(layout.x_minus(0), 2.0);
        assert_eq!(layout.y_minus(0), 20.0);
        assert_eq!(layout.x_plus(1), 3.0);
        assert_eq!(layout.y_plus(1), 30.0);
    }

    #[test]
    fn test_tsv_roundtrip() {
        let mut layout = Layout::new(2, 3);
        layout.set(0, 0, 0, 1.5);
        layout.set(0, 0, 1, 2.5);
        layout.set(0, 1, 0, 3.5);
        layout.set(0, 1, 1, 4.5);
        layout.set(1, 0, 0, 10.0);
        layout.set(1, 0, 1, 20.0);
        layout.set(1, 1, 0, 30.0);
        layout.set(1, 1, 1, 40.0);
        layout.set(2, 0, 0, 100.0);
        layout.set(2, 0, 1, 200.0);
        layout.set(2, 1, 0, 300.0);
        layout.set(2, 1, 1, 400.0);

        let mut buf = Vec::new();
        layout.write_tsv(&mut buf).unwrap();

        let mut reader = std::io::BufReader::new(&buf[..]);
        let loaded = Layout::read_tsv(&mut reader).unwrap();

        assert_eq!(loaded.dimensions, layout.dimensions);
        assert_eq!(loaded.num_nodes, layout.num_nodes);
        for i in 0..layout.coords.len() {
            assert!((loaded.coords[i] - layout.coords[i]).abs() < 1e-10);
        }
    }
}
