//! Core Mamba (Selective State Space) model implementation.
//!
//! This module implements the Mamba architecture for sequence modeling,
//! optimized for financial time series prediction.

use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

/// Mamba block implementing the selective state space mechanism.
///
/// Architecture:
/// 1. Input projection with gating
/// 2. Causal 1D convolution
/// 3. Selective SSM
/// 4. Output projection
#[derive(Clone, Serialize, Deserialize)]
pub struct MambaBlock {
    /// Model dimension
    pub d_model: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Convolution kernel size
    pub d_conv: usize,
    /// Inner dimension (d_model * expand)
    pub d_inner: usize,
    /// Rank for dt projection
    pub dt_rank: usize,

    // Weights
    /// Input projection weights (d_model -> 2*d_inner)
    pub in_proj_weight: Array2<f64>,
    /// Convolution weights (d_inner, d_conv)
    pub conv_weight: Array2<f64>,
    /// Convolution bias
    pub conv_bias: Array1<f64>,
    /// X projection weights (d_inner -> dt_rank + 2*d_state)
    pub x_proj_weight: Array2<f64>,
    /// Delta projection weights (dt_rank -> d_inner)
    pub dt_proj_weight: Array2<f64>,
    /// Delta projection bias
    pub dt_proj_bias: Array1<f64>,
    /// A parameter (log space)
    pub a_log: Array2<f64>,
    /// D parameter (skip connection)
    pub d_param: Array1<f64>,
    /// Output projection weights (d_inner -> d_model)
    pub out_proj_weight: Array2<f64>,
}

impl MambaBlock {
    /// Create a new Mamba block with random initialization.
    pub fn new(d_model: usize, d_state: usize, d_conv: usize, expand: usize) -> Self {
        let d_inner = d_model * expand;
        let dt_rank = (d_model as f64 / 16.0).ceil() as usize;

        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let in_scale = (2.0 / (d_model + d_inner * 2) as f64).sqrt();
        let out_scale = (2.0 / (d_inner + d_model) as f64).sqrt();
        let x_proj_scale = (2.0 / (d_inner + dt_rank + d_state * 2) as f64).sqrt();
        let dt_scale = (2.0 / (dt_rank + d_inner) as f64).sqrt();

        // Initialize weights
        let in_proj_weight = random_array2(&mut rng, d_model, d_inner * 2, in_scale);
        let conv_weight = random_array2(&mut rng, d_inner, d_conv, 0.1);
        let conv_bias = Array1::zeros(d_inner);
        let x_proj_weight = random_array2(&mut rng, d_inner, dt_rank + d_state * 2, x_proj_scale);
        let dt_proj_weight = random_array2(&mut rng, dt_rank, d_inner, dt_scale);

        // Initialize dt bias for proper initialization
        let dt_proj_bias = Array1::from_iter((0..d_inner).map(|_| {
            let dt_init = rng.gen_range(0.001..0.1_f64);
            dt_init + (-(-dt_init).exp_m1()).ln()
        }));

        // Initialize A in log space (HiPPO-inspired)
        let a_log = Array2::from_shape_fn((d_inner, d_state), |(_, j)| ((j + 1) as f64).ln());

        // D parameter (skip connection)
        let d_param = Array1::ones(d_inner);

        let out_proj_weight = random_array2(&mut rng, d_inner, d_model, out_scale);

        Self {
            d_model,
            d_state,
            d_conv,
            d_inner,
            dt_rank,
            in_proj_weight,
            conv_weight,
            conv_bias,
            x_proj_weight,
            dt_proj_weight,
            dt_proj_bias,
            a_log,
            d_param,
            out_proj_weight,
        }
    }

    /// Forward pass through the Mamba block.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (batch, seq_len, d_model)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (batch, seq_len, d_model)
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();

        // Input projection: (batch, seq_len, d_model) -> (batch, seq_len, 2*d_inner)
        let xz = self.linear_3d(x, &self.in_proj_weight);

        // Split into x and z (gating)
        let (x_part, z) = self.split_last_axis(&xz, self.d_inner);

        // Causal convolution
        let x_conv = self.causal_conv1d(&x_part);

        // SiLU activation
        let x_act = self.silu_3d(&x_conv);

        // Selective SSM
        let y = self.ssm(&x_act);

        // Gating
        let z_act = self.silu_3d(&z);
        let y_gated = &y * &z_act;

        // Output projection
        self.linear_3d(&y_gated, &self.out_proj_weight)
    }

    /// Selective State Space Model computation.
    fn ssm(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();

        // Project x to get dt, B, C
        let x_dbl = self.linear_3d(x, &self.x_proj_weight);

        // Get A from log space (negative for stability)
        let a: Array2<f64> = self.a_log.mapv(|v| -v.exp());

        let mut output = Array3::<f64>::zeros((batch, seq_len, self.d_inner));

        for b in 0..batch {
            let mut h = Array2::<f64>::zeros((self.d_inner, self.d_state));

            for t in 0..seq_len {
                // Extract dt, B, C from projection
                let x_dbl_t = x_dbl.slice(ndarray::s![b, t, ..]);

                let dt_input: Array1<f64> = x_dbl_t.slice(ndarray::s![..self.dt_rank]).to_owned();
                let b_t: Array1<f64> = x_dbl_t
                    .slice(ndarray::s![self.dt_rank..self.dt_rank + self.d_state])
                    .to_owned();
                let c_t: Array1<f64> = x_dbl_t
                    .slice(ndarray::s![self.dt_rank + self.d_state..])
                    .to_owned();

                // Compute delta with projection and softplus
                let dt = self.softplus(&self.linear_1d(&dt_input, &self.dt_proj_weight, Some(&self.dt_proj_bias)));

                // Get input for this timestep
                let x_t = x.slice(ndarray::s![b, t, ..]).to_owned();

                // Discretize: dA = exp(dt * A)
                let dt_expanded = dt.clone().insert_axis(Axis(1));
                let da = (&dt_expanded * &a).mapv(f64::exp);

                // dB = dt * B
                let db_outer = outer_product(&dt, &b_t);

                // State update: h = dA * h + dB * x
                h = &da * &h;
                for i in 0..self.d_inner {
                    for j in 0..self.d_state {
                        h[[i, j]] += db_outer[[i, j]] * x_t[i];
                    }
                }

                // Output: y = C * h
                for i in 0..self.d_inner {
                    let mut y_i = 0.0;
                    for j in 0..self.d_state {
                        y_i += h[[i, j]] * c_t[j];
                    }
                    // Add skip connection: y = y + D * x
                    output[[b, t, i]] = y_i + self.d_param[i] * x_t[i];
                }
            }
        }

        output
    }

    /// Apply 1D causal convolution.
    fn causal_conv1d(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, d_inner) = x.dim();
        let mut output = Array3::<f64>::zeros((batch, seq_len, d_inner));

        for b in 0..batch {
            for t in 0..seq_len {
                for i in 0..d_inner {
                    let mut sum = self.conv_bias[i];
                    for k in 0..self.d_conv {
                        let src_t = t as i32 - k as i32;
                        if src_t >= 0 {
                            sum += x[[b, src_t as usize, i]] * self.conv_weight[[i, k]];
                        }
                    }
                    output[[b, t, i]] = sum;
                }
            }
        }

        output
    }

    /// Linear transformation for 3D tensors.
    fn linear_3d(&self, x: &Array3<f64>, weight: &Array2<f64>) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();
        let out_dim = weight.dim().1;
        let mut output = Array3::<f64>::zeros((batch, seq_len, out_dim));

        for b in 0..batch {
            for t in 0..seq_len {
                let x_t = x.slice(ndarray::s![b, t, ..]);
                let out_t = x_t.dot(weight);
                output.slice_mut(ndarray::s![b, t, ..]).assign(&out_t);
            }
        }

        output
    }

    /// Linear transformation for 1D tensors.
    fn linear_1d(
        &self,
        x: &Array1<f64>,
        weight: &Array2<f64>,
        bias: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let mut out = x.dot(weight);
        if let Some(b) = bias {
            out = &out + b;
        }
        out
    }

    /// Split tensor along last axis.
    fn split_last_axis(&self, x: &Array3<f64>, split_size: usize) -> (Array3<f64>, Array3<f64>) {
        let (batch, seq_len, _) = x.dim();

        let first = x.slice(ndarray::s![.., .., ..split_size]).to_owned();
        let second = x.slice(ndarray::s![.., .., split_size..]).to_owned();

        (first, second)
    }

    /// SiLU (Swish) activation for 3D tensors.
    fn silu_3d(&self, x: &Array3<f64>) -> Array3<f64> {
        x.mapv(|v| v * sigmoid(v))
    }

    /// Softplus activation.
    fn softplus(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| (1.0 + v.exp()).ln())
    }
}

/// Mamba layer with residual connection and normalization.
#[derive(Clone, Serialize, Deserialize)]
pub struct MambaLayer {
    /// Layer normalization parameters
    pub norm_weight: Array1<f64>,
    pub norm_bias: Array1<f64>,
    /// Mamba block
    pub mamba: MambaBlock,
    /// Epsilon for numerical stability
    eps: f64,
}

impl MambaLayer {
    /// Create a new Mamba layer.
    pub fn new(d_model: usize, d_state: usize, d_conv: usize, expand: usize) -> Self {
        Self {
            norm_weight: Array1::ones(d_model),
            norm_bias: Array1::zeros(d_model),
            mamba: MambaBlock::new(d_model, d_state, d_conv, expand),
            eps: 1e-5,
        }
    }

    /// Forward pass with pre-norm and residual.
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        // Pre-norm
        let normed = self.layer_norm(x);

        // Mamba block
        let mamba_out = self.mamba.forward(&normed);

        // Residual connection
        x + &mamba_out
    }

    /// Layer normalization.
    fn layer_norm(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, d_model) = x.dim();
        let mut output = Array3::<f64>::zeros((batch, seq_len, d_model));

        for b in 0..batch {
            for t in 0..seq_len {
                let x_t = x.slice(ndarray::s![b, t, ..]);
                let mean = x_t.mean().unwrap_or(0.0);
                let var = x_t.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
                let std = (var + self.eps).sqrt();

                for i in 0..d_model {
                    output[[b, t, i]] =
                        (x[[b, t, i]] - mean) / std * self.norm_weight[i] + self.norm_bias[i];
                }
            }
        }

        output
    }
}

// Helper functions

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn random_array2<R: Rng>(rng: &mut R, rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    let dist = Normal::new(0.0, scale).unwrap();
    Array2::from_shape_fn((rows, cols), |_| dist.sample(rng))
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_block_creation() {
        let block = MambaBlock::new(64, 16, 4, 2);
        assert_eq!(block.d_model, 64);
        assert_eq!(block.d_state, 16);
        assert_eq!(block.d_inner, 128);
    }

    #[test]
    fn test_mamba_block_forward() {
        let block = MambaBlock::new(32, 8, 4, 2);
        let x = Array3::<f64>::zeros((2, 10, 32)); // batch=2, seq_len=10, d_model=32
        let output = block.forward(&x);
        assert_eq!(output.dim(), (2, 10, 32));
    }

    #[test]
    fn test_mamba_layer_forward() {
        let layer = MambaLayer::new(32, 8, 4, 2);
        let x = Array3::<f64>::ones((1, 5, 32));
        let output = layer.forward(&x);
        assert_eq!(output.dim(), (1, 5, 32));
    }
}
