//! ONNX-runtime backend for the [`crate::Embedder`] trait.
//!
//! Compiled only when the `onnx` feature is enabled (see `Cargo.toml`).
//! Requires:
//!
//! * The ONNX Runtime shared library on the loader path. On macOS Homebrew:
//!   `brew install onnxruntime` then set `ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib`.
//! * A pre-exported MiniLM ONNX model and matching tokenizer JSON. Fetch via
//!   `scripts/fetch-model.sh` which pulls `Xenova/all-MiniLM-L6-v2`.
//!
//! The PRD (§6.2) specifies an 80M-param ONNX transformer. This module works
//! with either the 22M L6 variant (ships by default for speed) or the 80M
//! L12 variant by swapping the model file — the tokenizer and pooling are
//! identical.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::{l2_normalize_in_place, Embedder};

pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
    max_len: usize,
}

impl OnnxEmbedder {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)
            .with_context(|| format!("loading ONNX model from {}", model_path.display()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        // MiniLM-L6 = 384, MiniLM-L12 = 384, mpnet = 768. Default to 384.
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: 384,
            max_len: 256,
        })
    }

    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

impl Embedder for OnnxEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        match self.embed_inner(text) {
            Ok(v) => v,
            Err(err) => {
                tracing::error!(?err, "onnx embedder failed, returning zero vector");
                vec![0.0; self.dim]
            }
        }
    }
}

impl OnnxEmbedder {
    fn embed_inner(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&u| u as i64).collect();
        let mut mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&u| u as i64)
            .collect();
        ids.truncate(self.max_len);
        mask.truncate(self.max_len);
        let seq_len = ids.len();

        let ids = Array2::from_shape_vec((1, seq_len), ids)?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), mask.clone())?;
        let type_ids = Array2::<i64>::zeros((1, seq_len));

        let mut session = self.session.lock().expect("onnx session mutex poisoned");

        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(ids)?,
            "attention_mask" => Tensor::from_array(mask_arr.clone())?,
            "token_type_ids" => Tensor::from_array(type_ids)?,
        ])?;

        // last_hidden_state: (1, seq_len, hidden)
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let hidden = shape[2] as usize;
        let arr = ndarray::ArrayView3::from_shape((1, seq_len, hidden), data)?;

        // Mean-pool with attention mask.
        let mask_f: Array1<f32> = mask_arr.row(0).mapv(|x| x as f32);
        let denom = mask_f.sum().max(1e-6);
        let mut pooled: Array1<f32> = Array1::zeros(hidden);
        for (t, weight) in mask_f.iter().enumerate() {
            if *weight > 0.0 {
                pooled += &arr.index_axis(Axis(1), t).index_axis(Axis(0), 0);
            }
        }
        pooled /= denom;

        let mut v: Vec<f32> = pooled.to_vec();
        // If the hidden size doesn't match `self.dim`, truncate or pad.
        v.resize(self.dim, 0.0);
        l2_normalize_in_place(&mut v);
        Ok(v)
    }
}
