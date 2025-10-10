pub mod backend;
mod cuda;
mod opencl;
mod vulkan;
mod metal;

use crate::CodeExecutor;
use crate::RustEditorError;
use backend::{BackendError, create_default_backend};

pub struct GPUExecutor {
    backend: Box<dyn backend::GPUBackend>,
}

impl GPUExecutor {
    pub fn new() -> Result<Self, RustEditorError> {
        let backend = create_default_backend()
            .map_err(|e| RustEditorError::GPUBackendError(e.to_string()))?;
        Ok(Self { backend })
    }
}

impl CodeExecutor for GPUExecutor {
    fn analyze_code(&self, code: &str) -> Result<super::AnalysisResult, RustEditorError> {
        self.backend.analyze_code(code)
            .map_err(|e| RustEditorError::GPUAnalysisError(e.to_string()))
    }

    fn format_code(&self, code: &str) -> Result<String, RustEditorError> {
        self.backend.format_code(code)
            .map_err(|e| RustEditorError::GPUFormatError(e.to_string()))
    }
}

pub mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "opencl")]
pub mod opencl;

pub use backend::{create_default_backend, BackendError, GPUBackend, AnalysisResult};