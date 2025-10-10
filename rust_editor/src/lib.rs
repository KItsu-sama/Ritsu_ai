mod error;
mod cpu;
mod gpu;

use error::RustEditorError;
use crate::cpu::CPUExecutor;
use crate::gpu::GPUExecutor;

mod analysis_result;
pub use analysis_result::AnalysisResult;

pub trait CodeExecutor: Send + Sync {
    fn analyze_code(&self, code: &str) -> Result<AnalysisResult, RustEditorError>;
    fn format_code(&self, code: &str) -> Result<String, RustEditorError>;
    // ... other methods
}

// We define AnalysisResult here for now.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    // ... we'll define later
}

pub struct RustEditor {
    executor: Box<dyn CodeExecutor>,
}

impl RustEditor {
    pub fn new(use_gpu: bool) -> Result<Self, RustEditorError> {
        let executor: Box<dyn CodeExecutor> = if use_gpu {
            match GPUExecutor::new() {
                Ok(gpu_executor) => Box::new(gpu_executor),
                Err(e) => {
                    eprintln!("Failed to initialize GPU executor: {}. Falling back to CPU.", e);
                    Box::new(CPUExecutor::new())
                }
            }
        } else {
            Box::new(CPUExecutor::new())
        };

        Ok(Self { executor })
    }

    pub fn analyze_code(&self, code: &str) -> Result<AnalysisResult, RustEditorError> {
        self.executor.analyze_code(code)
    }

    pub fn format_code(&self, code: &str) -> Result<String, RustEditorError> {
        self.executor.format_code(code)
    }
}

// We also need to implement the Python bindings.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn ritsu_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustEditor>()?;
    // ... other bindings
    Ok(())
}
