use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustEditorError {
    #[error("GPU backend error: {0}")]
    GPUBackendError(String),
    #[error("GPU analysis error: {0}")]
    GPUAnalysisError(String),
    #[error("GPU format error: {0}")]
    GPUFormatError(String),
    #[error("CPU analysis error: {0}")]
    CPUAnalysisError(String),
    #[error("CPU format error: {0}")]
    CPUFormatError(String),
    #[error("No executor available")]
    NoExecutorAvailable,
}