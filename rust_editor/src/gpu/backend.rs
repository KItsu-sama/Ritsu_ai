use thiserror::Error;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("OpenCL error: {0}")]
    OpenCL(String),
    #[error("Vulkan error: {0}")]
    Vulkan(String),
    #[error("Metal error: {0}")]
    Metal(String),
    #[error("No GPU backend available")]
    NoBackendAvailable,
    // ... other errors
}

pub type Result<T> = std::result::Result<T, BackendError>;

pub trait GPUBackend: Send + Sync {
    fn analyze_code(&self, code: &str) -> Result<AnalysisResult>;
    fn format_code(&self, code: &str) -> Result<String>;
    // ... other methods
}

// We also define the `AnalysisResult` and other types here.

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    // ... fields
}

// Then, we provide the `create_default_backend` function.

pub fn create_default_backend() -> Result<Box<dyn GPUBackend>> {
    // Try each backend in order of preference.
    // Note: We conditionally compile each backend.

    #[cfg(feature = "cuda")]
    if let Ok(backend) = cuda::CudaBackend::new() {
        return Ok(Box::new(backend));
    }

    #[cfg(feature = "metal")]
    if let Ok(backend) = metal::MetalBackend::new() {
        return Ok(Box::new(backend));
    }

    #[cfg(feature = "vulkan")]
    if let Ok(backend) = vulkan::VulkanBackend::new() {
        return Ok(Box::new(backend));
    }

    #[cfg(feature = "opencl")]
    if let Ok(backend) = opencl::OpenCLBackend::new() {
        return Ok(Box::new(backend));
    }

    Err(BackendError::NoBackendAvailable)
}

