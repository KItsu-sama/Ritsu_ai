use super::backend::{GPUBackend, create_default_backend};

pub struct GPUExecutor {
    backend: Box<dyn GPUBackend>,
}

impl GPUExecutor {
    pub fn new() -> Result<Self, BackendError> {
        let backend = create_default_backend()?;
        Ok(Self { backend })
    }

    // Then, we delegate the methods to the backend.

    pub fn analyze_code(&self, code: &str) -> Result<AnalysisResult> {
        self.backend.analyze_code(code)
    }

    // ... etc.
}