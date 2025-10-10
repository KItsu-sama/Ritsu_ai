use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GPUError {
    #[error("No compatible GPU backend found")]
    NoBackendAvailable,
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("OpenCL error: {0}")]
    OpenCLError(String),
    #[error("Vulkan error: {0}")]
    VulkanError(String),
    #[error("Metal error: {0}")]
    MetalError(String),
    #[error("CPU fallback error: {0}")]
    CpuError(String),
}

pub type Result<T> = std::result::Result<T, GPUError>;

#[derive(Debug, Clone, PartialEq)]
pub enum GPUBackend {
    Cuda,
    OpenCL,
    Vulkan,
    Metal,
    Cpu,  // Fallback to CPU parallel processing
}

#[derive(Debug)]
pub struct GPUExecutor {
    backend: GPUBackend,
    available_backends: Vec<GPUBackend>,
    device_memory: u64,
    is_initialized: bool,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub backend_used: GPUBackend,
    pub execution_time: f64,
    pub memory_used: u64,
    pub success: bool,
}

impl GPUExecutor {
    pub fn new() -> Result<Self> {
        let available_backends = Self::detect_available_backends();
        
        if available_backends.is_empty() {
            return Err(GPUError::NoBackendAvailable);
        }

        // Auto-select the best available backend
        let backend = available_backends.first()
            .cloned()
            .unwrap_or(GPUBackend::Cpu);

        Ok(Self {
            backend,
            available_backends,
            device_memory: 0,
            is_initialized: false,
        })
    }

    pub fn with_backend(backend: GPUBackend) -> Result<Self> {
        let available_backends = Self::detect_available_backends();
        
        if !available_backends.contains(&backend) {
            return Err(GPUError::NoBackendAvailable);
        }

        Ok(Self {
            backend,
            available_backends,
            device_memory: 0,
            is_initialized: false,
        })
    }

    /// SINGLE centralized GPU detection function
    fn detect_available_backends() -> Vec<GPUBackend> {
        let mut backends = Vec::new();

        // Check CUDA
        if Self::check_cuda_available() {
            backends.push(GPUBackend::Cuda);
        }

        // Check OpenCL  
        if Self::check_opencl_available() {
            backends.push(GPUBackend::OpenCL);
        }

        // Check Vulkan
        if Self::check_vulkan_available() {
            backends.push(GPUBackend::Vulkan);
        }

        // Check Metal (macOS only)
        if Self::check_metal_available() {
            backends.push(GPUBackend::Metal);
        }

        // Always include CPU fallback
        backends.push(GPUBackend::Cpu);

        backends
    }

    fn check_cuda_available() -> bool {
        cfg!(feature = "cuda") && {
            // Actual CUDA detection logic
            #[cfg(feature = "cuda")]
            {
                cuda::is_available()
            }
            #[cfg(not(feature = "cuda"))]
            false
        }
    }

    fn check_opencl_available() -> bool {
        cfg!(feature = "opencl") && {
            // Actual OpenCL detection logic
            #[cfg(feature = "opencl")]
            {
                opencl::is_available()
            }
            #[cfg(not(feature = "opencl"))]
            false
        }
    }

    fn check_vulkan_available() -> bool {
        cfg!(feature = "vulkan") && {
            // Actual Vulkan detection logic
            #[cfg(feature = "vulkan")]
            {
                vulkan::is_available()
            }
            #[cfg(not(feature = "vulkan"))]
            false
        }
    }

    fn check_metal_available() -> bool {
        cfg!(target_os = "macos") && cfg!(feature = "metal") && {
            // Actual Metal detection logic
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal::is_available()
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            false
        }
    }

    pub fn initialize(&mut self) -> Result<()> {
        if self.is_initialized {
            return Ok(());
        }

        match self.backend {
            GPUBackend::Cuda => self.initialize_cuda(),
            GPUBackend::OpenCL => self.initialize_opencl(),
            GPUBackend::Vulkan => self.initialize_vulkan(),
            GPUBackend::Metal => self.initialize_metal(),
            GPUBackend::Cpu => self.initialize_cpu(),
        }?;

        self.is_initialized = true;
        Ok(())
    }

    /// MAIN execution interface - single entry point for all GPU operations
    pub fn execute_analysis(&self, code: &str, operation: AnalysisOperation) -> Result<ExecutionResult> {
        if !self.is_initialized {
            return Err(GPUError::CpuError("Executor not initialized".to_string()));
        }

        let start_time = std::time::Instant::now();

        let result = match self.backend {
            GPUBackend::Cuda => self.execute_cuda(code, operation),
            GPUBackend::OpenCL => self.execute_opencl(code, operation),
            GPUBackend::Vulkan => self.execute_vulkan(code, operation),
            GPUBackend::Metal => self.execute_metal(code, operation),
            GPUBackend::Cpu => self.execute_cpu(code, operation),
        };

        let execution_time = start_time.elapsed().as_secs_f64();

        match result {
            Ok(_) => Ok(ExecutionResult {
                backend_used: self.backend.clone(),
                execution_time,
                memory_used: self.device_memory,
                success: true,
            }),
            Err(e) => {
                log::error!("Execution failed on {:?}: {}", self.backend, e);
                Err(e)
            }
        }
    }

    pub fn execute_batch_analysis(&self, files: Vec<String>, operation: AnalysisOperation) -> Result<Vec<ExecutionResult>> {
        // Batch processing logic that uses the same centralized execution
        files.into_iter()
            .map(|file| self.execute_analysis(&file, operation.clone()))
            .collect()
    }

    // Backend-specific implementations (private)
    fn initialize_cuda(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            match cuda::initialize() {
                Ok(memory) => {
                    self.device_memory = memory;
                    Ok(())
                }
                Err(e) => Err(GPUError::CudaError(e.to_string())),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(GPUError::CudaError("CUDA support not compiled".to_string()))
    }

    fn execute_cuda(&self, code: &str, operation: AnalysisOperation) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            cuda::analyze_code(code, operation)
                .map_err(|e| GPUError::CudaError(e.to_string()))
        }
        #[cfg(not(feature = "cuda"))]
        Err(GPUError::CudaError("CUDA support not compiled".to_string()))
    }

    // Similar implementations for OpenCL, Vulkan, Metal, CPU...
    fn initialize_opencl(&mut self) -> Result<()> { /* ... */ }
    fn execute_opencl(&self, code: &str, operation: AnalysisOperation) -> Result<()> { /* ... */ }
    
    fn initialize_vulkan(&mut self) -> Result<()> { /* ... */ }
    fn execute_vulkan(&self, code: &str, operation: AnalysisOperation) -> Result<()> { /* ... */ }
    
    fn initialize_metal(&mut self) -> Result<()> { /* ... */ }
    fn execute_metal(&self, code: &str, operation: AnalysisOperation) -> Result<()> { /* ... */ }
    
    fn initialize_cpu(&mut self) -> Result<()> { 
        // Initialize thread pool for CPU parallel processing
        self.device_memory = 0; // CPU doesn't have device memory
        Ok(())
    }
    
    fn execute_cpu(&self, code: &str, operation: AnalysisOperation) -> Result<()> {
        // Fallback to CPU parallel processing
        cpu::analyze_code_parallel(code, operation)
            .map_err(|e| GPUError::CpuError(e.to_string()))
    }

    // Getters and utility methods
    pub fn available_backends(&self) -> &[GPUBackend] {
        &self.available_backends
    }

    pub fn current_backend(&self) -> &GPUBackend {
        &self.backend
    }

    pub fn switch_backend(&mut self, backend: GPUBackend) -> Result<()> {
        if !self.available_backends.contains(&backend) {
            return Err(GPUError::NoBackendAvailable);
        }
        
        self.backend = backend;
        self.is_initialized = false;
        self.initialize()
    }
}

#[derive(Debug, Clone)]
pub enum AnalysisOperation {
    SyntaxAnalysis,
    ComplexityAnalysis,
    PatternDetection,
    CodeFormatting,
    Custom(String),
}

// Simplified Python bindings
#[pyclass]
pub struct PyGPUExecutor {
    inner: GPUExecutor,
}

#[pymethods]
impl PyGPUExecutor {
    #[new]
    pub fn new() -> PyResult<Self> {
        let executor = GPUExecutor::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: executor })
    }

    pub fn analyze_code(&mut self, code: String, operation: String) -> PyResult<PyExecutionResult> {
        let op = match operation.as_str() {
            "syntax" => AnalysisOperation::SyntaxAnalysis,
            "complexity" => AnalysisOperation::ComplexityAnalysis,
            "patterns" => AnalysisOperation::PatternDetection,
            "formatting" => AnalysisOperation::CodeFormatting,
            custom => AnalysisOperation::Custom(custom.to_string()),
        };

        self.inner.initialize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let result = self.inner.execute_analysis(&code, op)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyExecutionResult::from(result))
    }
}