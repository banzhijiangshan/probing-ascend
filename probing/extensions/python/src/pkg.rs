//! Small synchronous Python wrappers around async support crates.

use probing_store::store::TCPStore as RustTCPStore;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

#[pyclass]
pub struct TCPStore {
    inner: RustTCPStore,
    runtime: Runtime,
}

#[pymethods]
impl TCPStore {
    #[new]
    fn new(endpoint: String) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|error| PyRuntimeError::new_err(format!("create Tokio runtime: {error}")))?;
        Ok(Self {
            inner: RustTCPStore::new(endpoint),
            runtime,
        })
    }

    fn set(&self, key: &str, value: &str) -> PyResult<()> {
        self.runtime
            .block_on(self.inner.set(key, value))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn get(&self, key: &str) -> PyResult<String> {
        self.runtime
            .block_on(self.inner.get(key))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}
