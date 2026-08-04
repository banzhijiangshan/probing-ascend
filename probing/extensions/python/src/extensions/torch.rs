use std::collections::HashMap;

use async_trait::async_trait;
use probing_core::core::EngineError;
use probing_core::core::Maybe;
use probing_core::core::ProbeExtension;
use probing_core::core::ProbeExtensionCall;
use probing_core::core::ProbeExtensionOption;
use pyo3::prelude::*;

/// Truth key for live torch profiling sync (Python ``probing.config``).
pub const TRUTH_KEY_PROFILING: &str = "probing.torch.profiling";
/// Deprecated extension/SQL SET alias — forwards to [`TRUTH_KEY_PROFILING`] with a warn.
pub const DEPRECATED_ALIAS_PROFILING: &str = "torch.profiling";

#[derive(Debug, Default)]
pub struct TorchProbeExtension {
    /// Combined PyTorch profiling specification string (see TorchProbeConfig).
    profiling: Maybe<String>,
}

#[async_trait]
impl ProbeExtensionCall for TorchProbeExtension {
    async fn call(
        &self,
        path: &str,
        params: &HashMap<String, String>,
        _body: &[u8],
    ) -> Result<Vec<u8>, EngineError> {
        match path.trim_start_matches('/') {
            "flamegraph" => Ok(crate::features::torch::flamegraph().into_bytes()),
            "flamegraph/json" => {
                let metric = params.get("metric").map(|s| s.as_str());
                Ok(crate::features::torch::flamegraph_json(metric).into_bytes())
            }
            _ => Err(EngineError::UnsupportedCall),
        }
    }
}

impl TorchProbeExtension {
    pub const OPTION_PROFILING: &'static str = TRUTH_KEY_PROFILING;

    fn set_profiling(&mut self, profiling: Maybe<String>) -> Result<(), EngineError> {
        // IMPORTANT (Ascend/NPU): never call torch_probe.configure() from this
        // Tokio/env-sync path. configure() may `import torch` and register hooks;
        // doing that off the training main thread SIGABRTs on torch_npu images.
        // Only persist the spec into probing.config; main-thread activation
        // (import hook / ext.torch.init / first optimizer.step) picks it up.
        let py_result = Python::attach(|py| -> pyo3::PyResult<()> {
            let probing = py.import("probing")?;
            let config = probing.getattr("config")?;
            match &profiling {
                Maybe::Just(spec) if !spec.trim().is_empty() => {
                    config.call_method1("set", (TRUTH_KEY_PROFILING, spec.as_str()))?;
                }
                _ => {
                    let _ = config.call_method1("remove", (TRUTH_KEY_PROFILING,));
                }
            }
            Ok(())
        });

        match py_result {
            Ok(()) => {
                self.profiling = profiling;
                Ok(())
            }
            Err(err) => {
                let value: String = profiling.clone().into();
                log::error!("Failed to store torch profiling spec '{}': {}", value, err);
                Err(EngineError::InvalidOptionValue(
                    Self::OPTION_PROFILING.to_string(),
                    value,
                ))
            }
        }
    }

    fn parse_profiling(value: &str) -> Result<Maybe<String>, EngineError> {
        value
            .parse()
            .map_err(|_| EngineError::InvalidOptionValue(value.to_string(), value.to_string()))
    }
}

impl ProbeExtension for TorchProbeExtension {
    fn name(&self) -> String {
        "torchextension".to_string()
    }

    fn get(&self, key: &str) -> Result<String, EngineError> {
        match key {
            TRUTH_KEY_PROFILING | DEPRECATED_ALIAS_PROFILING | "profiling_mode" => {
                Ok(self.profiling.to_string())
            }
            _ => Err(EngineError::UnsupportedOption(key.to_string())),
        }
    }

    fn set(&mut self, key: &str, value: &str) -> Result<String, EngineError> {
        if key == DEPRECATED_ALIAS_PROFILING {
            log::warn!(
                "SET key '{}' is deprecated, use '{}'",
                DEPRECATED_ALIAS_PROFILING,
                TRUTH_KEY_PROFILING
            );
        }
        match key {
            TRUTH_KEY_PROFILING | DEPRECATED_ALIAS_PROFILING | "profiling_mode" => {
                let old = self.profiling.to_string();
                let new = Self::parse_profiling(value)?;
                self.set_profiling(new)?;
                Ok(old)
            }
            _ => Err(EngineError::UnsupportedOption(key.to_string())),
        }
    }

    fn options(&self) -> Vec<ProbeExtensionOption> {
        vec![ProbeExtensionOption {
            key: TRUTH_KEY_PROFILING.to_string(),
            value: Some(self.profiling.to_string()),
            help: "PyTorch profiling spec (truth key). Deprecated alias: torch.profiling.",
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deprecated_torch_profiling_alias_sets_truth_key() {
        let mut ext = TorchProbeExtension::default();
        let old = ext
            .set(DEPRECATED_ALIAS_PROFILING, "on,rate=0.5")
            .expect("deprecated alias should forward");
        assert!(old.is_empty() || old == "None" || old == "");
        let cur = ext.get(TRUTH_KEY_PROFILING).expect("truth key readable");
        assert!(cur.contains("on"));
        assert!(cur.contains("rate=0.5"));
    }

    #[test]
    fn truth_key_probing_torch_profiling_works() {
        let mut ext = TorchProbeExtension::default();
        ext.set(TRUTH_KEY_PROFILING, "on,rate=1.0")
            .expect("truth key set");
        let cur = ext.get(TRUTH_KEY_PROFILING).unwrap();
        assert!(cur.contains("rate=1.0"));
    }
}
