"""
Model Layer Fault Injector
Simulates model-related failures: drift, version issues, inference problems, memory leaks
"""

import time
import random
import threading
from typing import Dict, Any, Optional
from .base_injector import BaseFaultInjector
import pickle


class ModelFaultInjector(BaseFaultInjector):
    """Injects faults into model layer"""

    SUPPORTED_FAULTS = [
        "model_drift",
        "wrong_version",
        "slow_inference",
        "memory_leak",
        "prediction_corruption",
        "model_unavailable",
        "parameter_degradation",
        "batch_processing_failure"
    ]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.original_model = None
        self.model_wrapper = None
        self.memory_leak_active = False
        self.leak_thread = None

    def inject(self, fault_type: str, parameters: Dict[str, Any], duration: int):
        """Inject model layer fault"""
        if not self.validate(fault_type, parameters):
            raise ValueError(f"Invalid parameters for fault type: {fault_type}")

        self.logger.info(f"Injecting {fault_type} with params: {parameters}")

        if fault_type == "model_drift":
            self._inject_model_drift(parameters)
        elif fault_type == "wrong_version":
            self._inject_wrong_version(parameters)
        elif fault_type == "slow_inference":
            self._inject_slow_inference(parameters, duration)
        elif fault_type == "memory_leak":
            self._inject_memory_leak(parameters, duration)
        elif fault_type == "prediction_corruption":
            self._inject_prediction_corruption(parameters)
        elif fault_type == "model_unavailable":
            self._inject_model_unavailable(parameters, duration)
        elif fault_type == "parameter_degradation":
            self._inject_parameter_degradation(parameters)
        elif fault_type == "batch_processing_failure":
            self._inject_batch_failure(parameters)
        else:
            raise ValueError(f"Unsupported fault type: {fault_type}")

        self.active_faults.append(fault_type)

    def _inject_model_drift(self, params: Dict[str, Any]):
        """Simulate concept drift in model performance"""
        drift_rate = params.get("drift_rate", 0.1)
        drift_type = params.get("drift_type", "gradual")  # gradual or sudden

        class DriftedModelWrapper:
            def __init__(self, model, drift_rate, drift_type):
                self.model = model
                self.drift_rate = drift_rate
                self.drift_type = drift_type
                self.predictions_count = 0

            def predict(self, X):
                self.predictions_count += 1

                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(X)

                    # Calculate drift degradation
                    if self.drift_type == "gradual":
                        degradation = min(self.predictions_count * 0.001, self.drift_rate)
                    else:  # sudden
                        degradation = self.drift_rate

                    # Add noise to simulate drift
                    if hasattr(predictions, 'shape'):
                        import numpy as np
                        noise = np.random.normal(0, degradation, predictions.shape)
                        predictions = predictions + noise

                    return predictions
                else:
                    raise AttributeError("Model has no predict method")

            def __getattr__(self, name):
                return getattr(self.model, name)

        self.model_wrapper = lambda model: DriftedModelWrapper(model, drift_rate, drift_type)
        self.logger.info(f"Model drift injected: {drift_type} drift at rate {drift_rate}")

    def _inject_wrong_version(self, params: Dict[str, Any]):
        """Deploy wrong model version"""
        fallback_model = params.get("fallback_model", None)

        class WrongVersionWrapper:
            def __init__(self, fallback_model):
                self.fallback_model = fallback_model

            def predict(self, X):
                if self.fallback_model:
                    return self.fallback_model.predict(X)
                else:
                    # Return degraded predictions
                    import numpy as np
                    if hasattr(X, 'shape'):
                        return np.random.rand(*X.shape)
                    return None

            def __getattr__(self, name):
                if self.fallback_model:
                    return getattr(self.fallback_model, name)
                raise AttributeError(f"Wrong model version - attribute {name} not found")

        self.model_wrapper = lambda model: WrongVersionWrapper(fallback_model)
        self.logger.info("Wrong model version injected")

    def _inject_slow_inference(self, params: Dict[str, Any], duration: int):
        """Add latency to model inference"""
        latency_ms = params.get("latency_ms", 500)
        latency_variance = params.get("variance", 0.2)

        class SlowInferenceWrapper:
            def __init__(self, model, latency_ms, variance):
                self.model = model
                self.latency_ms = latency_ms
                self.variance = variance

            def predict(self, X):
                # Add random latency
                actual_latency = self.latency_ms * (1 + random.uniform(-self.variance, self.variance))
                time.sleep(actual_latency / 1000.0)

                if hasattr(self.model, 'predict'):
                    return self.model.predict(X)
                return None

            def __getattr__(self, name):
                return getattr(self.model, name)

        self.model_wrapper = lambda model: SlowInferenceWrapper(model, latency_ms, latency_variance)
        self.logger.info(f"Slow inference injected: {latency_ms}ms ±{latency_variance * 100}%")

    def _inject_memory_leak(self, params: Dict[str, Any], duration: int):
        """Simulate memory leak during inference"""
        leak_rate_mb = params.get("leak_rate_mb", 10)

        self.memory_leak_active = True
        leaked_memory = []

        def leak_memory():
            while self.memory_leak_active:
                # Allocate memory
                leaked_memory.append(bytearray(leak_rate_mb * 1024 * 1024))
                time.sleep(1)

        self.leak_thread = threading.Thread(target=leak_memory, daemon=True)
        self.leak_thread.start()

        self.logger.info(f"Memory leak injected: {leak_rate_mb}MB/s")

    def _inject_prediction_corruption(self, params: Dict[str, Any]):
        """Corrupt model predictions"""
        corruption_rate = params.get("corruption_rate", 0.1)
        corruption_type = params.get("type", "random")  # random, bias, flip

        class CorruptedPredictionWrapper:
            def __init__(self, model, corruption_rate, corruption_type):
                self.model = model
                self.corruption_rate = corruption_rate
                self.corruption_type = corruption_type

            def predict(self, X):
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(X)

                    import numpy as np
                    mask = np.random.random(predictions.shape[0]) < self.corruption_rate

                    if self.corruption_type == "random":
                        if len(predictions.shape) > 1:
                            predictions[mask] = np.random.rand(mask.sum(), predictions.shape[1])
                        else:
                            predictions[mask] = np.random.rand(mask.sum())
                    elif self.corruption_type == "bias":
                        predictions[mask] *= 1.5
                    elif self.corruption_type == "flip":
                        predictions[mask] = 1 - predictions[mask]

                    return predictions
                return None

            def __getattr__(self, name):
                return getattr(self.model, name)

        self.model_wrapper = lambda model: CorruptedPredictionWrapper(model, corruption_rate, corruption_type)
        self.logger.info(f"Prediction corruption injected: {corruption_rate * 100}% ({corruption_type})")

    def _inject_model_unavailable(self, params: Dict[str, Any], duration: int):
        """Make model unavailable (simulate service down)"""
        failure_rate = params.get("failure_rate", 1.0)

        class UnavailableModelWrapper:
            def __init__(self, failure_rate):
                self.failure_rate = failure_rate

            def predict(self, X):
                if random.random() < self.failure_rate:
                    raise RuntimeError("Model service unavailable")
                return None

            def __getattr__(self, name):
                if random.random() < self.failure_rate:
                    raise RuntimeError("Model service unavailable")
                raise AttributeError(f"Model unavailable - attribute {name} not accessible")

        self.model_wrapper = lambda model: UnavailableModelWrapper(failure_rate)
        self.logger.info(f"Model unavailable injected: {failure_rate * 100}% failure rate")

    def _inject_parameter_degradation(self, params: Dict[str, Any]):
        """Degrade model parameters"""
        degradation_factor = params.get("degradation_factor", 0.1)

        class DegradedParameterWrapper:
            def __init__(self, model, degradation_factor):
                self.model = model
                self.degradation_factor = degradation_factor
                self._degrade_parameters()

            def _degrade_parameters(self):
                # This would modify model weights/parameters
                # Implementation depends on model framework (TensorFlow, PyTorch, etc.)
                pass

            def predict(self, X):
                if hasattr(self.model, 'predict'):
                    return self.model.predict(X)
                return None

            def __getattr__(self, name):
                return getattr(self.model, name)

        self.model_wrapper = lambda model: DegradedParameterWrapper(model, degradation_factor)
        self.logger.info(f"Parameter degradation injected: {degradation_factor * 100}%")

    def _inject_batch_failure(self, params: Dict[str, Any]):
        """Fail batch processing randomly"""
        failure_rate = params.get("failure_rate", 0.2)

        class BatchFailureWrapper:
            def __init__(self, model, failure_rate):
                self.model = model
                self.failure_rate = failure_rate

            def predict(self, X):
                if random.random() < self.failure_rate:
                    raise RuntimeError("Batch processing failed")

                if hasattr(self.model, 'predict'):
                    return self.model.predict(X)
                return None

            def __getattr__(self, name):
                return getattr(self.model, name)

        self.model_wrapper = lambda model: BatchFailureWrapper(model, failure_rate)
        self.logger.info(f"Batch failure injected: {failure_rate * 100}% failure rate")

    def wrap_model(self, model):
        """Apply fault wrapper to model"""
        if self.model_wrapper:
            return self.model_wrapper(model)
        return model

    def rollback(self):
        """Remove all model faults"""
        self.logger.info("Rolling back model faults")

        # Stop memory leak
        if self.memory_leak_active:
            self.memory_leak_active = False
            if self.leak_thread:
                self.leak_thread.join(timeout=2)

        self.model_wrapper = None
        self.original_model = None
        self.active_faults.clear()

    def validate(self, fault_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate fault parameters"""
        if fault_type not in self.SUPPORTED_FAULTS:
            return False

        # Validate rates
        for rate_param in ["drift_rate", "corruption_rate", "failure_rate"]:
            if rate_param in parameters:
                rate = parameters[rate_param]
                if not (0 <= rate <= 1):
                    return False

        return True

    def get_supported_faults(self) -> list:
        """Return supported fault types"""
        return self.SUPPORTED_FAULTS
