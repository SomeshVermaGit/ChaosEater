"""
ChaosEater: Chaos Engineering Platform for ML Pipelines
Core Chaos Controller - Orchestrates experiments and manages fault injection
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


class ChaosLayer(Enum):
    DATA = "data"
    MODEL = "model"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class ExperimentMetrics:
    """Metrics collected during chaos experiment"""
    start_time: datetime
    end_time: Optional[datetime] = None
    recovery_time: Optional[float] = None  # seconds
    accuracy_degradation: Optional[float] = None  # percentage
    pipeline_downtime: Optional[float] = None  # seconds
    error_count: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "recovery_time": self.recovery_time,
            "accuracy_degradation": self.accuracy_degradation,
            "pipeline_downtime": self.pipeline_downtime,
            "error_count": self.error_count,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class ChaosExperiment:
    """Defines a chaos engineering experiment"""
    name: str
    layer: ChaosLayer
    fault_type: str
    parameters: Dict[str, Any]
    duration: int  # seconds
    status: ExperimentStatus = ExperimentStatus.PENDING
    metrics: Optional[ExperimentMetrics] = None
    rollback_strategy: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "layer": self.layer.value,
            "fault_type": self.fault_type,
            "parameters": self.parameters,
            "duration": self.duration,
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "rollback_strategy": self.rollback_strategy
        }


class ChaosController:
    """
    Main controller for chaos experiments
    Orchestrates fault injection, monitoring, and recovery
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.experiments: List[ChaosExperiment] = []
        self.injectors: Dict[ChaosLayer, Any] = {}
        self.observers: List[Any] = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ChaosEater - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def register_injector(self, layer: ChaosLayer, injector: Any):
        """Register a fault injector for specific layer"""
        self.injectors[layer] = injector
        self.logger.info(f"Registered injector for {layer.value} layer")

    def register_observer(self, observer: Any):
        """Register an observer for metrics collection"""
        self.observers.append(observer)
        self.logger.info(f"Registered observer: {observer.__class__.__name__}")

    def create_experiment(
        self,
        name: str,
        layer: ChaosLayer,
        fault_type: str,
        parameters: Dict[str, Any],
        duration: int = 60,
        rollback_strategy: Optional[str] = None
    ) -> ChaosExperiment:
        """Create a new chaos experiment"""
        experiment = ChaosExperiment(
            name=name,
            layer=layer,
            fault_type=fault_type,
            parameters=parameters,
            duration=duration,
            rollback_strategy=rollback_strategy
        )
        self.experiments.append(experiment)
        self.logger.info(f"Created experiment: {name} ({layer.value}/{fault_type})")
        return experiment

    def run_experiment(self, experiment: ChaosExperiment) -> ExperimentMetrics:
        """Execute a chaos experiment"""
        self.logger.info(f"Starting experiment: {experiment.name}")
        experiment.status = ExperimentStatus.RUNNING

        # Initialize metrics
        metrics = ExperimentMetrics(start_time=datetime.now())
        experiment.metrics = metrics

        try:
            # Get appropriate injector
            injector = self.injectors.get(experiment.layer)
            if not injector:
                raise ValueError(f"No injector registered for {experiment.layer.value}")

            # Notify observers - experiment start
            for observer in self.observers:
                observer.on_experiment_start(experiment)

            # Inject fault
            self.logger.info(f"Injecting fault: {experiment.fault_type}")
            injector.inject(experiment.fault_type, experiment.parameters, experiment.duration)

            # Monitor during experiment
            self._monitor_experiment(experiment)

            # Collect metrics
            metrics.end_time = datetime.now()
            metrics.recovery_time = (metrics.end_time - metrics.start_time).total_seconds()

            # Rollback
            if experiment.rollback_strategy:
                self.logger.info(f"Executing rollback: {experiment.rollback_strategy}")
                injector.rollback()

            experiment.status = ExperimentStatus.COMPLETED

            # Notify observers - experiment end
            for observer in self.observers:
                observer.on_experiment_end(experiment)

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            experiment.status = ExperimentStatus.FAILED
            metrics.error_count += 1

            # Attempt rollback on failure
            if experiment.layer in self.injectors:
                try:
                    self.injectors[experiment.layer].rollback()
                    experiment.status = ExperimentStatus.ROLLBACK
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {str(rollback_error)}")

            raise

        return metrics

    def _monitor_experiment(self, experiment: ChaosExperiment):
        """Monitor experiment execution and collect real-time metrics"""
        # This would integrate with monitoring stack (Prometheus, Grafana)
        # For now, placeholder for monitoring logic
        for observer in self.observers:
            if hasattr(observer, 'collect_metrics'):
                observer.collect_metrics(experiment)

    def run_batch(self, experiments: List[ChaosExperiment]) -> List[ExperimentMetrics]:
        """Run multiple experiments sequentially"""
        results = []
        for exp in experiments:
            try:
                metrics = self.run_experiment(exp)
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Batch experiment {exp.name} failed: {str(e)}")
                results.append(exp.metrics)
        return results

    def get_experiment_report(self, experiment: ChaosExperiment) -> Dict:
        """Generate detailed report for an experiment"""
        return {
            "experiment": experiment.to_dict(),
            "summary": self._generate_summary(experiment)
        }

    def _generate_summary(self, experiment: ChaosExperiment) -> Dict:
        """Generate summary statistics"""
        if not experiment.metrics:
            return {"status": "No metrics available"}

        metrics = experiment.metrics
        return {
            "duration": metrics.recovery_time,
            "status": experiment.status.value,
            "resilience_score": self._calculate_resilience_score(experiment),
            "errors": metrics.error_count
        }

    def _calculate_resilience_score(self, experiment: ChaosExperiment) -> float:
        """Calculate resilience score based on metrics (0-100)"""
        if not experiment.metrics:
            return 0.0

        metrics = experiment.metrics
        score = 100.0

        # Deduct for downtime
        if metrics.pipeline_downtime:
            score -= min(metrics.pipeline_downtime / 10, 40)  # max 40 point deduction

        # Deduct for accuracy degradation
        if metrics.accuracy_degradation:
            score -= min(metrics.accuracy_degradation, 30)  # max 30 point deduction

        # Deduct for errors
        score -= min(metrics.error_count * 5, 30)  # max 30 point deduction

        return max(score, 0.0)

    def export_metrics(self, filepath: str):
        """Export all experiment metrics to JSON"""
        data = {
            "experiments": [exp.to_dict() for exp in self.experiments],
            "summary": {
                "total_experiments": len(self.experiments),
                "completed": sum(1 for e in self.experiments if e.status == ExperimentStatus.COMPLETED),
                "failed": sum(1 for e in self.experiments if e.status == ExperimentStatus.FAILED)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Metrics exported to {filepath}")
