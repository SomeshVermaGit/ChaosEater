"""
Experiment Orchestrator
Manages experiment scheduling, execution, and CI/CD integration
"""

import time
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime
from chaos_controller import ChaosController, ChaosExperiment, ChaosLayer
from fault_injectors.data_injector import DataFaultInjector
from fault_injectors.model_injector import ModelFaultInjector
from fault_injectors.infra_injector import InfrastructureFaultInjector
from observers.metrics_observer import MetricsObserver, PrometheusObserver, LogObserver


class ExperimentOrchestrator:
    """
    Orchestrates chaos experiments from configuration files
    Supports scheduled experiments and CI/CD integration
    """

    def __init__(self, config_file: Optional[str] = None):
        self.controller = ChaosController()
        self.config = self._load_config(config_file) if config_file else {}
        self.experiments: List[ChaosExperiment] = []

        # Register injectors
        self._setup_injectors()

        # Register observers
        self._setup_observers()

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _setup_injectors(self):
        """Setup fault injectors for all layers"""
        data_injector = DataFaultInjector()
        model_injector = ModelFaultInjector()
        infra_injector = InfrastructureFaultInjector()

        self.controller.register_injector(ChaosLayer.DATA, data_injector)
        self.controller.register_injector(ChaosLayer.MODEL, model_injector)
        self.controller.register_injector(ChaosLayer.INFRASTRUCTURE, infra_injector)

    def _setup_observers(self):
        """Setup observers for metrics collection"""
        # Default metrics observer
        metrics_observer = MetricsObserver({"auto_export": True})
        self.controller.register_observer(metrics_observer)

        # Log observer
        log_observer = LogObserver()
        self.controller.register_observer(log_observer)

        # Prometheus observer (if configured)
        if "prometheus" in self.config:
            prom_config = self.config["prometheus"]
            prom_observer = PrometheusObserver(
                pushgateway_url=prom_config.get("pushgateway_url"),
                job_name=prom_config.get("job_name", "chaos_eater")
            )
            self.controller.register_observer(prom_observer)

    def load_experiments_from_config(self, config_file: str):
        """Load experiments from configuration file"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        experiments_config = config.get("experiments", [])

        for exp_config in experiments_config:
            layer = ChaosLayer(exp_config["layer"])
            experiment = self.controller.create_experiment(
                name=exp_config["name"],
                layer=layer,
                fault_type=exp_config["fault_type"],
                parameters=exp_config.get("parameters", {}),
                duration=exp_config.get("duration", 60),
                rollback_strategy=exp_config.get("rollback_strategy")
            )
            self.experiments.append(experiment)

    def run_all_experiments(self) -> List[Dict]:
        """Run all loaded experiments"""
        results = []

        for experiment in self.experiments:
            try:
                print(f"\n{'='*60}")
                print(f"Running: {experiment.name}")
                print(f"Layer: {experiment.layer.value} | Fault: {experiment.fault_type}")
                print(f"{'='*60}\n")

                metrics = self.controller.run_experiment(experiment)
                report = self.controller.get_experiment_report(experiment)
                results.append(report)

                # Print summary
                self._print_summary(experiment, report)

            except Exception as e:
                print(f"❌ Experiment {experiment.name} failed: {str(e)}")
                results.append({
                    "experiment": experiment.to_dict(),
                    "error": str(e)
                })

        return results

    def run_experiment_by_name(self, name: str) -> Optional[Dict]:
        """Run specific experiment by name"""
        for experiment in self.experiments:
            if experiment.name == name:
                try:
                    metrics = self.controller.run_experiment(experiment)
                    return self.controller.get_experiment_report(experiment)
                except Exception as e:
                    return {"error": str(e)}
        return None

    def schedule_experiments(self, interval_seconds: int = 3600):
        """Schedule experiments to run periodically"""
        print(f"Scheduling experiments to run every {interval_seconds}s")

        while True:
            print(f"\n[{datetime.now()}] Running scheduled experiments...")
            self.run_all_experiments()
            print(f"Sleeping for {interval_seconds}s until next run...")
            time.sleep(interval_seconds)

    def export_results(self, filepath: str = "experiment_results.json"):
        """Export all experiment results"""
        self.controller.export_metrics(filepath)
        print(f"Results exported to {filepath}")

    def _print_summary(self, experiment, report):
        """Print experiment summary"""
        summary = report["summary"]
        print(f"\n📊 Summary for {experiment.name}:")
        print(f"  Status: {summary.get('status', 'N/A')}")
        print(f"  Duration: {summary.get('duration', 0):.2f}s")
        print(f"  Resilience Score: {summary.get('resilience_score', 0):.1f}/100")
        print(f"  Errors: {summary.get('errors', 0)}")

    def validate_experiments(self) -> bool:
        """Validate all experiments before running"""
        all_valid = True

        for experiment in self.experiments:
            injector = self.controller.injectors.get(experiment.layer)
            if not injector:
                print(f"❌ No injector for {experiment.layer.value}")
                all_valid = False
                continue

            if not injector.validate(experiment.fault_type, experiment.parameters):
                print(f"❌ Invalid parameters for {experiment.name}")
                all_valid = False

        return all_valid

    def create_ci_cd_gate(self, min_resilience_score: float = 70.0) -> bool:
        """
        Run experiments as CI/CD gate
        Returns True if all experiments pass resilience threshold
        """
        print(f"🚦 CI/CD Gate: Minimum resilience score = {min_resilience_score}")

        results = self.run_all_experiments()
        passed = True

        for result in results:
            summary = result.get("summary", {})
            score = summary.get("resilience_score", 0)

            if score < min_resilience_score:
                print(f"❌ Failed: {result['experiment']['name']} (score: {score:.1f})")
                passed = False
            else:
                print(f"✅ Passed: {result['experiment']['name']} (score: {score:.1f})")

        return passed


def main():
    """Example usage"""
    orchestrator = ExperimentOrchestrator()

    # Create sample experiments
    print("Creating chaos experiments...\n")

    # Data layer experiments
    orchestrator.controller.create_experiment(
        name="data_corruption_test",
        layer=ChaosLayer.DATA,
        fault_type="data_corruption",
        parameters={"corruption_rate": 0.1, "columns": None},
        duration=30
    )

    orchestrator.controller.create_experiment(
        name="distribution_drift_test",
        layer=ChaosLayer.DATA,
        fault_type="distribution_drift",
        parameters={"magnitude": 0.3},
        duration=30
    )

    # Model layer experiments
    orchestrator.controller.create_experiment(
        name="model_drift_test",
        layer=ChaosLayer.MODEL,
        fault_type="model_drift",
        parameters={"drift_rate": 0.15, "drift_type": "gradual"},
        duration=30
    )

    orchestrator.controller.create_experiment(
        name="slow_inference_test",
        layer=ChaosLayer.MODEL,
        fault_type="slow_inference",
        parameters={"latency_ms": 500, "variance": 0.2},
        duration=30
    )

    # Infrastructure layer experiments
    orchestrator.controller.create_experiment(
        name="network_latency_test",
        layer=ChaosLayer.INFRASTRUCTURE,
        fault_type="network_latency",
        parameters={"latency_ms": 100, "jitter_ms": 20},
        duration=30
    )

    orchestrator.controller.create_experiment(
        name="memory_pressure_test",
        layer=ChaosLayer.INFRASTRUCTURE,
        fault_type="memory_pressure",
        parameters={"memory_mb": 512},
        duration=30
    )

    # Run experiments
    orchestrator.experiments = orchestrator.controller.experiments
    results = orchestrator.run_all_experiments()

    # Export results
    orchestrator.export_results()

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
