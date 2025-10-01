"""
Metrics Observer
Collects and tracks metrics during chaos experiments
Integrates with monitoring stacks (Prometheus, Grafana)
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class MetricsObserver:
    """
    Observes and collects metrics during chaos experiments
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics_buffer: List[Dict] = []
        self.current_experiment = None
        self.start_time = None

    def on_experiment_start(self, experiment):
        """Called when experiment starts"""
        self.current_experiment = experiment
        self.start_time = time.time()
        self.metrics_buffer.clear()

        metric = {
            "timestamp": datetime.now().isoformat(),
            "event": "experiment_start",
            "experiment_name": experiment.name,
            "layer": experiment.layer.value,
            "fault_type": experiment.fault_type
        }
        self.metrics_buffer.append(metric)

    def on_experiment_end(self, experiment):
        """Called when experiment ends"""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0

        metric = {
            "timestamp": datetime.now().isoformat(),
            "event": "experiment_end",
            "experiment_name": experiment.name,
            "duration": duration,
            "status": experiment.status.value
        }
        self.metrics_buffer.append(metric)

        # Export metrics
        if self.config.get("auto_export"):
            self.export_metrics()

    def collect_metrics(self, experiment):
        """Collect real-time metrics during experiment"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "event": "metrics_collection",
            "experiment_name": experiment.name,
            "custom_metrics": {}
        }

        # Collect system metrics if available
        try:
            import psutil
            metric["custom_metrics"]["cpu_percent"] = psutil.cpu_percent()
            metric["custom_metrics"]["memory_percent"] = psutil.virtual_memory().percent
            metric["custom_metrics"]["disk_io"] = psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        except ImportError:
            pass

        self.metrics_buffer.append(metric)

    def export_metrics(self, filepath: Optional[str] = None):
        """Export collected metrics to file"""
        if not filepath:
            filepath = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filepath, 'w') as f:
            json.dump(self.metrics_buffer, f, indent=2)

        return filepath

    def get_metrics(self) -> List[Dict]:
        """Get all collected metrics"""
        return self.metrics_buffer


class PrometheusObserver:
    """
    Pushes metrics to Prometheus Pushgateway
    """

    def __init__(self, pushgateway_url: str, job_name: str = "chaos_eater"):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.metrics = {}

    def on_experiment_start(self, experiment):
        """Record experiment start"""
        self.metrics[f"{experiment.name}_start_time"] = time.time()

    def on_experiment_end(self, experiment):
        """Record experiment end and push metrics"""
        end_time = time.time()
        start_time = self.metrics.get(f"{experiment.name}_start_time", end_time)
        duration = end_time - start_time

        # Prepare metrics
        metrics_data = {
            "chaos_experiment_duration_seconds": duration,
            "chaos_experiment_status": 1 if experiment.status.value == "completed" else 0,
        }

        if experiment.metrics:
            if experiment.metrics.recovery_time:
                metrics_data["chaos_recovery_time_seconds"] = experiment.metrics.recovery_time
            if experiment.metrics.accuracy_degradation:
                metrics_data["chaos_accuracy_degradation_percent"] = experiment.metrics.accuracy_degradation
            if experiment.metrics.error_count:
                metrics_data["chaos_error_count"] = experiment.metrics.error_count

        # Push to Pushgateway
        self._push_metrics(metrics_data, experiment)

    def _push_metrics(self, metrics_data: Dict, experiment):
        """Push metrics to Prometheus Pushgateway"""
        try:
            import requests

            # Format metrics in Prometheus format
            metrics_text = ""
            for metric_name, value in metrics_data.items():
                labels = f'experiment="{experiment.name}",layer="{experiment.layer.value}",fault="{experiment.fault_type}"'
                metrics_text += f"{metric_name}{{{labels}}} {value}\n"

            # Push to gateway
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}"
            response = requests.post(url, data=metrics_text)
            response.raise_for_status()

        except ImportError:
            print("Warning: requests library not available for Prometheus push")
        except Exception as e:
            print(f"Error pushing metrics to Prometheus: {e}")

    def collect_metrics(self, experiment):
        """Collect real-time metrics"""
        # Could push intermediate metrics here
        pass


class LogObserver:
    """
    Simple logging observer for debugging
    """

    def __init__(self, log_file: str = "chaos_experiments.log"):
        self.log_file = log_file

    def on_experiment_start(self, experiment):
        """Log experiment start"""
        self._write_log(f"[START] {experiment.name} - {experiment.layer.value}/{experiment.fault_type}")

    def on_experiment_end(self, experiment):
        """Log experiment end"""
        status = experiment.status.value
        self._write_log(f"[END] {experiment.name} - Status: {status}")

    def collect_metrics(self, experiment):
        """Log metrics collection"""
        self._write_log(f"[METRICS] {experiment.name} - Collecting...")

    def _write_log(self, message: str):
        """Write to log file"""
        timestamp = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {message}\n")
