"""
Grafana Dashboard Generator
Automatically generates Grafana dashboards for chaos experiment visualization
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class GrafanaDashboardGenerator:
    """
    Generates Grafana dashboard JSON configurations for chaos experiments
    Supports custom panels, queries, and layouts
    """

    def __init__(self, dashboard_title: str = "ChaosEater - Chaos Engineering Dashboard"):
        self.dashboard_title = dashboard_title
        self.panels: List[Dict] = []
        self.logger = logging.getLogger(__name__)

    def create_dashboard(
        self,
        datasource: str = "Prometheus",
        refresh_interval: str = "5s",
        time_range: str = "1h"
    ) -> Dict:
        """Create base dashboard configuration"""
        return {
            "dashboard": {
                "title": self.dashboard_title,
                "tags": ["chaos-engineering", "ml-pipeline", "resilience"],
                "timezone": "browser",
                "editable": True,
                "graphTooltip": 1,
                "time": {
                    "from": f"now-{time_range}",
                    "to": "now"
                },
                "refresh": refresh_interval,
                "schemaVersion": 38,
                "panels": self.panels,
                "templating": {
                    "list": [
                        {
                            "name": "datasource",
                            "type": "datasource",
                            "query": "prometheus"
                        },
                        {
                            "name": "experiment",
                            "type": "query",
                            "query": "label_values(chaos_experiment_status, experiment_name)",
                            "datasource": datasource,
                            "refresh": 1,
                            "multi": True,
                            "includeAll": True
                        }
                    ]
                },
                "annotations": {
                    "list": [
                        {
                            "datasource": datasource,
                            "name": "Experiments",
                            "enable": True,
                            "iconColor": "red",
                            "query": "chaos_experiment_status == 1"
                        }
                    ]
                }
            },
            "overwrite": True
        }

    def add_resilience_score_panel(self, datasource: str = "Prometheus") -> int:
        """Add resilience score gauge panel"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Resilience Score",
            "type": "gauge",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            },
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 50, "color": "yellow"},
                            {"value": 70, "color": "green"}
                        ]
                    },
                    "unit": "short"
                }
            },
            "targets": [
                {
                    "expr": "avg(chaos_resilience_score{experiment=~\"$experiment\"})",
                    "refId": "A"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_experiment_status_panel(self, datasource: str = "Prometheus") -> int:
        """Add experiment status panel"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Experiment Status",
            "type": "stat",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
            "options": {
                "graphMode": "area",
                "colorMode": "background",
                "justifyMode": "auto",
                "textMode": "auto"
            },
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {"value": 0, "text": "Pending", "color": "blue"},
                        {"value": 1, "text": "Running", "color": "yellow"},
                        {"value": 2, "text": "Completed", "color": "green"},
                        {"value": 3, "text": "Failed", "color": "red"}
                    ]
                }
            },
            "targets": [
                {
                    "expr": "chaos_experiment_status{experiment=~\"$experiment\"}",
                    "refId": "A",
                    "legendFormat": "{{experiment_name}}"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_recovery_time_panel(self, datasource: str = "Prometheus") -> int:
        """Add recovery time graph panel"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Recovery Time (seconds)",
            "type": "timeseries",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 10
                    },
                    "unit": "s"
                }
            },
            "options": {
                "tooltip": {"mode": "multi"},
                "legend": {"displayMode": "list", "placement": "bottom"}
            },
            "targets": [
                {
                    "expr": "chaos_recovery_time_seconds{experiment=~\"$experiment\"}",
                    "refId": "A",
                    "legendFormat": "{{experiment_name}}"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_error_count_panel(self, datasource: str = "Prometheus") -> int:
        """Add error count panel"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Error Count",
            "type": "timeseries",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "bars",
                        "fillOpacity": 80
                    },
                    "color": {"mode": "palette-classic"}
                }
            },
            "targets": [
                {
                    "expr": "rate(chaos_error_count{experiment=~\"$experiment\"}[5m])",
                    "refId": "A",
                    "legendFormat": "{{experiment_name}}"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_accuracy_degradation_panel(self, datasource: str = "Prometheus") -> int:
        """Add accuracy degradation panel"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Accuracy Degradation (%)",
            "type": "timeseries",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 20
                    },
                    "unit": "percent",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 10, "color": "yellow"},
                            {"value": 30, "color": "red"}
                        ]
                    }
                }
            },
            "targets": [
                {
                    "expr": "chaos_accuracy_degradation_percent{experiment=~\"$experiment\"}",
                    "refId": "A",
                    "legendFormat": "{{experiment_name}}"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_experiment_duration_panel(self, datasource: str = "Prometheus") -> int:
        """Add experiment duration heatmap"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Experiment Duration Distribution",
            "type": "heatmap",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
            "options": {
                "calculate": False,
                "cellGap": 2,
                "color": {
                    "mode": "scheme",
                    "scheme": "Spectral"
                }
            },
            "targets": [
                {
                    "expr": "rate(chaos_experiment_duration_seconds_bucket{experiment=~\"$experiment\"}[5m])",
                    "refId": "A",
                    "format": "heatmap"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_fault_type_distribution_panel(self, datasource: str = "Prometheus") -> int:
        """Add fault type distribution pie chart"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Fault Type Distribution",
            "type": "piechart",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 8, "x": 0, "y": 24},
            "options": {
                "legend": {"displayMode": "table", "placement": "right"},
                "pieType": "donut",
                "displayLabels": ["percent"]
            },
            "targets": [
                {
                    "expr": "count by (fault_type) (chaos_experiment_status{experiment=~\"$experiment\"})",
                    "refId": "A",
                    "legendFormat": "{{fault_type}}"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_layer_success_rate_panel(self, datasource: str = "Prometheus") -> int:
        """Add chaos layer success rate panel"""
        panel_id = len(self.panels)
        panel = {
            "id": panel_id,
            "title": "Success Rate by Layer",
            "type": "bargauge",
            "datasource": datasource,
            "gridPos": {"h": 8, "w": 16, "x": 8, "y": 24},
            "options": {
                "orientation": "horizontal",
                "displayMode": "gradient",
                "showUnfilled": True
            },
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "unit": "percent",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 70, "color": "yellow"},
                            {"value": 90, "color": "green"}
                        ]
                    }
                }
            },
            "targets": [
                {
                    "expr": "100 * (sum by (layer) (chaos_experiment_status{status=\"completed\",experiment=~\"$experiment\"}) / sum by (layer) (chaos_experiment_status{experiment=~\"$experiment\"}))",
                    "refId": "A",
                    "legendFormat": "{{layer}}"
                }
            ]
        }
        self.panels.append(panel)
        return panel_id

    def add_custom_panel(self, panel_config: Dict) -> int:
        """Add a custom panel with user-defined configuration"""
        panel_id = len(self.panels)
        panel_config["id"] = panel_id
        self.panels.append(panel_config)
        return panel_id

    def generate_complete_dashboard(
        self,
        datasource: str = "Prometheus",
        include_all_panels: bool = True
    ) -> Dict:
        """Generate complete dashboard with all standard panels"""
        if include_all_panels:
            self.panels = []  # Reset panels
            self.add_resilience_score_panel(datasource)
            self.add_experiment_status_panel(datasource)
            self.add_recovery_time_panel(datasource)
            self.add_error_count_panel(datasource)
            self.add_accuracy_degradation_panel(datasource)
            self.add_experiment_duration_panel(datasource)
            self.add_fault_type_distribution_panel(datasource)
            self.add_layer_success_rate_panel(datasource)

        return self.create_dashboard(datasource)

    def export_to_file(self, filepath: str, datasource: str = "Prometheus"):
        """Export dashboard to JSON file"""
        dashboard = self.generate_complete_dashboard(datasource)

        with open(filepath, 'w') as f:
            json.dump(dashboard, f, indent=2)

        self.logger.info(f"Dashboard exported to {filepath}")

    def push_to_grafana(
        self,
        grafana_url: str,
        api_key: str,
        datasource: str = "Prometheus"
    ) -> bool:
        """Push dashboard to Grafana instance"""
        import requests

        dashboard = self.generate_complete_dashboard(datasource)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                f"{grafana_url}/api/dashboards/db",
                headers=headers,
                json=dashboard,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"✅ Dashboard pushed successfully: {result.get('url')}")
                return True
            else:
                self.logger.error(f"❌ Failed to push dashboard: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error pushing dashboard to Grafana: {str(e)}")
            return False


class GrafanaAlertManager:
    """Manages Grafana alerts for chaos experiments"""

    def __init__(self, grafana_url: str, api_key: str):
        self.grafana_url = grafana_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def create_alert_rule(
        self,
        rule_name: str,
        metric_query: str,
        threshold: float,
        condition: str = "gt",
        notification_channel: str = "default"
    ) -> Dict:
        """Create Grafana alert rule"""
        alert_rule = {
            "name": rule_name,
            "conditions": [
                {
                    "evaluator": {
                        "params": [threshold],
                        "type": condition
                    },
                    "query": {
                        "model": {
                            "expr": metric_query,
                            "refId": "A"
                        }
                    },
                    "type": "query"
                }
            ],
            "frequency": "1m",
            "for": "5m",
            "message": f"Chaos experiment alert: {rule_name}",
            "notifications": [
                {"uid": notification_channel}
            ]
        }
        return alert_rule

    def create_resilience_score_alert(self, min_score: float = 70.0) -> Dict:
        """Create alert for low resilience scores"""
        return self.create_alert_rule(
            rule_name="Low Resilience Score",
            metric_query="avg(chaos_resilience_score)",
            threshold=min_score,
            condition="lt"
        )

    def create_high_error_rate_alert(self, max_error_rate: float = 0.1) -> Dict:
        """Create alert for high error rates"""
        return self.create_alert_rule(
            rule_name="High Error Rate in Chaos Experiment",
            metric_query="rate(chaos_error_count[5m])",
            threshold=max_error_rate,
            condition="gt"
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate dashboard
    generator = GrafanaDashboardGenerator(
        dashboard_title="ChaosEater - ML Pipeline Resilience"
    )

    # Export to file
    generator.export_to_file("grafana_chaos_dashboard.json")

    print("✅ Grafana dashboard generated successfully!")
    print("📁 Dashboard saved to: grafana_chaos_dashboard.json")
    print("\n📋 To import:")
    print("1. Open Grafana UI")
    print("2. Go to Dashboards > Import")
    print("3. Upload grafana_chaos_dashboard.json")
    print("4. Select your Prometheus datasource")
    print("5. Click Import")
