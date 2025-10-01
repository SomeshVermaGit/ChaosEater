# ChaosEater 🦖

**Chaos Engineering Platform for ML Pipelines**

ChaosEater is a comprehensive chaos engineering framework designed specifically for machine learning pipelines. It automatically injects failures into data, model, and infrastructure layers to validate resilience and reliability before production deployment.

---

## 🎯 Features

### **3-Layer Chaos Injection**
- **Data Layer**: Corruption, drift, missing data, schema changes, outliers
- **Model Layer**: Drift, version issues, slow inference, memory leaks, prediction corruption
- **Infrastructure Layer**: Network latency, packet loss, node failures, resource constraints

### **Observability & Metrics**
- Real-time metrics collection
- Prometheus integration
- Resilience scoring (0-100)
- Recovery time tracking
- Accuracy degradation monitoring

### **Experiment Orchestration**
- YAML-based configuration
- Batch experiment execution
- Scheduled chaos testing
- CI/CD pipeline integration
- Automatic rollback strategies

---

## 🚀 Quick Start

### **Installation**

```bash
pip install -r requirements.txt
```

### **Basic Usage**

```python
from chaos_controller import ChaosController, ChaosLayer
from fault_injectors.data_injector import DataFaultInjector
from observers.metrics_observer import MetricsObserver

# Initialize controller
controller = ChaosController()

# Register injectors
controller.register_injector(ChaosLayer.DATA, DataFaultInjector())

# Register observers
controller.register_observer(MetricsObserver())

# Create experiment
experiment = controller.create_experiment(
    name="data_corruption_test",
    layer=ChaosLayer.DATA,
    fault_type="data_corruption",
    parameters={"corruption_rate": 0.1},
    duration=60
)

# Run experiment
metrics = controller.run_experiment(experiment)

# Get report
report = controller.get_experiment_report(experiment)
print(report)
```

### **Using Orchestrator**

```bash
python orchestrator.py
```

### **Load from Config**

```python
from orchestrator import ExperimentOrchestrator

orchestrator = ExperimentOrchestrator()
orchestrator.load_experiments_from_config("config.example.yaml")
orchestrator.run_all_experiments()
```

---

## 🧪 Supported Fault Types

### **Data Layer**
| Fault Type | Description | Parameters |
|------------|-------------|------------|
| `data_corruption` | Inject random noise/corruption | `corruption_rate`, `columns` |
| `missing_data` | Add missing values | `missing_rate`, `columns` |
| `schema_drift` | Simulate schema changes | `drift_type`, `column_name` |
| `distribution_drift` | Shift data distribution | `magnitude`, `columns` |
| `outlier_injection` | Add outliers | `outlier_rate`, `magnitude` |
| `duplicate_records` | Create duplicates | `duplication_rate` |

### **Model Layer**
| Fault Type | Description | Parameters |
|------------|-------------|------------|
| `model_drift` | Degrade model performance | `drift_rate`, `drift_type` |
| `slow_inference` | Add latency to predictions | `latency_ms`, `variance` |
| `prediction_corruption` | Corrupt model outputs | `corruption_rate`, `type` |
| `model_unavailable` | Simulate service downtime | `failure_rate` |
| `memory_leak` | Leak memory during inference | `leak_rate_mb` |
| `wrong_version` | Deploy incorrect model version | `fallback_model` |

### **Infrastructure Layer**
| Fault Type | Description | Parameters |
|------------|-------------|------------|
| `network_latency` | Add network delay | `latency_ms`, `jitter_ms` |
| `packet_loss` | Drop network packets | `loss_rate` |
| `node_failure` | Simulate node crash/hang | `failure_type` |
| `gpu_unavailable` | Make GPU unavailable | - |
| `memory_pressure` | Consume system memory | `memory_mb` |
| `cpu_throttle` | Throttle CPU resources | `throttle_percent` |
| `storage_bottleneck` | Limit disk I/O | `iops_limit` |
| `disk_full` | Simulate disk full | - |

---

## 📊 Metrics & Observability

### **Resilience Score Calculation**
```
score = 100
- (pipeline_downtime / 10) * 40  # max -40 points
- accuracy_degradation * 30       # max -30 points
- error_count * 5 * 30           # max -30 points
```

### **Collected Metrics**
- Recovery time (seconds)
- Accuracy degradation (%)
- Pipeline downtime (seconds)
- Error count
- Custom metrics via observers

### **Prometheus Integration**

```yaml
prometheus:
  pushgateway_url: "http://localhost:9091"
  job_name: "chaos_eater"
```

Metrics pushed:
- `chaos_experiment_duration_seconds`
- `chaos_experiment_status`
- `chaos_recovery_time_seconds`
- `chaos_accuracy_degradation_percent`
- `chaos_error_count`

---

## 🔧 Configuration

### **YAML Config Example**

```yaml
experiments:
  - name: "data_corruption_test"
    layer: "data"
    fault_type: "data_corruption"
    duration: 60
    parameters:
      corruption_rate: 0.1
    rollback_strategy: "auto"

  - name: "slow_inference_test"
    layer: "model"
    fault_type: "slow_inference"
    duration: 60
    parameters:
      latency_ms: 500
      variance: 0.2
```

---

## 🎯 CI/CD Integration

Use ChaosEater as a deployment gate:

```python
orchestrator = ExperimentOrchestrator()
orchestrator.load_experiments_from_config("chaos_tests.yaml")

# Require 70+ resilience score to pass
if orchestrator.create_ci_cd_gate(min_resilience_score=70.0):
    print("✅ Deployment approved")
    exit(0)
else:
    print("❌ Deployment blocked - resilience tests failed")
    exit(1)
```

---

## 🏗️ Architecture

```
ChaosEater/
├── chaos_controller.py          # Core orchestration logic
├── orchestrator.py              # Experiment scheduling & CI/CD
├── fault_injectors/
│   ├── base_injector.py        # Abstract injector interface
│   ├── data_injector.py        # Data layer faults
│   ├── model_injector.py       # Model layer faults
│   └── infra_injector.py       # Infrastructure faults
├── observers/
│   └── metrics_observer.py     # Metrics collection & export
├── config.example.yaml         # Example configuration
└── requirements.txt
```

---

## 📈 Roadmap

- [x] Data layer fault injection
- [x] Model layer fault injection
- [x] Infrastructure layer fault injection
- [x] Metrics collection & observability
- [x] YAML-based configuration
- [ ] Kubernetes integration (Chaos Mesh)
- [ ] MLflow integration
- [ ] Kubeflow integration
- [ ] Airflow DAG integration
- [ ] Grafana dashboard templates
- [ ] Advanced rollback strategies
- [ ] Multi-region chaos testing
- [ ] Automated resilience optimization

---

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

---

## 📄 License

MIT License

---

**Built for ML Engineers who believe:** *If it hasn't failed in chaos testing, it will fail in production.* 🔥
