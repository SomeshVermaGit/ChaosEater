# ChaosEater v2.0 - New Features Guide

This document provides detailed information about the new features added to ChaosEater.

---

## 1. Distributed Chaos Testing

**File:** `distributed_chaos.py`

### Overview
Run chaos experiments across multiple regions, clusters, or environments simultaneously.

### Key Features
- **Multi-region support**: Test across AWS, GCP, Azure regions
- **Execution modes**: Parallel, sequential, or blast-radius controlled
- **Async architecture**: High-performance async execution with aiohttp
- **Health monitoring**: Real-time health checks across all regions
- **Aggregate reporting**: Combined metrics and analysis

### Usage Example

```python
import asyncio
from distributed_chaos import DistributedChaosController, RegionConfig
from chaos_controller import ChaosExperiment, ChaosLayer

# Setup regions
regions = [
    RegionConfig(name="us-east-1", endpoint="http://chaos-east.example.com", priority=1),
    RegionConfig(name="us-west-2", endpoint="http://chaos-west.example.com", priority=2),
    RegionConfig(name="eu-west-1", endpoint="http://chaos-eu.example.com", priority=3),
]

controller = DistributedChaosController(regions)

# Create experiment
experiment = ChaosExperiment(
    name="network_latency_test",
    layer=ChaosLayer.INFRASTRUCTURE,
    fault_type="network_latency",
    parameters={"latency_ms": 200},
    duration=60
)

# Run across all regions in parallel
results = await controller.run_parallel(experiment, max_concurrent=3)

# Get aggregate report
report = controller.get_aggregate_report()
print(f"Success Rate: {report['success_rate']:.1f}%")
```

### API Methods
- `run_parallel(experiment, max_concurrent)` - Run in parallel with concurrency limit
- `run_sequential(experiment, fail_fast)` - Run sequentially by priority
- `run_blast_radius(experiment, blast_radius_percent)` - Control blast radius
- `health_check_all_regions()` - Check health of all regions
- `get_aggregate_report()` - Get combined results

---

## 2. Experiment History & Analytics

**File:** `experiment_history.py`

### Overview
Persistent SQLite storage for all chaos experiments with powerful querying and analytics.

### Key Features
- **SQLite database**: Lightweight, serverless persistence
- **Historical trends**: Track metrics over time
- **Advanced queries**: Filter by layer, fault type, status, resilience score
- **CSV export**: Export data for external analysis
- **Tag support**: Add custom metadata to experiments

### Usage Example

```python
from experiment_history import ExperimentHistory

history = ExperimentHistory("chaos_experiments.db")

# Save experiment
exp_id = history.save_experiment(
    experiment,
    resilience_score=85.5,
    tags={"env": "staging", "version": "v1.2.3"}
)

# Query experiments
results = history.query_experiments(
    layer="data",
    min_resilience_score=70.0,
    limit=10
)

# Get trends
trends = history.get_experiment_trends("data_corruption_test", days=30)
print(f"Avg Score: {trends['avg_resilience_score']}")
print(f"Success Rate: {trends['success_rate']}%")

# Export to CSV
history.export_to_csv("experiments.csv")
```

### API Methods
- `save_experiment(experiment, resilience_score, tags)` - Save experiment
- `query_experiments(filters)` - Query with multiple filters
- `get_experiment_trends(name, days)` - Analyze trends
- `compare_experiments(experiment_ids)` - Compare multiple experiments
- `get_summary_statistics()` - Overall statistics
- `export_to_csv(filepath)` - Export to CSV

---

## 3. Advanced Rollback Strategies

**File:** `advanced_rollback.py`

### Overview
Implement sophisticated rollback mechanisms to minimize risk during chaos experiments.

### Key Features
- **5 rollback strategies**: Immediate, gradual, canary, blue-green, circuit breaker
- **Health monitoring**: Continuous health checks during rollback
- **Auto-pause**: Automatic pause on error detection
- **Configurable thresholds**: Control failure thresholds and timing

### Strategies

#### Immediate Rollback
Revert all changes at once - fastest but highest risk.

#### Gradual Rollback
Incremental rollback in configurable steps with health checks.

#### Canary Rollback
Test rollback on small percentage first, then full rollback if successful.

#### Blue-Green Rollback
Switch traffic between blue (current) and green (previous) environments.

#### Circuit Breaker Rollback
Automatically triggers rollback when error threshold exceeded.

### Usage Example

```python
from advanced_rollback import RollbackManager, RollbackConfig, RollbackStrategy

manager = RollbackManager()

# Gradual rollback
config = RollbackConfig(
    strategy=RollbackStrategy.GRADUAL,
    gradual_steps=5,
    health_check_interval=5,
    failure_threshold=0.1
)

def my_rollback(percentage=100):
    print(f"Rolling back {percentage}%")

def health_check():
    return True  # Check system health

success = manager.execute_rollback(config, my_rollback, health_check)
```

### Configuration Options
- `strategy`: IMMEDIATE, GRADUAL, CANARY, BLUE_GREEN, CIRCUIT_BREAKER
- `timeout_seconds`: Maximum rollback time
- `health_check_interval`: Seconds between health checks
- `failure_threshold`: Error rate that triggers halt (0.0-1.0)
- `canary_percentage`: Canary traffic percentage
- `gradual_steps`: Number of gradual rollback steps

---

## 4. Grafana Dashboard Generator

**File:** `grafana_dashboard.py`

### Overview
Automatically generate comprehensive Grafana dashboards for chaos experiment visualization.

### Key Features
- **Pre-built panels**: Resilience score, recovery time, error rates, etc.
- **Customizable**: Add custom panels and queries
- **Prometheus integration**: Built-in Prometheus queries
- **Alert rules**: Auto-generate alert rules
- **Push to Grafana**: Direct API integration

### Usage Example

```python
from grafana_dashboard import GrafanaDashboardGenerator

generator = GrafanaDashboardGenerator(
    dashboard_title="ChaosEater - ML Pipeline Resilience"
)

# Export to file
generator.export_to_file("chaos_dashboard.json")

# Or push directly to Grafana
generator.push_to_grafana(
    grafana_url="http://grafana.example.com",
    api_key="your-api-key",
    datasource="Prometheus"
)
```

### Included Panels
1. **Resilience Score Gauge** - Current resilience score
2. **Experiment Status** - Status of running experiments
3. **Recovery Time Graph** - Recovery time over time
4. **Error Count** - Error rates and counts
5. **Accuracy Degradation** - ML model accuracy impact
6. **Duration Heatmap** - Experiment duration distribution
7. **Fault Type Distribution** - Pie chart of fault types
8. **Layer Success Rate** - Success rate by chaos layer

### Alert Rules
```python
from grafana_dashboard import GrafanaAlertManager

alert_mgr = GrafanaAlertManager(grafana_url, api_key)

# Create low resilience score alert
alert = alert_mgr.create_resilience_score_alert(min_score=70.0)

# Create high error rate alert
alert = alert_mgr.create_high_error_rate_alert(max_error_rate=0.1)
```

---

## 5. A/B Testing & Comparison

**File:** `experiment_comparison.py`

### Overview
Compare chaos experiments and run A/B tests to find optimal configurations.

### Key Features
- **Side-by-side comparison**: Compare multiple experiments
- **A/B testing**: Test different configurations
- **Statistical analysis**: Determine winning variant
- **Ranking**: Rank experiments by any metric
- **Detailed reports**: Text and JSON export

### Usage Example

```python
from experiment_comparison import ExperimentComparator, ABTestRunner

# Compare experiments
comparator = ExperimentComparator()
comparison = comparator.compare_experiments(
    [experiment1, experiment2, experiment3],
    [metrics1, metrics2, metrics3]
)

# Generate report
report = comparator.generate_comparison_report(comparison, "text")
print(report)

# Rank experiments
ranked = comparator.rank_experiments(
    experiments,
    metrics,
    ComparisonMetric.RECOVERY_TIME
)

# A/B testing
ab_runner = ABTestRunner(controller)
variant_a, variant_b = ab_runner.create_ab_test(
    experiment_v1,
    experiment_v2,
    sample_size=10
)

results = ab_runner.run_ab_test(variant_a, variant_b)
ab_runner.print_ab_test_report(results)
```

### Comparison Metrics
- `RESILIENCE_SCORE` - Overall resilience score
- `RECOVERY_TIME` - Time to recover from failure
- `ACCURACY_DEGRADATION` - ML model accuracy impact
- `ERROR_COUNT` - Number of errors
- `PIPELINE_DOWNTIME` - Total downtime

---

## 6. AI-Powered Resilience Optimization

**File:** `resilience_optimizer.py`

### Overview
Machine learning-based analysis and recommendations for improving system resilience.

### Key Features
- **Pattern detection**: Identify weak points automatically
- **Anomaly detection**: Find outlier experiments
- **Predictive scoring**: Predict resilience scores before running
- **Parameter optimization**: Suggest optimal parameters
- **Automated recommendations**: Prioritized improvement suggestions

### Usage Example

```python
from resilience_optimizer import ResilienceOptimizer

optimizer = ResilienceOptimizer()

# Analyze historical patterns
patterns = optimizer.analyze_historical_patterns(days=30)

# Generate recommendations
recommendations = optimizer.generate_optimization_recommendations(patterns)

# Generate full report
report = optimizer.generate_optimization_report()
print(report)

# Predict resilience score
prediction = optimizer.predict_resilience_score(
    "network_latency",
    {"latency_ms": 100}
)
print(f"Predicted Score: {prediction['predicted_score']}")

# Get optimal parameters
optimal = optimizer.suggest_optimal_parameters(
    "data_corruption",
    target_resilience_score=70.0
)
print(f"Suggested Parameters: {optimal['suggested_parameters']}")
```

### Recommendation Categories
1. **Architecture** - System design improvements
2. **Configuration** - Parameter tuning
3. **Monitoring** - Observability enhancements

### Features
- **Historical Analysis**: Analyze 30+ days of experiment data
- **Weak Point Detection**: Automatically identify failure-prone areas
- **Trend Analysis**: Track resilience trends over time
- **Smart Predictions**: ML-based resilience score predictions
- **Parameter Tuning**: Data-driven parameter suggestions

---

## Integration Examples

### Complete Workflow

```python
import asyncio
from chaos_controller import ChaosController, ChaosLayer, ChaosExperiment
from distributed_chaos import DistributedChaosController, RegionConfig
from experiment_history import ExperimentHistory
from resilience_optimizer import ResilienceOptimizer
from grafana_dashboard import GrafanaDashboardGenerator

# 1. Setup
controller = ChaosController()
history = ExperimentHistory()
optimizer = ResilienceOptimizer()

# 2. Get optimal parameters from AI
optimal_params = optimizer.suggest_optimal_parameters(
    "network_latency",
    target_resilience_score=75.0
)

# 3. Create experiment with optimal parameters
experiment = ChaosExperiment(
    name="optimized_network_test",
    layer=ChaosLayer.INFRASTRUCTURE,
    fault_type="network_latency",
    parameters=optimal_params['suggested_parameters'],
    duration=60
)

# 4. Run distributed test
regions = [
    RegionConfig("us-east-1", "http://chaos-east.example.com"),
    RegionConfig("us-west-2", "http://chaos-west.example.com")
]
dist_controller = DistributedChaosController(regions)
results = await dist_controller.run_parallel(experiment)

# 5. Save to history
for result in results:
    history.save_experiment(
        experiment,
        resilience_score=result.resilience_score,
        tags={"region": result.region}
    )

# 6. Generate Grafana dashboard
dashboard_gen = GrafanaDashboardGenerator()
dashboard_gen.export_to_file("chaos_dashboard.json")

# 7. Get optimization recommendations
report = optimizer.generate_optimization_report()
print(report)
```

---

## Best Practices

### 1. Start with History Tracking
Always enable experiment history to build a dataset for optimization.

### 2. Use Gradual Rollback in Production
Prefer gradual or canary rollback strategies to minimize blast radius.

### 3. Leverage A/B Testing
Test new configurations with A/B tests before rolling out broadly.

### 4. Monitor with Grafana
Set up dashboards early to track trends and spot issues.

### 5. Follow AI Recommendations
Use the optimizer's recommendations to systematically improve resilience.

### 6. Run Distributed Tests for Critical Systems
Test multi-region resilience for production-critical systems.

---

## Configuration Tips

### Database Location
```python
# Use custom database location
history = ExperimentHistory("/path/to/chaos_db.sqlite")
```

### Concurrent Distributed Tests
```python
# Control parallelism to avoid overwhelming systems
results = await controller.run_parallel(experiment, max_concurrent=2)
```

### Custom Grafana Panels
```python
# Add custom panels
generator.add_custom_panel({
    "title": "Custom Metric",
    "type": "graph",
    "targets": [{"expr": "your_custom_metric"}]
})
```

---

## Troubleshooting

### Distributed Testing Timeouts
Increase timeout in experiment configuration:
```python
experiment.duration = 120  # seconds
```

### Database Locked Errors
Use separate database files for concurrent processes or enable WAL mode.

### Missing Prometheus Metrics
Ensure Prometheus exporters are configured in observers.

---

## Next Steps

1. **Kubernetes Integration**: Deploy distributed chaos agents on K8s
2. **MLflow Integration**: Track model versions in experiments
3. **Real-time Streaming**: Stream metrics to Kafka/Kinesis
4. **Advanced ML Models**: Use XGBoost/Random Forest for predictions
5. **Custom Fault Injectors**: Build domain-specific fault injectors

---

For more information, see the main [README.md](README.md) or check individual module docstrings.
