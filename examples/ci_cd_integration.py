"""
Example: CI/CD Integration
Use ChaosEater as a deployment gate in CI/CD pipeline
"""

import sys
sys.path.append('..')

from orchestrator import ExperimentOrchestrator
from chaos_controller import ChaosLayer


def run_chaos_gate():
    """
    Run chaos experiments as CI/CD gate
    Returns exit code 0 if all tests pass, 1 if any fail
    """
    print("\n🚦 CI/CD Chaos Gate - Starting Validation\n")

    orchestrator = ExperimentOrchestrator()

    # Define critical experiments for deployment validation
    critical_experiments = [
        {
            "name": "data_corruption_validation",
            "layer": ChaosLayer.DATA,
            "fault_type": "data_corruption",
            "parameters": {"corruption_rate": 0.05},
            "duration": 30,
            "threshold": 75.0  # Minimum resilience score
        },
        {
            "name": "model_drift_validation",
            "layer": ChaosLayer.MODEL,
            "fault_type": "model_drift",
            "parameters": {"drift_rate": 0.1, "drift_type": "gradual"},
            "duration": 30,
            "threshold": 70.0
        },
        {
            "name": "inference_latency_validation",
            "layer": ChaosLayer.MODEL,
            "fault_type": "slow_inference",
            "parameters": {"latency_ms": 300, "variance": 0.2},
            "duration": 30,
            "threshold": 80.0
        },
        {
            "name": "network_resilience_validation",
            "layer": ChaosLayer.INFRASTRUCTURE,
            "fault_type": "network_latency",
            "parameters": {"latency_ms": 50, "jitter_ms": 10},
            "duration": 30,
            "threshold": 75.0
        }
    ]

    # Create experiments
    for exp_config in critical_experiments:
        orchestrator.controller.create_experiment(
            name=exp_config["name"],
            layer=exp_config["layer"],
            fault_type=exp_config["fault_type"],
            parameters=exp_config["parameters"],
            duration=exp_config["duration"]
        )

    orchestrator.experiments = orchestrator.controller.experiments

    # Run validation
    results = orchestrator.run_all_experiments()

    # Evaluate results against thresholds
    passed = True
    for i, result in enumerate(results):
        exp_config = critical_experiments[i]
        summary = result.get("summary", {})
        score = summary.get("resilience_score", 0)
        threshold = exp_config["threshold"]

        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        print(f"{status} | {exp_config['name']}: {score:.1f}/{threshold}")

        if score < threshold:
            passed = False

    # Summary
    print("\n" + "="*60)
    if passed:
        print("🎉 DEPLOYMENT APPROVED - All chaos tests passed")
        print("="*60)
        return 0
    else:
        print("🚨 DEPLOYMENT BLOCKED - Resilience tests failed")
        print("="*60)
        return 1


def run_scheduled_chaos_testing():
    """
    Run scheduled chaos testing in production
    """
    print("\n⏰ Scheduled Chaos Testing - Production Environment\n")

    orchestrator = ExperimentOrchestrator()

    # Load from config
    try:
        orchestrator.load_experiments_from_config("../config.example.yaml")
    except FileNotFoundError:
        print("⚠️  Config file not found, creating sample experiments...")

        # Create sample experiments
        orchestrator.controller.create_experiment(
            name="production_data_validation",
            layer=ChaosLayer.DATA,
            fault_type="missing_data",
            parameters={"missing_rate": 0.1},
            duration=60
        )

        orchestrator.experiments = orchestrator.controller.experiments

    # Run experiments every 6 hours
    print("Starting scheduled chaos testing (every 6 hours)...")
    print("Press Ctrl+C to stop\n")

    try:
        orchestrator.schedule_experiments(interval_seconds=21600)  # 6 hours
    except KeyboardInterrupt:
        print("\n\nScheduled testing stopped by user")


def generate_chaos_report():
    """
    Generate detailed chaos engineering report
    """
    print("\n📊 Generating Chaos Engineering Report\n")

    orchestrator = ExperimentOrchestrator()

    # Run comprehensive test suite
    test_suite = [
        ("data_corruption", ChaosLayer.DATA, "data_corruption", {"corruption_rate": 0.1}),
        ("missing_data", ChaosLayer.DATA, "missing_data", {"missing_rate": 0.15}),
        ("distribution_drift", ChaosLayer.DATA, "distribution_drift", {"magnitude": 0.3}),
        ("model_drift", ChaosLayer.MODEL, "model_drift", {"drift_rate": 0.1, "drift_type": "gradual"}),
        ("slow_inference", ChaosLayer.MODEL, "slow_inference", {"latency_ms": 500, "variance": 0.2}),
        ("network_latency", ChaosLayer.INFRASTRUCTURE, "network_latency", {"latency_ms": 100, "jitter_ms": 20}),
        ("memory_pressure", ChaosLayer.INFRASTRUCTURE, "memory_pressure", {"memory_mb": 512})
    ]

    for name, layer, fault_type, params in test_suite:
        orchestrator.controller.create_experiment(
            name=name,
            layer=layer,
            fault_type=fault_type,
            parameters=params,
            duration=30
        )

    orchestrator.experiments = orchestrator.controller.experiments
    results = orchestrator.run_all_experiments()

    # Export detailed report
    orchestrator.export_results("chaos_report.json")

    # Print summary
    print("\n" + "="*60)
    print("📈 Chaos Engineering Summary Report")
    print("="*60)

    total_experiments = len(results)
    passed = sum(1 for r in results if r.get("summary", {}).get("resilience_score", 0) >= 70)
    avg_score = sum(r.get("summary", {}).get("resilience_score", 0) for r in results) / total_experiments

    print(f"\nTotal Experiments: {total_experiments}")
    print(f"Passed (≥70): {passed}")
    print(f"Failed (<70): {total_experiments - passed}")
    print(f"Average Resilience Score: {avg_score:.1f}/100")

    print("\nDetailed report saved to: chaos_report.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChaosEater CI/CD Integration")
    parser.add_argument("mode", choices=["gate", "schedule", "report"],
                        help="Mode: 'gate' for CI/CD gate, 'schedule' for scheduled testing, 'report' for detailed report")

    args = parser.parse_args()

    if args.mode == "gate":
        exit_code = run_chaos_gate()
        sys.exit(exit_code)
    elif args.mode == "schedule":
        run_scheduled_chaos_testing()
    elif args.mode == "report":
        generate_chaos_report()
