"""
Experiment Comparison and A/B Testing
Compare chaos experiments and run A/B tests for resilience strategies
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
from chaos_controller import ChaosExperiment, ExperimentMetrics, ChaosController


class ComparisonMetric(Enum):
    RESILIENCE_SCORE = "resilience_score"
    RECOVERY_TIME = "recovery_time"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    ERROR_COUNT = "error_count"
    PIPELINE_DOWNTIME = "pipeline_downtime"


@dataclass
class ABTestVariant:
    """A/B test variant configuration"""
    name: str
    experiment: ChaosExperiment
    traffic_percentage: float = 50.0  # 0-100
    sample_size: int = 10
    results: List[ExperimentMetrics] = field(default_factory=list)

    def get_average_metric(self, metric: ComparisonMetric) -> float:
        """Calculate average value for a specific metric"""
        if not self.results:
            return 0.0

        values = []
        for result in self.results:
            if metric == ComparisonMetric.RECOVERY_TIME:
                if result.recovery_time:
                    values.append(result.recovery_time)
            elif metric == ComparisonMetric.ACCURACY_DEGRADATION:
                if result.accuracy_degradation:
                    values.append(result.accuracy_degradation)
            elif metric == ComparisonMetric.ERROR_COUNT:
                values.append(result.error_count)
            elif metric == ComparisonMetric.PIPELINE_DOWNTIME:
                if result.pipeline_downtime:
                    values.append(result.pipeline_downtime)

        return statistics.mean(values) if values else 0.0


class ExperimentComparator:
    """
    Compare multiple chaos experiments side-by-side
    Analyze performance, resilience, and identify best configurations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ExperimentComparator - %(levelname)s - %(message)s'
        )

    def compare_experiments(
        self,
        experiments: List[ChaosExperiment],
        metrics: List[ExperimentMetrics]
    ) -> Dict[str, Any]:
        """Compare multiple experiments and return detailed analysis"""
        if len(experiments) != len(metrics):
            raise ValueError("Experiments and metrics lists must have same length")

        comparison_results = []

        for exp, met in zip(experiments, metrics):
            comparison_results.append({
                "name": exp.name,
                "layer": exp.layer.value,
                "fault_type": exp.fault_type,
                "duration": exp.duration,
                "status": exp.status.value,
                "recovery_time": met.recovery_time or 0,
                "accuracy_degradation": met.accuracy_degradation or 0,
                "error_count": met.error_count,
                "pipeline_downtime": met.pipeline_downtime or 0
            })

        # Find best and worst performers
        best_recovery = min(comparison_results, key=lambda x: x["recovery_time"])
        worst_recovery = max(comparison_results, key=lambda x: x["recovery_time"])

        best_accuracy = min(comparison_results, key=lambda x: x["accuracy_degradation"])
        worst_accuracy = max(comparison_results, key=lambda x: x["accuracy_degradation"])

        # Calculate averages
        avg_recovery = statistics.mean([r["recovery_time"] for r in comparison_results])
        avg_accuracy_deg = statistics.mean([r["accuracy_degradation"] for r in comparison_results])
        avg_errors = statistics.mean([r["error_count"] for r in comparison_results])

        return {
            "total_experiments": len(experiments),
            "comparison_results": comparison_results,
            "best_performers": {
                "recovery_time": {
                    "name": best_recovery["name"],
                    "value": best_recovery["recovery_time"]
                },
                "accuracy": {
                    "name": best_accuracy["name"],
                    "value": best_accuracy["accuracy_degradation"]
                }
            },
            "worst_performers": {
                "recovery_time": {
                    "name": worst_recovery["name"],
                    "value": worst_recovery["recovery_time"]
                },
                "accuracy": {
                    "name": worst_accuracy["name"],
                    "value": worst_accuracy["accuracy_degradation"]
                }
            },
            "averages": {
                "recovery_time": avg_recovery,
                "accuracy_degradation": avg_accuracy_deg,
                "error_count": avg_errors
            }
        }

    def rank_experiments(
        self,
        experiments: List[ChaosExperiment],
        metrics: List[ExperimentMetrics],
        ranking_criteria: ComparisonMetric = ComparisonMetric.RECOVERY_TIME
    ) -> List[Dict]:
        """Rank experiments based on specific criteria"""
        ranked = []

        for exp, met in zip(experiments, metrics):
            value = 0.0
            if ranking_criteria == ComparisonMetric.RECOVERY_TIME:
                value = met.recovery_time or 0
            elif ranking_criteria == ComparisonMetric.ACCURACY_DEGRADATION:
                value = met.accuracy_degradation or 0
            elif ranking_criteria == ComparisonMetric.ERROR_COUNT:
                value = met.error_count
            elif ranking_criteria == ComparisonMetric.PIPELINE_DOWNTIME:
                value = met.pipeline_downtime or 0

            ranked.append({
                "experiment": exp.name,
                "metric": ranking_criteria.value,
                "value": value
            })

        # Sort ascending (lower is better)
        ranked.sort(key=lambda x: x["value"])

        # Add rank
        for i, item in enumerate(ranked, 1):
            item["rank"] = i

        return ranked

    def generate_comparison_report(
        self,
        comparison: Dict[str, Any],
        output_format: str = "text"
    ) -> str:
        """Generate formatted comparison report"""
        if output_format == "text":
            report = []
            report.append("=" * 80)
            report.append("CHAOS EXPERIMENT COMPARISON REPORT")
            report.append("=" * 80)
            report.append(f"\nTotal Experiments: {comparison['total_experiments']}")

            report.append("\n" + "-" * 80)
            report.append("BEST PERFORMERS")
            report.append("-" * 80)
            report.append(f"Best Recovery Time: {comparison['best_performers']['recovery_time']['name']}")
            report.append(f"  Value: {comparison['best_performers']['recovery_time']['value']:.2f}s")
            report.append(f"Best Accuracy: {comparison['best_performers']['accuracy']['name']}")
            report.append(f"  Degradation: {comparison['best_performers']['accuracy']['value']:.2f}%")

            report.append("\n" + "-" * 80)
            report.append("AVERAGES")
            report.append("-" * 80)
            report.append(f"Avg Recovery Time: {comparison['averages']['recovery_time']:.2f}s")
            report.append(f"Avg Accuracy Degradation: {comparison['averages']['accuracy_degradation']:.2f}%")
            report.append(f"Avg Error Count: {comparison['averages']['error_count']:.2f}")

            report.append("\n" + "-" * 80)
            report.append("DETAILED RESULTS")
            report.append("-" * 80)
            for result in comparison['comparison_results']:
                report.append(f"\n{result['name']} ({result['layer']}/{result['fault_type']})")
                report.append(f"  Recovery Time: {result['recovery_time']:.2f}s")
                report.append(f"  Accuracy Degradation: {result['accuracy_degradation']:.2f}%")
                report.append(f"  Error Count: {result['error_count']}")
                report.append(f"  Status: {result['status']}")

            report.append("\n" + "=" * 80)

            return "\n".join(report)

        elif output_format == "json":
            import json
            return json.dumps(comparison, indent=2)

        else:
            return str(comparison)


class ABTestRunner:
    """
    Run A/B tests for chaos experiments
    Compare different configurations or strategies
    """

    def __init__(self, controller: ChaosController):
        self.controller = controller
        self.logger = logging.getLogger(__name__)

    def create_ab_test(
        self,
        variant_a: ChaosExperiment,
        variant_b: ChaosExperiment,
        sample_size: int = 10,
        traffic_split: Tuple[float, float] = (50.0, 50.0)
    ) -> Tuple[ABTestVariant, ABTestVariant]:
        """Create A/B test with two variants"""
        return (
            ABTestVariant(
                name="Variant A",
                experiment=variant_a,
                traffic_percentage=traffic_split[0],
                sample_size=sample_size
            ),
            ABTestVariant(
                name="Variant B",
                experiment=variant_b,
                traffic_percentage=traffic_split[1],
                sample_size=sample_size
            )
        )

    def run_ab_test(
        self,
        variant_a: ABTestVariant,
        variant_b: ABTestVariant,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """Run A/B test and collect results"""
        self.logger.info(f"🧪 Starting A/B test: {variant_a.name} vs {variant_b.name}")

        # Run variant A samples
        self.logger.info(f"Running {variant_a.sample_size} samples for {variant_a.name}...")
        for i in range(variant_a.sample_size):
            try:
                metrics = self.controller.run_experiment(variant_a.experiment)
                variant_a.results.append(metrics)
                self.logger.info(f"  Sample {i+1}/{variant_a.sample_size} completed")
            except Exception as e:
                self.logger.error(f"  Sample {i+1} failed: {str(e)}")

        # Run variant B samples
        self.logger.info(f"Running {variant_b.sample_size} samples for {variant_b.name}...")
        for i in range(variant_b.sample_size):
            try:
                metrics = self.controller.run_experiment(variant_b.experiment)
                variant_b.results.append(metrics)
                self.logger.info(f"  Sample {i+1}/{variant_b.sample_size} completed")
            except Exception as e:
                self.logger.error(f"  Sample {i+1} failed: {str(e)}")

        # Analyze results
        return self._analyze_ab_test(variant_a, variant_b, significance_level)

    def _analyze_ab_test(
        self,
        variant_a: ABTestVariant,
        variant_b: ABTestVariant,
        significance_level: float
    ) -> Dict[str, Any]:
        """Analyze A/B test results and determine winner"""
        metrics_comparison = {}

        for metric in ComparisonMetric:
            avg_a = variant_a.get_average_metric(metric)
            avg_b = variant_b.get_average_metric(metric)

            # Simple winner determination (could use statistical tests)
            winner = variant_a.name if avg_a < avg_b else variant_b.name
            improvement = abs(avg_a - avg_b) / max(avg_a, avg_b) * 100 if max(avg_a, avg_b) > 0 else 0

            metrics_comparison[metric.value] = {
                "variant_a_avg": avg_a,
                "variant_b_avg": avg_b,
                "winner": winner,
                "improvement_percent": improvement
            }

        # Overall winner (based on recovery time)
        recovery_a = variant_a.get_average_metric(ComparisonMetric.RECOVERY_TIME)
        recovery_b = variant_b.get_average_metric(ComparisonMetric.RECOVERY_TIME)
        overall_winner = variant_a.name if recovery_a < recovery_b else variant_b.name

        return {
            "variant_a": {
                "name": variant_a.name,
                "experiment": variant_a.experiment.name,
                "samples": len(variant_a.results)
            },
            "variant_b": {
                "name": variant_b.name,
                "experiment": variant_b.experiment.name,
                "samples": len(variant_b.results)
            },
            "metrics_comparison": metrics_comparison,
            "overall_winner": overall_winner,
            "confidence_level": 1 - significance_level
        }

    def print_ab_test_report(self, results: Dict[str, Any]):
        """Print formatted A/B test report"""
        print("\n" + "=" * 80)
        print("A/B TEST RESULTS")
        print("=" * 80)

        print(f"\nVariant A: {results['variant_a']['experiment']} ({results['variant_a']['samples']} samples)")
        print(f"Variant B: {results['variant_b']['experiment']} ({results['variant_b']['samples']} samples)")

        print("\n" + "-" * 80)
        print("METRIC COMPARISON")
        print("-" * 80)

        for metric, data in results['metrics_comparison'].items():
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"  Variant A: {data['variant_a_avg']:.2f}")
            print(f"  Variant B: {data['variant_b_avg']:.2f}")
            print(f"  Winner: {data['winner']} (↑{data['improvement_percent']:.1f}% improvement)")

        print("\n" + "=" * 80)
        print(f"🏆 OVERALL WINNER: {results['overall_winner']}")
        print(f"Confidence Level: {results['confidence_level']*100:.0f}%")
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    from chaos_controller import ChaosLayer

    # Create comparator
    comparator = ExperimentComparator()

    # Create sample experiments for comparison
    exp1 = ChaosExperiment(
        name="data_corruption_v1",
        layer=ChaosLayer.DATA,
        fault_type="data_corruption",
        parameters={"corruption_rate": 0.1},
        duration=60
    )
    exp1.metrics = ExperimentMetrics(
        start_time=datetime.now(),
        end_time=datetime.now(),
        recovery_time=45.0,
        accuracy_degradation=5.0,
        error_count=2
    )

    exp2 = ChaosExperiment(
        name="data_corruption_v2",
        layer=ChaosLayer.DATA,
        fault_type="data_corruption",
        parameters={"corruption_rate": 0.15},
        duration=60
    )
    exp2.metrics = ExperimentMetrics(
        start_time=datetime.now(),
        end_time=datetime.now(),
        recovery_time=52.0,
        accuracy_degradation=8.5,
        error_count=4
    )

    # Compare experiments
    comparison = comparator.compare_experiments(
        [exp1, exp2],
        [exp1.metrics, exp2.metrics]
    )

    # Print report
    report = comparator.generate_comparison_report(comparison, "text")
    print(report)

    # Rank experiments
    ranked = comparator.rank_experiments(
        [exp1, exp2],
        [exp1.metrics, exp2.metrics],
        ComparisonMetric.RECOVERY_TIME
    )
    print("\n📊 Ranking by Recovery Time:")
    for item in ranked:
        print(f"  {item['rank']}. {item['experiment']}: {item['value']:.2f}s")
