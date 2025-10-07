"""
Automated Resilience Optimization
Uses ML-based analysis to suggest optimal chaos configurations and improvements
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import json


@dataclass
class OptimizationRecommendation:
    """Recommendation for improving resilience"""
    category: str  # "configuration", "architecture", "monitoring"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    expected_improvement: float  # Expected % improvement in resilience score
    implementation_effort: str  # "low", "medium", "high"
    rationale: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class ResilienceOptimizer:
    """
    Analyzes historical chaos experiment data to identify patterns
    and suggest optimizations using ML-based techniques
    """

    def __init__(self, history_db_path: str = "chaos_experiments.db"):
        from experiment_history import ExperimentHistory
        self.history = ExperimentHistory(history_db_path)
        self.logger = logging.getLogger(__name__)
        self.recommendations: List[OptimizationRecommendation] = []

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ResilienceOptimizer - %(levelname)s - %(message)s'
        )

    def analyze_historical_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze historical experiment data for patterns"""
        summary = self.history.get_summary_statistics()

        patterns = {
            "total_experiments": summary.get("total_experiments", 0),
            "avg_resilience_score": summary.get("avg_resilience_score", 0),
            "success_rate": summary.get("success_rate", 0),
            "fault_type_distribution": summary.get("fault_type_distribution", {}),
            "weak_points": [],
            "strong_points": []
        }

        # Identify weak points (low resilience scores)
        weak_experiments = self.history.query_experiments(
            min_resilience_score=0,
            limit=100
        )

        # Calculate weak points by fault type
        fault_scores = defaultdict(list)
        for exp in weak_experiments:
            if exp.get("resilience_score"):
                fault_scores[exp["fault_type"]].append(exp["resilience_score"])

        # Identify weakest fault types
        avg_scores = {
            fault: np.mean(scores)
            for fault, scores in fault_scores.items()
            if scores
        }

        sorted_faults = sorted(avg_scores.items(), key=lambda x: x[1])

        if sorted_faults:
            weakest = sorted_faults[:3]  # Top 3 weakest
            patterns["weak_points"] = [
                {
                    "fault_type": fault,
                    "avg_score": round(score, 2),
                    "severity": "high" if score < 50 else "medium" if score < 70 else "low"
                }
                for fault, score in weakest
            ]

            strongest = sorted_faults[-3:]  # Top 3 strongest
            patterns["strong_points"] = [
                {
                    "fault_type": fault,
                    "avg_score": round(score, 2)
                }
                for fault, score in strongest
            ]

        return patterns

    def detect_anomalies(
        self,
        experiment_name: str,
        days: int = 30,
        std_threshold: float = 2.0
    ) -> List[Dict]:
        """Detect anomalous experiment results"""
        trends = self.history.get_experiment_trends(experiment_name, days)

        if trends.get("total_runs", 0) < 5:
            return []

        # Get all runs for this experiment
        experiments = self.history.query_experiments(
            name_pattern=experiment_name,
            limit=100
        )

        if not experiments:
            return []

        # Calculate statistics
        scores = [exp.get("resilience_score", 0) for exp in experiments if exp.get("resilience_score")]

        if len(scores) < 5:
            return []

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Find anomalies (> 2 std deviations from mean)
        anomalies = []
        for exp in experiments:
            score = exp.get("resilience_score", 0)
            if abs(score - mean_score) > std_threshold * std_score:
                anomalies.append({
                    "experiment_id": exp.get("id"),
                    "name": exp.get("name"),
                    "score": score,
                    "deviation": abs(score - mean_score) / std_score if std_score > 0 else 0,
                    "type": "outlier_low" if score < mean_score else "outlier_high"
                })

        return anomalies

    def generate_optimization_recommendations(
        self,
        patterns: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate ML-based optimization recommendations"""
        recommendations = []

        # Recommendation 1: Address weak fault types
        if patterns.get("weak_points"):
            for weak_point in patterns["weak_points"]:
                if weak_point["severity"] == "high":
                    rec = OptimizationRecommendation(
                        category="architecture",
                        priority="high",
                        title=f"Critical: Improve resilience to {weak_point['fault_type']}",
                        description=f"System shows low resilience (avg score: {weak_point['avg_score']}) "
                                    f"when experiencing {weak_point['fault_type']} faults. "
                                    f"Consider implementing retry mechanisms, circuit breakers, or fallback strategies.",
                        expected_improvement=30.0,
                        implementation_effort="medium",
                        rationale=f"Historical data shows consistent low performance ({weak_point['avg_score']}/100) "
                                  f"for this fault type across {patterns['total_experiments']} experiments.",
                        metrics=weak_point
                    )
                    recommendations.append(rec)

        # Recommendation 2: Improve overall success rate
        if patterns["success_rate"] < 80:
            rec = OptimizationRecommendation(
                category="configuration",
                priority="high",
                title="Increase experiment success rate",
                description=f"Current success rate is {patterns['success_rate']:.1f}%. "
                            f"Review failure logs and adjust fault injection parameters to reduce false positives.",
                expected_improvement=20.0,
                implementation_effort="low",
                rationale="Low success rate may indicate overly aggressive fault injection or system instability.",
                metrics={"current_success_rate": patterns["success_rate"]}
            )
            recommendations.append(rec)

        # Recommendation 3: Balance fault type coverage
        fault_dist = patterns.get("fault_type_distribution", {})
        if fault_dist:
            max_count = max(fault_dist.values())
            min_count = min(fault_dist.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 3:
                rec = OptimizationRecommendation(
                    category="monitoring",
                    priority="medium",
                    title="Balance chaos experiment coverage",
                    description=f"Fault type distribution is imbalanced (ratio: {imbalance_ratio:.1f}:1). "
                                f"Increase testing frequency for underrepresented fault types.",
                    expected_improvement=15.0,
                    implementation_effort="low",
                    rationale="Comprehensive coverage of all fault types provides better resilience insights.",
                    metrics={"fault_distribution": fault_dist, "imbalance_ratio": imbalance_ratio}
                )
                recommendations.append(rec)

        # Recommendation 4: Enhance monitoring for low-score areas
        if patterns["avg_resilience_score"] < 70:
            rec = OptimizationRecommendation(
                category="monitoring",
                priority="high",
                title="Deploy enhanced monitoring and alerting",
                description=f"Average resilience score ({patterns['avg_resilience_score']:.1f}) is below target (70). "
                            f"Implement comprehensive monitoring, distributed tracing, and real-time alerting.",
                expected_improvement=25.0,
                implementation_effort="medium",
                rationale="Better observability enables faster detection and recovery from failures.",
                metrics={"current_avg_score": patterns["avg_resilience_score"]}
            )
            recommendations.append(rec)

        # Recommendation 5: Implement gradual rollback
        rec = OptimizationRecommendation(
            category="configuration",
            priority="medium",
            title="Adopt gradual or canary rollback strategies",
            description="Implement progressive rollback strategies to minimize blast radius during failures. "
                        "Use canary deployments to test changes on subset of traffic first.",
            expected_improvement=20.0,
            implementation_effort="medium",
            rationale="Gradual rollback reduces risk and allows for early detection of issues.",
            metrics={}
        )
        recommendations.append(rec)

        self.recommendations = recommendations
        return recommendations

    def predict_resilience_score(
        self,
        fault_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict expected resilience score using historical data (simple ML)"""
        # Get similar historical experiments
        similar_experiments = self.history.query_experiments(
            fault_type=fault_type,
            limit=50
        )

        if not similar_experiments:
            return {
                "predicted_score": 0.0,
                "confidence": 0.0,
                "recommendation": "No historical data available for this fault type"
            }

        scores = [exp.get("resilience_score", 0) for exp in similar_experiments if exp.get("resilience_score")]

        if not scores:
            return {
                "predicted_score": 0.0,
                "confidence": 0.0,
                "recommendation": "No resilience scores available"
            }

        # Simple prediction: weighted average based on parameter similarity
        # In production, use actual ML model (Random Forest, XGBoost, etc.)
        predicted_score = np.mean(scores)
        std_dev = np.std(scores)
        confidence = 1.0 / (1.0 + std_dev / 100.0)  # Lower std = higher confidence

        recommendation = ""
        if predicted_score < 50:
            recommendation = "High risk: Consider reducing fault injection parameters"
        elif predicted_score < 70:
            recommendation = "Medium risk: Monitor closely during experiment"
        else:
            recommendation = "Low risk: Expected to pass resilience threshold"

        return {
            "predicted_score": round(predicted_score, 2),
            "confidence": round(confidence, 2),
            "recommendation": recommendation,
            "historical_samples": len(scores),
            "score_range": {
                "min": round(min(scores), 2),
                "max": round(max(scores), 2),
                "std": round(std_dev, 2)
            }
        }

    def suggest_optimal_parameters(
        self,
        fault_type: str,
        target_resilience_score: float = 70.0
    ) -> Dict[str, Any]:
        """Suggest optimal fault injection parameters"""
        # Get historical data for this fault type
        experiments = self.history.query_experiments(
            fault_type=fault_type,
            min_resilience_score=target_resilience_score,
            limit=20
        )

        if not experiments:
            return {
                "fault_type": fault_type,
                "status": "No successful experiments found for this fault type",
                "suggested_parameters": {}
            }

        # Analyze successful configurations
        # In production: use ML clustering or parameter optimization
        parameter_sets = []
        for exp in experiments:
            params = json.loads(exp.get("parameters", "{}"))
            score = exp.get("resilience_score", 0)
            parameter_sets.append((params, score))

        # Find best performing parameters
        best_params, best_score = max(parameter_sets, key=lambda x: x[1])

        return {
            "fault_type": fault_type,
            "target_score": target_resilience_score,
            "suggested_parameters": best_params,
            "expected_score": round(best_score, 2),
            "based_on_samples": len(parameter_sets),
            "recommendation": f"Use these parameters to achieve ~{best_score:.1f} resilience score"
        }

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        patterns = self.analyze_historical_patterns(days=30)
        recommendations = self.generate_optimization_recommendations(patterns)

        report = []
        report.append("=" * 80)
        report.append("RESILIENCE OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report.append("\n" + "-" * 80)
        report.append("CURRENT STATE")
        report.append("-" * 80)
        report.append(f"Total Experiments: {patterns['total_experiments']}")
        report.append(f"Avg Resilience Score: {patterns['avg_resilience_score']:.1f}/100")
        report.append(f"Success Rate: {patterns['success_rate']:.1f}%")

        if patterns["weak_points"]:
            report.append("\n⚠️  WEAK POINTS:")
            for wp in patterns["weak_points"]:
                report.append(f"  • {wp['fault_type']}: {wp['avg_score']:.1f}/100 ({wp['severity']} severity)")

        if patterns["strong_points"]:
            report.append("\n✅ STRONG POINTS:")
            for sp in patterns["strong_points"]:
                report.append(f"  • {sp['fault_type']}: {sp['avg_score']:.1f}/100")

        report.append("\n" + "-" * 80)
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 80)

        for i, rec in enumerate(recommendations, 1):
            priority_emoji = "🔴" if rec.priority == "high" else "🟡" if rec.priority == "medium" else "🟢"
            report.append(f"\n{i}. {priority_emoji} [{rec.priority.upper()}] {rec.title}")
            report.append(f"   Category: {rec.category}")
            report.append(f"   Expected Improvement: +{rec.expected_improvement:.0f}%")
            report.append(f"   Implementation Effort: {rec.implementation_effort}")
            report.append(f"   Description: {rec.description}")
            report.append(f"   Rationale: {rec.rationale}")

        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def export_recommendations(self, filepath: str):
        """Export recommendations to JSON file"""
        data = {
            "generated_at": datetime.now().isoformat(),
            "recommendations": [
                {
                    "category": rec.category,
                    "priority": rec.priority,
                    "title": rec.title,
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "implementation_effort": rec.implementation_effort,
                    "rationale": rec.rationale,
                    "metrics": rec.metrics
                }
                for rec in self.recommendations
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Recommendations exported to {filepath}")


# Example usage
if __name__ == "__main__":
    optimizer = ResilienceOptimizer()

    print("🔍 Analyzing historical patterns...")
    patterns = optimizer.analyze_historical_patterns(days=30)

    print(f"\n📊 Analysis Results:")
    print(f"  Total Experiments: {patterns['total_experiments']}")
    print(f"  Avg Resilience Score: {patterns['avg_resilience_score']:.1f}")
    print(f"  Success Rate: {patterns['success_rate']:.1f}%")

    print("\n🎯 Generating optimization recommendations...")
    recommendations = optimizer.generate_optimization_recommendations(patterns)

    print(f"\n✅ Generated {len(recommendations)} recommendations")

    # Generate full report
    print("\n" + "=" * 80)
    report = optimizer.generate_optimization_report()
    print(report)

    # Export recommendations
    optimizer.export_recommendations("resilience_recommendations.json")
    print("\n💾 Recommendations saved to resilience_recommendations.json")

    # Predict resilience score
    print("\n🔮 Predicting resilience score for 'network_latency'...")
    prediction = optimizer.predict_resilience_score(
        "network_latency",
        {"latency_ms": 100, "jitter_ms": 20}
    )
    print(f"  Predicted Score: {prediction['predicted_score']:.1f}/100")
    print(f"  Confidence: {prediction['confidence']:.2f}")
    print(f"  Recommendation: {prediction['recommendation']}")
