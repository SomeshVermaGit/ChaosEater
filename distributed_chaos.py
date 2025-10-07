"""
Distributed Chaos Testing
Enables multi-region and multi-cluster chaos experiments
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from chaos_controller import ChaosExperiment, ExperimentStatus


class RegionStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class RegionConfig:
    """Configuration for a region/cluster"""
    name: str
    endpoint: str
    priority: int = 1  # Higher priority regions tested first
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedExperimentResult:
    """Results from distributed chaos experiment"""
    region: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    resilience_score: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "region": self.region,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "resilience_score": self.resilience_score,
            "error_message": self.error_message,
            "metrics": self.metrics
        }


class DistributedChaosController:
    """
    Orchestrates chaos experiments across multiple regions/clusters
    Supports multi-region testing, blast radius control, and coordinated failures
    """

    def __init__(self, regions: Optional[List[RegionConfig]] = None):
        self.regions = regions or []
        self.results: List[DistributedExperimentResult] = []
        self.logger = logging.getLogger(__name__)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - DistributedChaos - %(levelname)s - %(message)s'
        )

    def add_region(self, region: RegionConfig):
        """Add a region/cluster to the distributed test"""
        self.regions.append(region)
        self.logger.info(f"Added region: {region.name} ({region.endpoint})")

    async def run_experiment_async(
        self,
        region: RegionConfig,
        experiment: ChaosExperiment,
        session: aiohttp.ClientSession
    ) -> DistributedExperimentResult:
        """Run chaos experiment on a single region asynchronously"""
        result = DistributedExperimentResult(
            region=region.name,
            experiment_name=experiment.name,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Call remote chaos controller API
            payload = {
                "name": experiment.name,
                "layer": experiment.layer.value,
                "fault_type": experiment.fault_type,
                "parameters": experiment.parameters,
                "duration": experiment.duration
            }

            self.logger.info(f"[{region.name}] Starting experiment: {experiment.name}")

            async with session.post(
                f"{region.endpoint}/api/v1/experiments/run",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=experiment.duration + 30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result.status = ExperimentStatus.COMPLETED
                    result.resilience_score = data.get("resilience_score", 0.0)
                    result.metrics = data.get("metrics", {})
                    self.logger.info(
                        f"[{region.name}] ✅ Completed: {experiment.name} "
                        f"(score: {result.resilience_score:.1f})"
                    )
                else:
                    result.status = ExperimentStatus.FAILED
                    result.error_message = f"HTTP {response.status}"
                    self.logger.error(f"[{region.name}] ❌ Failed: {experiment.name}")

        except asyncio.TimeoutError:
            result.status = ExperimentStatus.FAILED
            result.error_message = "Timeout"
            self.logger.error(f"[{region.name}] ⏱ Timeout: {experiment.name}")
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"[{region.name}] ❌ Error: {str(e)}")

        result.end_time = datetime.now()
        return result

    async def run_parallel(
        self,
        experiment: ChaosExperiment,
        max_concurrent: int = 3
    ) -> List[DistributedExperimentResult]:
        """Run experiment across all regions in parallel with concurrency limit"""
        self.logger.info(
            f"Running experiment '{experiment.name}' across {len(self.regions)} regions "
            f"(max concurrent: {max_concurrent})"
        )

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_limit(region: RegionConfig):
                async with semaphore:
                    return await self.run_experiment_async(region, experiment, session)

            tasks = [run_with_limit(region) for region in self.regions]
            results = await asyncio.gather(*tasks)

        self.results.extend(results)
        return results

    async def run_sequential(
        self,
        experiment: ChaosExperiment,
        fail_fast: bool = False
    ) -> List[DistributedExperimentResult]:
        """Run experiment across regions sequentially (by priority)"""
        # Sort by priority (highest first)
        sorted_regions = sorted(self.regions, key=lambda r: r.priority, reverse=True)

        results = []
        async with aiohttp.ClientSession() as session:
            for region in sorted_regions:
                result = await self.run_experiment_async(region, experiment, session)
                results.append(result)

                # Fail fast if enabled and experiment failed
                if fail_fast and result.status == ExperimentStatus.FAILED:
                    self.logger.warning(
                        f"Fail-fast triggered at region {region.name}. "
                        "Stopping sequential execution."
                    )
                    break

        self.results.extend(results)
        return results

    async def run_blast_radius(
        self,
        experiment: ChaosExperiment,
        blast_radius_percent: float = 0.3
    ) -> List[DistributedExperimentResult]:
        """
        Run experiment on a subset of regions to control blast radius

        Args:
            blast_radius_percent: Percentage of regions to affect (0.0-1.0)
        """
        num_regions = max(1, int(len(self.regions) * blast_radius_percent))
        selected_regions = self.regions[:num_regions]

        self.logger.info(
            f"Blast radius: {blast_radius_percent*100:.0f}% "
            f"({num_regions}/{len(self.regions)} regions)"
        )

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.run_experiment_async(region, experiment, session)
                for region in selected_regions
            ]
            results = await asyncio.gather(*tasks)

        self.results.extend(results)
        return results

    def get_aggregate_report(self) -> Dict:
        """Generate aggregate report across all regions"""
        if not self.results:
            return {"status": "No results available"}

        total = len(self.results)
        completed = sum(1 for r in self.results if r.status == ExperimentStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == ExperimentStatus.FAILED)

        avg_score = sum(r.resilience_score for r in self.results) / total if total > 0 else 0.0

        min_score_result = min(self.results, key=lambda r: r.resilience_score)
        max_score_result = max(self.results, key=lambda r: r.resilience_score)

        return {
            "total_regions": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total * 100) if total > 0 else 0.0,
            "average_resilience_score": avg_score,
            "min_resilience_score": {
                "region": min_score_result.region,
                "score": min_score_result.resilience_score
            },
            "max_resilience_score": {
                "region": max_score_result.region,
                "score": max_score_result.resilience_score
            },
            "results": [r.to_dict() for r in self.results]
        }

    async def health_check_all_regions(self) -> Dict[str, RegionStatus]:
        """Check health of all regions"""
        async def check_region(session: aiohttp.ClientSession, region: RegionConfig):
            try:
                async with session.get(
                    f"{region.endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return region.name, RegionStatus.HEALTHY
                    else:
                        return region.name, RegionStatus.DEGRADED
            except:
                return region.name, RegionStatus.FAILED

        async with aiohttp.ClientSession() as session:
            tasks = [check_region(session, region) for region in self.regions]
            results = await asyncio.gather(*tasks)

        return dict(results)

    def export_distributed_results(self, filepath: str):
        """Export distributed experiment results to JSON"""
        import json

        report = self.get_aggregate_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Distributed results exported to {filepath}")


# Example usage
async def example_usage():
    """Example: distributed chaos testing across 3 regions"""
    from chaos_controller import ChaosLayer

    # Setup regions
    regions = [
        RegionConfig(name="us-east-1", endpoint="http://chaos-east.example.com", priority=1),
        RegionConfig(name="us-west-2", endpoint="http://chaos-west.example.com", priority=2),
        RegionConfig(name="eu-west-1", endpoint="http://chaos-eu.example.com", priority=3),
    ]

    controller = DistributedChaosController(regions)

    # Create experiment
    from chaos_controller import ChaosExperiment
    experiment = ChaosExperiment(
        name="network_latency_multi_region",
        layer=ChaosLayer.INFRASTRUCTURE,
        fault_type="network_latency",
        parameters={"latency_ms": 200, "jitter_ms": 50},
        duration=60
    )

    # Run parallel across all regions
    print("🌍 Running parallel distributed chaos test...")
    results = await controller.run_parallel(experiment, max_concurrent=2)

    # Get report
    report = controller.get_aggregate_report()
    print(f"\n📊 Distributed Chaos Report:")
    print(f"  Success Rate: {report['success_rate']:.1f}%")
    print(f"  Avg Resilience Score: {report['average_resilience_score']:.1f}")
    print(f"  Min Score: {report['min_resilience_score']['score']:.1f} ({report['min_resilience_score']['region']})")
    print(f"  Max Score: {report['max_resilience_score']['score']:.1f} ({report['max_resilience_score']['region']})")


if __name__ == "__main__":
    asyncio.run(example_usage())
