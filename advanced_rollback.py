"""
Advanced Rollback Strategies
Implements gradual, canary, and blue-green rollback mechanisms
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import threading


class RollbackStrategy(Enum):
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    CIRCUIT_BREAKER = "circuit_breaker"


class RollbackStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class RollbackConfig:
    """Configuration for rollback strategies"""
    strategy: RollbackStrategy
    timeout_seconds: int = 300
    health_check_interval: int = 5
    failure_threshold: float = 0.1  # 10% error rate triggers rollback
    canary_percentage: int = 10  # For canary deployments
    gradual_steps: int = 5  # For gradual rollback
    auto_pause_on_error: bool = True


class BaseRollbackStrategy(ABC):
    """Abstract base class for rollback strategies"""

    def __init__(self, config: RollbackConfig):
        self.config = config
        self.status = RollbackStatus.NOT_STARTED
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rollback_percentage = 0
        self.error_count = 0
        self.request_count = 0

    @abstractmethod
    def execute(self, rollback_func: Callable, health_check_func: Optional[Callable] = None) -> bool:
        """Execute the rollback strategy"""
        pass

    def check_health(self, health_check_func: Optional[Callable]) -> bool:
        """Check if system is healthy"""
        if health_check_func:
            try:
                return health_check_func()
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                return False
        return True

    def calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def should_halt(self) -> bool:
        """Check if rollback should halt due to errors"""
        error_rate = self.calculate_error_rate()
        return error_rate > self.config.failure_threshold


class ImmediateRollback(BaseRollbackStrategy):
    """Immediate rollback - revert all changes at once"""

    def execute(self, rollback_func: Callable, health_check_func: Optional[Callable] = None) -> bool:
        self.logger.info("Executing immediate rollback...")
        self.status = RollbackStatus.IN_PROGRESS

        try:
            # Execute rollback
            rollback_func()
            self.rollback_percentage = 100

            # Verify health
            if self.check_health(health_check_func):
                self.status = RollbackStatus.COMPLETED
                self.logger.info("✅ Immediate rollback completed successfully")
                return True
            else:
                self.status = RollbackStatus.FAILED
                self.logger.error("❌ Rollback completed but health check failed")
                return False

        except Exception as e:
            self.status = RollbackStatus.FAILED
            self.logger.error(f"❌ Immediate rollback failed: {str(e)}")
            return False


class GradualRollback(BaseRollbackStrategy):
    """Gradual rollback - revert changes in incremental steps"""

    def execute(self, rollback_func: Callable, health_check_func: Optional[Callable] = None) -> bool:
        self.logger.info(f"Executing gradual rollback in {self.config.gradual_steps} steps...")
        self.status = RollbackStatus.IN_PROGRESS

        step_percentage = 100 / self.config.gradual_steps

        try:
            for step in range(1, self.config.gradual_steps + 1):
                target_percentage = step * step_percentage

                self.logger.info(f"Rollback step {step}/{self.config.gradual_steps} ({target_percentage:.0f}%)")

                # Execute partial rollback
                rollback_func(percentage=target_percentage)
                self.rollback_percentage = target_percentage

                # Health check after each step
                time.sleep(self.config.health_check_interval)

                if not self.check_health(health_check_func):
                    if self.config.auto_pause_on_error:
                        self.status = RollbackStatus.PAUSED
                        self.logger.warning(f"⏸ Rollback paused at {target_percentage:.0f}% due to health check failure")
                        return False
                    else:
                        self.logger.warning(f"⚠️ Health check failed at {target_percentage:.0f}%, continuing...")

                # Check error threshold
                if self.should_halt():
                    self.status = RollbackStatus.FAILED
                    self.logger.error(f"❌ Rollback halted at {target_percentage:.0f}% - error threshold exceeded")
                    return False

            self.status = RollbackStatus.COMPLETED
            self.logger.info("✅ Gradual rollback completed successfully")
            return True

        except Exception as e:
            self.status = RollbackStatus.FAILED
            self.logger.error(f"❌ Gradual rollback failed: {str(e)}")
            return False


class CanaryRollback(BaseRollbackStrategy):
    """Canary rollback - test rollback on subset first"""

    def execute(self, rollback_func: Callable, health_check_func: Optional[Callable] = None) -> bool:
        self.logger.info(f"Executing canary rollback ({self.config.canary_percentage}% canary)...")
        self.status = RollbackStatus.IN_PROGRESS

        try:
            # Phase 1: Canary rollback
            self.logger.info(f"Phase 1: Rolling back {self.config.canary_percentage}% (canary)")
            rollback_func(percentage=self.config.canary_percentage)
            self.rollback_percentage = self.config.canary_percentage

            # Monitor canary
            self.logger.info(f"Monitoring canary for {self.config.health_check_interval * 3}s...")
            for _ in range(3):
                time.sleep(self.config.health_check_interval)

                if not self.check_health(health_check_func):
                    self.status = RollbackStatus.FAILED
                    self.logger.error("❌ Canary rollback failed health check")
                    return False

                if self.should_halt():
                    self.status = RollbackStatus.FAILED
                    self.logger.error("❌ Canary rollback exceeded error threshold")
                    return False

            # Phase 2: Full rollback
            self.logger.info("✅ Canary successful, proceeding to full rollback")
            rollback_func(percentage=100)
            self.rollback_percentage = 100

            # Final health check
            time.sleep(self.config.health_check_interval)
            if self.check_health(health_check_func):
                self.status = RollbackStatus.COMPLETED
                self.logger.info("✅ Canary rollback completed successfully")
                return True
            else:
                self.status = RollbackStatus.FAILED
                self.logger.error("❌ Full rollback failed health check")
                return False

        except Exception as e:
            self.status = RollbackStatus.FAILED
            self.logger.error(f"❌ Canary rollback failed: {str(e)}")
            return False


class BlueGreenRollback(BaseRollbackStrategy):
    """Blue-Green rollback - switch traffic between environments"""

    def execute(self, rollback_func: Callable, health_check_func: Optional[Callable] = None) -> bool:
        self.logger.info("Executing blue-green rollback...")
        self.status = RollbackStatus.IN_PROGRESS

        try:
            # Phase 1: Prepare green environment (old/stable version)
            self.logger.info("Phase 1: Preparing green environment (stable version)")
            rollback_func(action="prepare_green")

            # Health check green environment
            time.sleep(self.config.health_check_interval)
            if not self.check_health(health_check_func):
                self.status = RollbackStatus.FAILED
                self.logger.error("❌ Green environment health check failed")
                return False

            # Phase 2: Switch traffic from blue to green
            self.logger.info("Phase 2: Switching traffic to green environment")
            rollback_func(action="switch_traffic", percentage=100)
            self.rollback_percentage = 100

            # Verify traffic switch
            time.sleep(self.config.health_check_interval)
            if self.check_health(health_check_func):
                # Phase 3: Shutdown blue environment
                self.logger.info("Phase 3: Shutting down blue environment")
                rollback_func(action="shutdown_blue")

                self.status = RollbackStatus.COMPLETED
                self.logger.info("✅ Blue-green rollback completed successfully")
                return True
            else:
                # Rollback the switch
                self.logger.error("❌ Traffic switch failed, reverting to blue")
                rollback_func(action="switch_traffic", percentage=0)
                self.status = RollbackStatus.FAILED
                return False

        except Exception as e:
            self.status = RollbackStatus.FAILED
            self.logger.error(f"❌ Blue-green rollback failed: {str(e)}")
            return False


class CircuitBreakerRollback(BaseRollbackStrategy):
    """Circuit breaker rollback - automatic rollback based on error thresholds"""

    def __init__(self, config: RollbackConfig):
        super().__init__(config)
        self.circuit_open = False
        self.consecutive_failures = 0
        self.failure_threshold_count = 5

    def execute(self, rollback_func: Callable, health_check_func: Optional[Callable] = None) -> bool:
        self.logger.info("Executing circuit breaker rollback...")
        self.status = RollbackStatus.IN_PROGRESS

        try:
            # Monitor and auto-trigger rollback
            monitoring_duration = self.config.timeout_seconds
            start_time = time.time()

            while time.time() - start_time < monitoring_duration:
                time.sleep(self.config.health_check_interval)

                # Check health
                is_healthy = self.check_health(health_check_func)

                if not is_healthy:
                    self.consecutive_failures += 1
                    self.logger.warning(
                        f"⚠️ Health check failed ({self.consecutive_failures}/{self.failure_threshold_count})"
                    )

                    # Open circuit and rollback
                    if self.consecutive_failures >= self.failure_threshold_count:
                        self.circuit_open = True
                        self.logger.error("❌ Circuit breaker opened - triggering rollback")

                        rollback_func()
                        self.rollback_percentage = 100

                        self.status = RollbackStatus.COMPLETED
                        return True
                else:
                    self.consecutive_failures = 0

            # Monitoring completed without triggering
            self.status = RollbackStatus.COMPLETED
            self.logger.info("✅ Circuit breaker monitoring completed - no rollback needed")
            return True

        except Exception as e:
            self.status = RollbackStatus.FAILED
            self.logger.error(f"❌ Circuit breaker rollback failed: {str(e)}")
            return False


class RollbackManager:
    """Manages and orchestrates different rollback strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def execute_rollback(
        self,
        config: RollbackConfig,
        rollback_func: Callable,
        health_check_func: Optional[Callable] = None
    ) -> bool:
        """Execute rollback with specified strategy"""

        strategy_map = {
            RollbackStrategy.IMMEDIATE: ImmediateRollback,
            RollbackStrategy.GRADUAL: GradualRollback,
            RollbackStrategy.CANARY: CanaryRollback,
            RollbackStrategy.BLUE_GREEN: BlueGreenRollback,
            RollbackStrategy.CIRCUIT_BREAKER: CircuitBreakerRollback
        }

        strategy_class = strategy_map.get(config.strategy)
        if not strategy_class:
            self.logger.error(f"Unknown rollback strategy: {config.strategy}")
            return False

        strategy = strategy_class(config)
        return strategy.execute(rollback_func, health_check_func)


# Example usage
def example_usage():
    """Example: using different rollback strategies"""

    def mock_rollback(percentage: int = 100, action: str = "rollback"):
        print(f"  🔄 Executing {action} at {percentage}%")
        time.sleep(0.5)

    def mock_health_check():
        print("  🏥 Running health check...")
        return True  # Simulate healthy state

    manager = RollbackManager()

    # Example 1: Gradual rollback
    print("\n" + "="*60)
    print("Example 1: Gradual Rollback")
    print("="*60)
    config = RollbackConfig(
        strategy=RollbackStrategy.GRADUAL,
        gradual_steps=3,
        health_check_interval=2
    )
    success = manager.execute_rollback(config, mock_rollback, mock_health_check)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Example 2: Canary rollback
    print("\n" + "="*60)
    print("Example 2: Canary Rollback")
    print("="*60)
    config = RollbackConfig(
        strategy=RollbackStrategy.CANARY,
        canary_percentage=10,
        health_check_interval=1
    )
    success = manager.execute_rollback(config, mock_rollback, mock_health_check)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Example 3: Blue-Green rollback
    print("\n" + "="*60)
    print("Example 3: Blue-Green Rollback")
    print("="*60)
    config = RollbackConfig(
        strategy=RollbackStrategy.BLUE_GREEN,
        health_check_interval=1
    )
    success = manager.execute_rollback(config, mock_rollback, mock_health_check)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")


if __name__ == "__main__":
    example_usage()
