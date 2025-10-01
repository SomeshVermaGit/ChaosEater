"""
Base Fault Injector Interface
All fault injectors must implement this interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging


class BaseFaultInjector(ABC):
    """Abstract base class for all fault injectors"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.active_faults = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def inject(self, fault_type: str, parameters: Dict[str, Any], duration: int):
        """
        Inject a specific fault

        Args:
            fault_type: Type of fault to inject
            parameters: Fault-specific parameters
            duration: How long to maintain the fault (seconds)
        """
        pass

    @abstractmethod
    def rollback(self):
        """Rollback all injected faults"""
        pass

    @abstractmethod
    def validate(self, fault_type: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate fault parameters before injection

        Returns:
            True if parameters are valid
        """
        pass

    def get_supported_faults(self) -> list:
        """Return list of supported fault types"""
        return []
