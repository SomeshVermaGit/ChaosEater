"""
Data Layer Fault Injector
Simulates data-related failures: corruption, drift, missing data, schema changes
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from .base_injector import BaseFaultInjector
import time
import random


class DataFaultInjector(BaseFaultInjector):
    """Injects faults into data layer"""

    SUPPORTED_FAULTS = [
        "data_corruption",
        "missing_data",
        "schema_drift",
        "distribution_drift",
        "outlier_injection",
        "latency_injection",
        "duplicate_records"
    ]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.original_data = None
        self.data_transform_fn: Optional[Callable] = None

    def inject(self, fault_type: str, parameters: Dict[str, Any], duration: int):
        """Inject data layer fault"""
        if not self.validate(fault_type, parameters):
            raise ValueError(f"Invalid parameters for fault type: {fault_type}")

        self.logger.info(f"Injecting {fault_type} with params: {parameters}")

        if fault_type == "data_corruption":
            self._inject_corruption(parameters)
        elif fault_type == "missing_data":
            self._inject_missing_data(parameters)
        elif fault_type == "schema_drift":
            self._inject_schema_drift(parameters)
        elif fault_type == "distribution_drift":
            self._inject_distribution_drift(parameters)
        elif fault_type == "outlier_injection":
            self._inject_outliers(parameters)
        elif fault_type == "latency_injection":
            self._inject_latency(parameters, duration)
        elif fault_type == "duplicate_records":
            self._inject_duplicates(parameters)
        else:
            raise ValueError(f"Unsupported fault type: {fault_type}")

        self.active_faults.append(fault_type)

    def _inject_corruption(self, params: Dict[str, Any]):
        """Corrupt data with random noise or bit flips"""
        corruption_rate = params.get("corruption_rate", 0.1)
        target_columns = params.get("columns", None)

        def corrupt_data(data):
            if isinstance(data, pd.DataFrame):
                corrupted = data.copy()
                cols = target_columns if target_columns else data.select_dtypes(include=[np.number]).columns

                for col in cols:
                    if col in corrupted.columns:
                        mask = np.random.random(len(corrupted)) < corruption_rate
                        if pd.api.types.is_numeric_dtype(corrupted[col]):
                            # Add random noise
                            noise = np.random.normal(0, corrupted[col].std(), mask.sum())
                            corrupted.loc[mask, col] += noise
                        else:
                            # Corrupt string data
                            corrupted.loc[mask, col] = None

                return corrupted
            return data

        self.data_transform_fn = corrupt_data
        self.logger.info(f"Data corruption injected: {corruption_rate * 100}% rate")

    def _inject_missing_data(self, params: Dict[str, Any]):
        """Inject missing values into dataset"""
        missing_rate = params.get("missing_rate", 0.2)
        target_columns = params.get("columns", None)

        def add_missing(data):
            if isinstance(data, pd.DataFrame):
                missing_data = data.copy()
                cols = target_columns if target_columns else data.columns

                for col in cols:
                    if col in missing_data.columns:
                        mask = np.random.random(len(missing_data)) < missing_rate
                        missing_data.loc[mask, col] = np.nan

                return missing_data
            return data

        self.data_transform_fn = add_missing
        self.logger.info(f"Missing data injected: {missing_rate * 100}% rate")

    def _inject_schema_drift(self, params: Dict[str, Any]):
        """Simulate schema changes"""
        drift_type = params.get("drift_type", "add_column")

        def schema_drift(data):
            if isinstance(data, pd.DataFrame):
                drifted = data.copy()

                if drift_type == "add_column":
                    new_col = params.get("column_name", "unexpected_column")
                    drifted[new_col] = np.random.randn(len(drifted))
                elif drift_type == "remove_column":
                    col_to_remove = params.get("column_name")
                    if col_to_remove and col_to_remove in drifted.columns:
                        drifted = drifted.drop(columns=[col_to_remove])
                elif drift_type == "rename_column":
                    old_name = params.get("old_name")
                    new_name = params.get("new_name")
                    if old_name in drifted.columns:
                        drifted = drifted.rename(columns={old_name: new_name})
                elif drift_type == "type_change":
                    col = params.get("column_name")
                    if col in drifted.columns:
                        drifted[col] = drifted[col].astype(str)

                return drifted
            return data

        self.data_transform_fn = schema_drift
        self.logger.info(f"Schema drift injected: {drift_type}")

    def _inject_distribution_drift(self, params: Dict[str, Any]):
        """Inject concept/distribution drift"""
        drift_magnitude = params.get("magnitude", 0.5)
        target_columns = params.get("columns", None)

        def distribution_drift(data):
            if isinstance(data, pd.DataFrame):
                drifted = data.copy()
                cols = target_columns if target_columns else data.select_dtypes(include=[np.number]).columns

                for col in cols:
                    if col in drifted.columns and pd.api.types.is_numeric_dtype(drifted[col]):
                        # Shift mean and scale variance
                        mean_shift = drifted[col].mean() * drift_magnitude
                        scale_factor = 1 + drift_magnitude
                        drifted[col] = (drifted[col] + mean_shift) * scale_factor

                return drifted
            return data

        self.data_transform_fn = distribution_drift
        self.logger.info(f"Distribution drift injected: magnitude {drift_magnitude}")

    def _inject_outliers(self, params: Dict[str, Any]):
        """Inject outliers into data"""
        outlier_rate = params.get("outlier_rate", 0.05)
        magnitude = params.get("magnitude", 3.0)
        target_columns = params.get("columns", None)

        def add_outliers(data):
            if isinstance(data, pd.DataFrame):
                outlier_data = data.copy()
                cols = target_columns if target_columns else data.select_dtypes(include=[np.number]).columns

                for col in cols:
                    if col in outlier_data.columns and pd.api.types.is_numeric_dtype(outlier_data[col]):
                        mask = np.random.random(len(outlier_data)) < outlier_rate
                        std = outlier_data[col].std()
                        outlier_values = outlier_data[col].mean() + (magnitude * std * np.random.choice([-1, 1], mask.sum()))
                        outlier_data.loc[mask, col] = outlier_values

                return outlier_data
            return data

        self.data_transform_fn = add_outliers
        self.logger.info(f"Outliers injected: {outlier_rate * 100}% rate, magnitude {magnitude}σ")

    def _inject_latency(self, params: Dict[str, Any], duration: int):
        """Inject latency into data loading"""
        delay_ms = params.get("delay_ms", 1000)

        def add_latency(data):
            time.sleep(delay_ms / 1000.0)
            return data

        self.data_transform_fn = add_latency
        self.logger.info(f"Latency injection: {delay_ms}ms delay")

    def _inject_duplicates(self, params: Dict[str, Any]):
        """Inject duplicate records"""
        duplication_rate = params.get("duplication_rate", 0.1)

        def add_duplicates(data):
            if isinstance(data, pd.DataFrame):
                n_duplicates = int(len(data) * duplication_rate)
                duplicate_indices = np.random.choice(len(data), n_duplicates, replace=True)
                duplicates = data.iloc[duplicate_indices]
                return pd.concat([data, duplicates], ignore_index=True)
            return data

        self.data_transform_fn = add_duplicates
        self.logger.info(f"Duplicate records injected: {duplication_rate * 100}% rate")

    def apply_fault(self, data):
        """Apply active fault transformation to data"""
        if self.data_transform_fn:
            return self.data_transform_fn(data)
        return data

    def rollback(self):
        """Remove all data faults"""
        self.logger.info("Rolling back data faults")
        self.data_transform_fn = None
        self.original_data = None
        self.active_faults.clear()

    def validate(self, fault_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate fault parameters"""
        if fault_type not in self.SUPPORTED_FAULTS:
            return False

        # Validate common parameters
        if "corruption_rate" in parameters:
            rate = parameters["corruption_rate"]
            if not (0 <= rate <= 1):
                return False

        if "missing_rate" in parameters:
            rate = parameters["missing_rate"]
            if not (0 <= rate <= 1):
                return False

        return True

    def get_supported_faults(self) -> list:
        """Return supported fault types"""
        return self.SUPPORTED_FAULTS
