"""
Example: Integrating ChaosEater with ML Pipeline
Demonstrates how to test resilience of a typical ML pipeline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append('..')

from chaos_controller import ChaosController, ChaosLayer
from fault_injectors.data_injector import DataFaultInjector
from fault_injectors.model_injector import ModelFaultInjector
from observers.metrics_observer import MetricsObserver


class MLPipeline:
    """Simple ML pipeline for testing"""

    def __init__(self):
        self.model = None
        self.data_injector = None
        self.model_injector = None

    def load_data(self):
        """Load sample data"""
        # Generate synthetic dataset
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y

        return df

    def preprocess_data(self, df, apply_faults=False):
        """Preprocess data (with optional chaos injection)"""
        if apply_faults and self.data_injector:
            df = self.data_injector.apply_fault(df)

        X = df.drop('target', axis=1)
        y = df['target']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        """Train model"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, X_test, apply_faults=False):
        """Make predictions (with optional chaos injection)"""
        model = self.model

        if apply_faults and self.model_injector:
            model = self.model_injector.wrap_model(self.model)

        return model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """Evaluate model"""
        return accuracy_score(y_true, y_pred)


def test_data_layer_resilience():
    """Test ML pipeline resilience to data faults"""
    print("\n" + "="*60)
    print("Testing Data Layer Resilience")
    print("="*60 + "\n")

    pipeline = MLPipeline()
    controller = ChaosController()

    # Setup injector
    data_injector = DataFaultInjector()
    controller.register_injector(ChaosLayer.DATA, data_injector)
    controller.register_observer(MetricsObserver())

    # Baseline (no faults)
    print("📊 Baseline Performance (No Faults)")
    df = pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(df, apply_faults=False)
    pipeline.train_model(X_train, y_train)
    y_pred = pipeline.predict(X_test, apply_faults=False)
    baseline_acc = pipeline.evaluate(y_test, y_pred)
    print(f"   Accuracy: {baseline_acc:.3f}\n")

    # Test with data corruption
    print("🔥 Testing with Data Corruption")
    experiment = controller.create_experiment(
        name="data_corruption_pipeline_test",
        layer=ChaosLayer.DATA,
        fault_type="data_corruption",
        parameters={"corruption_rate": 0.15},
        duration=30
    )

    pipeline.data_injector = data_injector
    data_injector.inject("data_corruption", {"corruption_rate": 0.15}, 30)

    df = pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(df, apply_faults=True)
    pipeline.train_model(X_train, y_train)
    y_pred = pipeline.predict(X_test, apply_faults=False)
    corrupted_acc = pipeline.evaluate(y_test, y_pred)

    print(f"   Accuracy: {corrupted_acc:.3f}")
    print(f"   Degradation: {(baseline_acc - corrupted_acc) * 100:.1f}%")

    # Update metrics
    if experiment.metrics:
        experiment.metrics.accuracy_degradation = (baseline_acc - corrupted_acc) * 100

    data_injector.rollback()

    # Test with missing data
    print("\n🔥 Testing with Missing Data")
    data_injector.inject("missing_data", {"missing_rate": 0.2}, 30)

    df = pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(df, apply_faults=True)

    # Fill missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    pipeline.train_model(X_train, y_train)
    y_pred = pipeline.predict(X_test, apply_faults=False)
    missing_acc = pipeline.evaluate(y_test, y_pred)

    print(f"   Accuracy: {missing_acc:.3f}")
    print(f"   Degradation: {(baseline_acc - missing_acc) * 100:.1f}%")

    data_injector.rollback()


def test_model_layer_resilience():
    """Test ML pipeline resilience to model faults"""
    print("\n" + "="*60)
    print("Testing Model Layer Resilience")
    print("="*60 + "\n")

    pipeline = MLPipeline()
    controller = ChaosController()

    # Setup injector
    model_injector = ModelFaultInjector()
    controller.register_injector(ChaosLayer.MODEL, model_injector)

    # Train baseline model
    df = pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(df, apply_faults=False)
    pipeline.train_model(X_train, y_train)

    # Baseline
    print("📊 Baseline Performance")
    y_pred = pipeline.predict(X_test, apply_faults=False)
    baseline_acc = pipeline.evaluate(y_test, y_pred)
    print(f"   Accuracy: {baseline_acc:.3f}\n")

    # Test with prediction corruption
    print("🔥 Testing with Prediction Corruption")
    pipeline.model_injector = model_injector
    model_injector.inject("prediction_corruption", {"corruption_rate": 0.1, "type": "random"}, 30)

    y_pred = pipeline.predict(X_test, apply_faults=True)
    corrupted_acc = pipeline.evaluate(y_test, y_pred)

    print(f"   Accuracy: {corrupted_acc:.3f}")
    print(f"   Degradation: {(baseline_acc - corrupted_acc) * 100:.1f}%")

    model_injector.rollback()


def test_inference_latency():
    """Test model inference under latency"""
    print("\n" + "="*60)
    print("Testing Inference Latency Resilience")
    print("="*60 + "\n")

    import time

    pipeline = MLPipeline()
    model_injector = ModelFaultInjector()

    # Setup
    df = pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(df, apply_faults=False)
    pipeline.train_model(X_train, y_train)

    # Baseline inference time
    print("📊 Baseline Inference")
    start = time.time()
    _ = pipeline.predict(X_test, apply_faults=False)
    baseline_time = time.time() - start
    print(f"   Time: {baseline_time*1000:.1f}ms\n")

    # With latency injection
    print("🔥 Testing with Slow Inference (500ms latency)")
    pipeline.model_injector = model_injector
    model_injector.inject("slow_inference", {"latency_ms": 500, "variance": 0.1}, 30)

    start = time.time()
    _ = pipeline.predict(X_test, apply_faults=True)
    slow_time = time.time() - start

    print(f"   Time: {slow_time*1000:.1f}ms")
    print(f"   Slowdown: {slow_time/baseline_time:.1f}x")

    model_injector.rollback()


def main():
    """Run all tests"""
    print("\n🦖 ChaosEater: ML Pipeline Resilience Testing")

    test_data_layer_resilience()
    test_model_layer_resilience()
    test_inference_latency()

    print("\n" + "="*60)
    print("✅ All resilience tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
