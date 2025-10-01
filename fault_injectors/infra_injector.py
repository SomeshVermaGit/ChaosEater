"""
Infrastructure Layer Fault Injector
Simulates infrastructure failures: network issues, node failures, resource constraints
"""

import time
import random
import threading
import socket
from typing import Dict, Any, Optional, Callable
from .base_injector import BaseFaultInjector


class InfrastructureFaultInjector(BaseFaultInjector):
    """Injects faults into infrastructure layer"""

    SUPPORTED_FAULTS = [
        "network_latency",
        "packet_loss",
        "node_failure",
        "gpu_unavailable",
        "storage_bottleneck",
        "disk_full",
        "cpu_throttle",
        "memory_pressure",
        "network_partition",
        "dns_failure"
    ]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.active_threads = []
        self.stop_signals = {}

    def inject(self, fault_type: str, parameters: Dict[str, Any], duration: int):
        """Inject infrastructure fault"""
        if not self.validate(fault_type, parameters):
            raise ValueError(f"Invalid parameters for fault type: {fault_type}")

        self.logger.info(f"Injecting {fault_type} with params: {parameters}")

        if fault_type == "network_latency":
            self._inject_network_latency(parameters, duration)
        elif fault_type == "packet_loss":
            self._inject_packet_loss(parameters, duration)
        elif fault_type == "node_failure":
            self._inject_node_failure(parameters, duration)
        elif fault_type == "gpu_unavailable":
            self._inject_gpu_unavailable(parameters, duration)
        elif fault_type == "storage_bottleneck":
            self._inject_storage_bottleneck(parameters, duration)
        elif fault_type == "disk_full":
            self._inject_disk_full(parameters, duration)
        elif fault_type == "cpu_throttle":
            self._inject_cpu_throttle(parameters, duration)
        elif fault_type == "memory_pressure":
            self._inject_memory_pressure(parameters, duration)
        elif fault_type == "network_partition":
            self._inject_network_partition(parameters, duration)
        elif fault_type == "dns_failure":
            self._inject_dns_failure(parameters, duration)
        else:
            raise ValueError(f"Unsupported fault type: {fault_type}")

        self.active_faults.append(fault_type)

    def _inject_network_latency(self, params: Dict[str, Any], duration: int):
        """Inject network latency"""
        latency_ms = params.get("latency_ms", 100)
        jitter_ms = params.get("jitter_ms", 20)

        # Monkey-patch socket to add latency
        original_send = socket.socket.send
        original_recv = socket.socket.recv

        def delayed_send(self, data, *args, **kwargs):
            delay = (latency_ms + random.uniform(-jitter_ms, jitter_ms)) / 1000.0
            time.sleep(delay)
            return original_send(self, data, *args, **kwargs)

        def delayed_recv(self, bufsize, *args, **kwargs):
            delay = (latency_ms + random.uniform(-jitter_ms, jitter_ms)) / 1000.0
            time.sleep(delay)
            return original_recv(self, bufsize, *args, **kwargs)

        socket.socket.send = delayed_send
        socket.socket.recv = delayed_recv

        # Store originals for rollback
        self.config['_original_send'] = original_send
        self.config['_original_recv'] = original_recv

        self.logger.info(f"Network latency injected: {latency_ms}ms ±{jitter_ms}ms")

    def _inject_packet_loss(self, params: Dict[str, Any], duration: int):
        """Inject packet loss"""
        loss_rate = params.get("loss_rate", 0.1)

        original_send = socket.socket.send

        def lossy_send(self, data, *args, **kwargs):
            if random.random() < loss_rate:
                raise ConnectionError("Packet dropped (simulated)")
            return original_send(self, data, *args, **kwargs)

        socket.socket.send = lossy_send
        self.config['_original_send'] = original_send

        self.logger.info(f"Packet loss injected: {loss_rate * 100}%")

    def _inject_node_failure(self, params: Dict[str, Any], duration: int):
        """Simulate node failure"""
        failure_type = params.get("failure_type", "crash")  # crash, hang, slow
        target_service = params.get("target_service", None)

        if failure_type == "crash":
            # Simulate service crash
            def crash_simulator():
                self.logger.warning(f"Simulating node crash for {duration}s")
                # This would kill processes or containers in real implementation
                time.sleep(duration)

            thread = threading.Thread(target=crash_simulator, daemon=True)
            thread.start()
            self.active_threads.append(thread)

        elif failure_type == "hang":
            # Simulate hanging service
            def hang_simulator():
                self.logger.warning(f"Simulating node hang for {duration}s")
                # Block operations
                time.sleep(duration)

            thread = threading.Thread(target=hang_simulator, daemon=True)
            thread.start()
            self.active_threads.append(thread)

        self.logger.info(f"Node failure injected: {failure_type}")

    def _inject_gpu_unavailable(self, params: Dict[str, Any], duration: int):
        """Simulate GPU unavailability"""
        # Mock GPU unavailability by raising errors

        class GPUUnavailableError(Exception):
            pass

        # This would integrate with actual GPU frameworks (CUDA, etc.)
        self.config['gpu_unavailable'] = True
        self.logger.info("GPU unavailability injected")

    def _inject_storage_bottleneck(self, params: Dict[str, Any], duration: int):
        """Simulate storage I/O bottleneck"""
        iops_limit = params.get("iops_limit", 100)

        # Throttle file I/O operations
        import builtins
        original_open = builtins.open

        def throttled_open(*args, **kwargs):
            time.sleep(1.0 / iops_limit)  # Limit IOPS
            return original_open(*args, **kwargs)

        builtins.open = throttled_open
        self.config['_original_open'] = original_open

        self.logger.info(f"Storage bottleneck injected: {iops_limit} IOPS limit")

    def _inject_disk_full(self, params: Dict[str, Any], duration: int):
        """Simulate disk full condition"""
        # Mock disk full by intercepting write operations
        import builtins
        original_open = builtins.open

        def disk_full_open(file, mode='r', *args, **kwargs):
            if 'w' in mode or 'a' in mode:
                raise OSError("No space left on device (simulated)")
            return original_open(file, mode, *args, **kwargs)

        builtins.open = disk_full_open
        self.config['_original_open'] = original_open

        self.logger.info("Disk full condition injected")

    def _inject_cpu_throttle(self, params: Dict[str, Any], duration: int):
        """Simulate CPU throttling"""
        throttle_percent = params.get("throttle_percent", 50)

        def cpu_hog():
            stop_key = f"cpu_throttle_{id(self)}"
            self.stop_signals[stop_key] = False

            end_time = time.time() + duration
            while time.time() < end_time and not self.stop_signals[stop_key]:
                # Consume CPU cycles
                _ = sum(range(1000))
                time.sleep(0.001 * (100 - throttle_percent) / 100)

        thread = threading.Thread(target=cpu_hog, daemon=True)
        thread.start()
        self.active_threads.append(thread)

        self.logger.info(f"CPU throttle injected: {throttle_percent}% load")

    def _inject_memory_pressure(self, params: Dict[str, Any], duration: int):
        """Simulate memory pressure"""
        memory_mb = params.get("memory_mb", 1024)

        def memory_hog():
            stop_key = f"memory_pressure_{id(self)}"
            self.stop_signals[stop_key] = False

            # Allocate memory
            memory_blocks = []
            try:
                while len(memory_blocks) * 10 < memory_mb and not self.stop_signals[stop_key]:
                    memory_blocks.append(bytearray(10 * 1024 * 1024))  # 10MB blocks
                    time.sleep(0.1)

                # Hold memory for duration
                end_time = time.time() + duration
                while time.time() < end_time and not self.stop_signals[stop_key]:
                    time.sleep(1)
            finally:
                memory_blocks.clear()

        thread = threading.Thread(target=memory_hog, daemon=True)
        thread.start()
        self.active_threads.append(thread)

        self.logger.info(f"Memory pressure injected: {memory_mb}MB")

    def _inject_network_partition(self, params: Dict[str, Any], duration: int):
        """Simulate network partition"""
        blocked_hosts = params.get("blocked_hosts", [])

        original_connect = socket.socket.connect

        def blocked_connect(self, address):
            host = address[0] if isinstance(address, tuple) else address
            if host in blocked_hosts:
                raise ConnectionError(f"Network partition - {host} unreachable (simulated)")
            return original_connect(self, address)

        socket.socket.connect = blocked_connect
        self.config['_original_connect'] = original_connect

        self.logger.info(f"Network partition injected: blocked {len(blocked_hosts)} hosts")

    def _inject_dns_failure(self, params: Dict[str, Any], duration: int):
        """Simulate DNS resolution failure"""
        failure_rate = params.get("failure_rate", 0.5)

        import socket
        original_getaddrinfo = socket.getaddrinfo

        def failing_getaddrinfo(*args, **kwargs):
            if random.random() < failure_rate:
                raise socket.gaierror("DNS resolution failed (simulated)")
            return original_getaddrinfo(*args, **kwargs)

        socket.getaddrinfo = failing_getaddrinfo
        self.config['_original_getaddrinfo'] = original_getaddrinfo

        self.logger.info(f"DNS failure injected: {failure_rate * 100}% failure rate")

    def rollback(self):
        """Remove all infrastructure faults"""
        self.logger.info("Rolling back infrastructure faults")

        # Stop all active threads
        for key in self.stop_signals:
            self.stop_signals[key] = True

        for thread in self.active_threads:
            thread.join(timeout=2)

        # Restore original functions
        if '_original_send' in self.config:
            socket.socket.send = self.config['_original_send']
        if '_original_recv' in self.config:
            socket.socket.recv = self.config['_original_recv']
        if '_original_connect' in self.config:
            socket.socket.connect = self.config['_original_connect']
        if '_original_getaddrinfo' in self.config:
            import socket as sock_module
            sock_module.getaddrinfo = self.config['_original_getaddrinfo']
        if '_original_open' in self.config:
            import builtins
            builtins.open = self.config['_original_open']

        self.active_threads.clear()
        self.stop_signals.clear()
        self.active_faults.clear()
        self.config.clear()

    def validate(self, fault_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate fault parameters"""
        if fault_type not in self.SUPPORTED_FAULTS:
            return False

        # Validate rates
        if "loss_rate" in parameters:
            if not (0 <= parameters["loss_rate"] <= 1):
                return False

        if "failure_rate" in parameters:
            if not (0 <= parameters["failure_rate"] <= 1):
                return False

        return True

    def get_supported_faults(self) -> list:
        """Return supported fault types"""
        return self.SUPPORTED_FAULTS
