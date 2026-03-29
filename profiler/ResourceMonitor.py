import psutil
import csv
import threading
from datetime import datetime
from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU,
    nvmlDeviceGetPowerUsage, NVMLError
)

class ResourceMonitor:
    def __init__(self, output_path="resource_log.csv", interval=0.5, gpu_index=0):
        self.output_path = output_path
        self.interval = interval
        self.label = "idle"
        self._stop_event = threading.Event()
        self._thread = None

        nvmlInit()
        self.gpu_handle = nvmlDeviceGetHandleByIndex(gpu_index)
        gpu_name = nvmlDeviceGetName(self.gpu_handle)
        mem = nvmlDeviceGetMemoryInfo(self.gpu_handle)
        self.gpu_mem_total = mem.total / 1024**2
        print(f"Monitoring GPU: {gpu_name} ({self.gpu_mem_total:.0f} MB)")

        with open(self.output_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "label",
                "cpu_percent",
                "ram_used_mb", "ram_percent",
                "gpu_util_percent",
                "gpu_mem_used_mb", "gpu_mem_free_mb", "gpu_mem_total_mb",
                "gpu_temp_c", "gpu_power_w"
            ])

    def _sample(self):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem  = nvmlDeviceGetMemoryInfo(self.gpu_handle)
        temp = nvmlDeviceGetTemperature(self.gpu_handle, NVML_TEMPERATURE_GPU)

        try:
            power = nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # mW → W
        except NVMLError:
            power = "N/A"

        return [
            ts, self.label,
            cpu,
            round(ram.used  / 1024**2, 1), ram.percent,
            util.gpu,
            round(mem.used  / 1024**2, 1),
            round(mem.free  / 1024**2, 1),
            round(mem.total / 1024**2, 1),
            temp, power
        ]

    def _loop(self):
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            row = self._sample()
            with open(self.output_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            self._stop_event.wait(self.interval)

    def start(self, label="running"):
        self.label = label
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        nvmlShutdown()

    def set_label(self, label):
        self.label = label