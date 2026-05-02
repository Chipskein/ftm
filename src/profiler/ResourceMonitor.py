import os
import psutil
import csv
import threading
from datetime import datetime

try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, 
        NVML_TEMPERATURE_GPU, nvmlDeviceGetPowerUsage, NVMLError
    )
    NVIDIA_INSTALLED = True
except ImportError:
    NVIDIA_INSTALLED = False

class ResourceMonitor:
    def __init__(self, output_path="resource_log.csv", interval=0.01, gpu_index=0):
        self.output_path = output_path
        self.interval = interval
        self.label = "idle"
        self._stop_event = threading.Event()
        self._thread = None
        self._process = psutil.Process(os.getpid())

        self.has_gpu = False
        if NVIDIA_INSTALLED:
            try:
                nvmlInit()
                self.gpu_handle = nvmlDeviceGetHandleByIndex(gpu_index)
                gpu_name = nvmlDeviceGetName(self.gpu_handle)
                mem = nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_total = mem.total / 1024**2
                print(f"Monitoring NVIDIA GPU: {gpu_name} ({gpu_mem_total:.0f} MB)")
                self.has_gpu = True
            except Exception as e:
                print(f"NVIDIA library present, but GPU initialization failed: {e}")
        else:
            print("NVIDIA library (nvidia-ml-py) not installed. Standard logging only.")

        header = [
            "timestamp", "label",
            "cpu_percent",
            "ram_used_mb", "ram_percent",
            "proc_ram_used_mb"
        ]
        
        if self.has_gpu:
            header += [
                "gpu_util_percent",
                "gpu_mem_used_mb", "gpu_mem_free_mb", "gpu_mem_total_mb",
                "gpu_temp_c", "gpu_power_w"
            ]

        with open(self.output_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def _sample(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        proc_ram = self._process.memory_info().rss / 1024**2
        row = [
            ts, self.label,
            cpu,
            round(ram.used / 1024**2, 1), ram.percent,
            round(proc_ram, 1)
        ]

        if self.has_gpu:
            try:
                util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = nvmlDeviceGetMemoryInfo(self.gpu_handle)
                temp = nvmlDeviceGetTemperature(self.gpu_handle, NVML_TEMPERATURE_GPU)
                try:
                    power = nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000
                except NVMLError:
                    power = "N/A"

                row += [
                    util.gpu,
                    round(mem.used / 1024**2, 1),
                    round(mem.free / 1024**2, 1),
                    round(mem.total / 1024**2, 1),
                    temp, power
                ]
            except NVMLError:
                row += ["err", "err", "err", "err", "err", "err"]

        return row

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
        
        if self.has_gpu:
            try:
                nvmlShutdown()
            except:
                pass

    def set_label(self, label):
        self.label = label