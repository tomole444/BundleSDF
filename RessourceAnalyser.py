import time
import numpy as np
from datetime import datetime
import psutil
import pynvml


class RessourceAnalyser:
    def __init__(self, activated = True) -> None:
        pynvml.nvmlInit()

        self.current_gpu_usage = -1 # GPU load in MiB
        self.current_cpu_usage = -1 # CPU load in Percent
        self.current_memory_usage = -1 # Memory usage in MiB

        self.combined_info = {
            "GPU": [],
            "CPU": [],
            "memory": []
            }
        
        self.activated = activated

    def addDataPoint(self, meta_info):
        if not self.activated:
            return None
        #start_time = time.time()
        self.getGPUMemoryUsage()
        self.getMemoryUsage()
        self.getCPUMemoryUsage()

        gpu_dict = {
            "time": time.time(),
            "meta": meta_info,
            "usage": self.current_gpu_usage
        }
        cpu_dict = {
            "time": time.time(),
            "meta": meta_info,
            "usage": self.current_cpu_usage
        }
        memory_dict = {
            "time": time.time(),
            "meta": meta_info,
            "usage": self.current_memory_usage
        }
        
        self.combined_info["GPU"].append(gpu_dict)
        self.combined_info["CPU"].append(cpu_dict)
        self.combined_info["memory"].append(memory_dict)
        #print("retrieving takes ", time.time() - start_time)

    def save(self, file_name):
        if self.activated:
            save_arr = np.array(self.combined_info, dtype=object)
            np.save(file_name,save_arr, allow_pickle=True)
    
    def load(self, file_name):
        self.combined_info = np.load(file_name, allow_pickle=True).item()

    def getGPUMemoryUsage(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.current_gpu_usage =  info.used / 1024**2
        return self.current_gpu_usage

    def getMemoryUsage(self):
        self.current_memory_usage = psutil.virtual_memory().used / 1024**2  # in MiB
        return self.current_memory_usage

    def getCPUMemoryUsage(self):
        self.current_cpu_usage = psutil.cpu_percent()
        return self.current_cpu_usage
    

 

if __name__ == "__main__":
    ressource_analyser = RessourceAnalyser()

    for i in range(100):
        time.sleep(3)
        ressource_analyser.addDataPoint(None)
    