import pickle
import numpy as np
import time


class TimeAnalyser:
    def __init__(self) -> None:
        self.time_save = dict()

    def add(self, save_point_name:str, meta):
        if(save_point_name in self.time_save):
            time_points = self.time_save[save_point_name]
        else:
            time_points = list()
        time_point = dict()
        time_point["time"] = time.time()
        time_point["meta"] = meta
        time_points.append(time_point)
        self.time_save[save_point_name] = time_points
    
    def save(self, file_name):
        save_arr = np.array(self.time_save, dtype=object)
        np.save(file_name,save_arr, allow_pickle=True)
    def load(self, file_name):
        self.time_save = np.load(file_name, allow_pickle=True).item()

if __name__ == "__main__":
    timer = TimeAnalyser()

    timer.add("point1",0)
    time.sleep(1)
    timer.add("point1",1)
    time.sleep(1)
    timer.add("point1",3)
    time.sleep(1)


    timer.add("point2",0)
    time.sleep(1)
    timer.add("point2",1)
    time.sleep(1)

    timer.save("benchmarks/BuchVideo/time_analysis/test_time.npy")
    timer.load("benchmarks/BuchVideo/time_analysis/test_time.npy")
