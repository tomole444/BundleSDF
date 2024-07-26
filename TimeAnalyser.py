import pickle
import numpy as np
import time
import matplotlib.pyplot as plt


class TimeAnalyser:
    def __init__(self, activated = True) -> None:
        self.time_save = dict()

        self.activated = activated

    def add(self, save_point_name:str, meta):
        if not self.activated:
            return None

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
        if self.activated:
            save_arr = np.array(self.time_save, dtype=object)
            np.save(file_name,save_arr, allow_pickle=True)
    def load(self, file_name):
        self.time_save = np.load(file_name, allow_pickle=True).item()

    def getSyncTimeByFrameID(self, time_pair):
        time_points_1 = self.time_save[time_pair[0]]
        time_points_2 = self.time_save[time_pair[1]]

        time_points_1_ids = []
        time_points_1_times = []
        time_points_2_ids = []
        time_points_2_times = []


        for time_elem in time_points_1:
            time_points_1_ids.append(time_elem["meta"])
            time_points_1_times.append(time_elem["time"])

        for time_elem in time_points_2:
            time_points_2_ids.append(time_elem["meta"])
            time_points_2_times.append(time_elem["time"])
        
        del_indices = []
        for index, id in enumerate(time_points_1_ids):
            if not id in time_points_2_ids:
                del_indices.append(index)

        time_points_1_ids_sync = [time_points_1_ids[i] for i in range(len(time_points_1_ids)) if i not in del_indices]
        time_points_1_times_sync = [time_points_1_times[i] for i in range(len(time_points_1_times)) if i not in del_indices]
        
        del_indices = []
        for index, id in enumerate(time_points_2_ids):
            if not id in time_points_1_ids_sync:
                del_indices.append(index)

        #time_points_2_ids_sync = [time_points_2_ids[i] for i in range(len(time_points_2_ids)) if i not in del_indices]
        time_points_2_times_sync = [time_points_2_times[i] for i in range(len(time_points_2_times)) if i not in del_indices]
        
        #now they are synced
        time_points_1_times_sync = np.array(time_points_1_times_sync)
        time_points_2_times_sync = np.array(time_points_2_times_sync)
        combined_times = time_points_2_times_sync - time_points_1_times_sync

        return time_points_1_ids_sync, combined_times

    def visualizeProcessPVNet(self):

        time_pair_1 = ("process_new_frame_pvnet", "invalidatePixelsByMask")
        time_pair_2 = ("invalidatePixelsByMask", "denoise_cloud")
        time_pair_3 = ("denoise_cloud", "pvnet adjust_every")
        time_pair_4 = ("pvnet adjust_every", "find_corres")
        time_pair_5 = ("find_corres", "selectKeyFramesForBA")
        time_pair_5_2 = ("selectKeyFramesForBA", "getFeatureMatchPairs")
        time_pair_6 = ("getFeatureMatchPairs", "optimizeGPU")
        time_pair_7 = ("optimizeGPU", "process_new_frame_pvnet done")
        time_pair_8 = ("process_new_frame_pvnet done", "runNoNerf done")

        time_pair_process = ("process_new_frame_pvnet", "process_new_frame_pvnet done")

        time_pair_1_ids, time_pair_1_execution_times = self.getSyncTimeByFrameID(time_pair_1) 
        time_pair_2_ids, time_pair_2_execution_times = self.getSyncTimeByFrameID(time_pair_2)
        time_pair_3_ids, time_pair_3_execution_times = self.getSyncTimeByFrameID(time_pair_3)
        time_pair_4_ids, time_pair_4_execution_times = self.getSyncTimeByFrameID(time_pair_4)
        time_pair_5_ids, time_pair_5_execution_times = self.getSyncTimeByFrameID(time_pair_5)
        time_pair_5_2_ids, time_pair_5_2_execution_times = self.getSyncTimeByFrameID(time_pair_5_2)
        time_pair_6_ids, time_pair_6_execution_times = self.getSyncTimeByFrameID(time_pair_6)
        time_pair_7_ids, time_pair_7_execution_times = self.getSyncTimeByFrameID(time_pair_7)
        time_pair_8_ids, time_pair_8_execution_times = self.getSyncTimeByFrameID(time_pair_8)

        time_pair_process_ids, time_pair_process_execution_times = self.getSyncTimeByFrameID(time_pair_process)



        plt.plot(time_pair_1_ids, time_pair_1_execution_times, label = "process_new_frame_pvnet, invalidatePixelsByMask")
        plt.plot(time_pair_2_ids, time_pair_2_execution_times, label = "invalidatePixelsByMask, denoise_cloud")
        plt.plot(time_pair_3_ids, time_pair_3_execution_times, label = "denoise_cloud, pvnet adjust_every")
        plt.plot(time_pair_4_ids, time_pair_4_execution_times, label = "pvnet adjust_every, find_corres")
        plt.plot(time_pair_5_ids, time_pair_5_execution_times, label = "find_corres, selectKeyFramesForBA")
        plt.plot(time_pair_5_2_ids, time_pair_5_2_execution_times, label = "selectKeyFramesForBA, getFeatureMatchPairs")
        plt.plot(time_pair_6_ids, time_pair_6_execution_times, label = "getFeatureMatchPairs, optimizeGPU")
        plt.plot(time_pair_7_ids, time_pair_7_execution_times, label = "optimizeGPU, process_new_frame_pvnet done")
        plt.plot(time_pair_8_ids, time_pair_8_execution_times, label = "process_new_frame_pvnet done, runNoNerf done")
        plt.plot(time_pair_process_ids, time_pair_process_execution_times, label = "whole process")

if __name__ == "__main__":
    timer = TimeAnalyser()

    # timer.add("point1",0)
    # time.sleep(1)
    # timer.add("point1",1)
    # time.sleep(1)
    # timer.add("point1",3)
    # time.sleep(1)


    # timer.add("point2",0)
    # time.sleep(1)
    # timer.add("point2",1)
    # time.sleep(1)

    # timer.save("benchmarks/BuchVideo/time_analysis/test_time.npy")
    timer.load("benchmarks/BuchVideo/time_analysis/timing_orig.npy")

    keys = timer.time_save.keys()
    print(keys)
    
    timer.visualizeProcessPVNet()

    ax = plt.gca()
    

    plt.legend(loc="upper left")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time [s]")
    ax.grid(True)
    plt.show()
