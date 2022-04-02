from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import math

ans = ''
peaks_info = ''
all_peaktimes = []


def savgol(data):
    for columns in data.iloc[:, 1:]:
        data[columns] = savgol_filter(data[columns], 101, 2)
    return data


# plt.plot(data['time'], w, 'b')  # high frequency noise removed

def peak(data, time, thres):
    # Find peaks
    peaks = find_peaks(data, height=thres, distance=15)  # thres,dist
    height = peaks[1]['peak_heights']  # list containing the height of the peaks
    peak_pos = time[peaks[0]]  # list containing the positions of the peaks
    return height, peak_pos

##finds time to compare for simultanous peaks
def find_peaktimes(data, time, thres):
    numCol = len(data.columns)
    times = []
    global all_peaktimes
    for columns in data.iloc[:, 2:numCol - 1]:
        height, peaktime = peak_time(data[columns], time, thres)
        all_peaktimes = np.append(all_peaktimes,np.ceil(peaktime))
        times = np.append(times,np.ceil(peaktime))
    return times


def test_leak(data, inj, exit, thres):
    global peaks_info
    global ans
    leak = 0
    for columns in data.iloc[:, 2:len(data.columns) - 1]:
        #print(columns)
        printed = 0
        height, ptimes = peak_time(data[columns], data['time'], thres)
        ptimes = np.ceil(ptimes)
        times = find_peaktimes(data.loc[:, data.columns != columns],data['time'],thres)
        #print(ptimes,times)
        for idx, x in np.ndenumerate(height):
            #print(columns)
            #print(x,ptimes[idx],thres,inj,exit)
            if x > thres and inj < ptimes[idx] < exit:
                #print("Hii")
                res = True
                list = [ptimes[idx]-2,ptimes[idx]-1,ptimes[idx],ptimes[idx]+1,ptimes[idx]+2]
                if len(set(list).intersection(set(times))) <= 0:
                    if printed==0:
                        ans += "\nLeakage present in " + columns
                        printed=1
                    peaks_info += "\nPeak present in detector " + str(columns) + " at " + str(
                        ptimes[idx]) + " with value " + str(np.around(x, 3))
                    leak = 1
            if x < thres or ptimes[idx] < inj or ptimes[idx] > exit:
                peaks_info += "\nFalse peak present in detector " + str(columns) + " at " + str(ptimes[idx])
    return leak, peaks_info



def is_leak(detector, time, inj, exit, name, thres):
    global peaks_info
    height, peaktime = peak_time(detector, time, thres)
    flag = 0
    leak = 0
    for idx, x in np.ndenumerate(height):
        if x > thres and inj < peaktime[idx] < exit:
            peaks_info += "\nPeak present in detector " + str(name) + " at " + str(
                peaktime[idx]) + " with value " + str(np.around(x, 3))
            leak = 1
        if x < thres or peaktime[idx] < inj or peaktime[idx] > exit:
            peaks_info += "\nFalse peak present in detector " + str(name) + " at " + str(peaktime[idx])
    return leak, peaks_info


def peak_time(detector, time, thres):
    height, time = peak(detector, time, thres)
    time = time.to_numpy(dtype='float32')
    return height, time


def inj_and_exit_time(detector, time, thres):
    height, time = peak(detector, time, thres)
    time = time.to_numpy(dtype='float32')
    if height.size > 0:
        index = np.argmax(height)
        return height[index], time[index]
    return 0, 0


def find_bgrad(data):
    data_bg = data.iloc[:10]
    data_bg = data_bg.drop('time', axis=1)
    avg_rad = data_bg.stack().mean()
    return avg_rad


def data_leak(data, threshold):
    global ans
    global peaks_info
    global all_peaktimes
    ans = ''
    peaks_info = ''
    all_peaktimes = []
    thres, avgrad = find_threshold(data)
    if threshold != 0:
        thres = threshold
    # plot_all(data['d1'],data['d2'],data['d3'],data['d4'],data['d5'],data['d6'],data['time'])
    sav_data = savgol(data)
    # plot_all(sav_data['d1'],sav_data['d2'],sav_data['d3'],sav_data['d4'],sav_data['d5'],sav_data['d6'],sav_data['time'])
    inj_height, inj_time = inj_and_exit_time(sav_data[sav_data.columns[1]], sav_data[sav_data.columns[0]], thres)
    exit_height, exit_time = inj_and_exit_time(sav_data[sav_data.columns[-1]], sav_data[sav_data.columns[0]], thres)
    numCol = len(sav_data.columns)
    flag = 0
    find_peaktimes(sav_data, sav_data['time'], thres)
    #print(all_peaktimes)
    leak,peaks_infocol = test_leak(data,inj_time,exit_time,thres)
    peaks_info+=peaks_infocol
    #for columns in sav_data.iloc[:, 2:numCol - 1]:
    #    leak, peak_infocol = is_leak(sav_data[columns], sav_data['time'], inj_time, exit_time, columns, thres)
    #    if leak == 1:
    #        ans += "\nLeakage present in " + columns
    #        flag = 1
    #        peaks_info += peak_infocol
    # print(inj_time,exit_time)
    if leak == 0:
        ans += "\nNo leakage present"
    return ans, inj_time, exit_time, thres, avgrad, peaks_info


def find_threshold(data):
    avg_rad = find_bgrad(data)
    threshold = avg_rad
    if avg_rad < 10:
        avg_rad = 10
        threshold = avg_rad * 5
        statistical_variation = 5 * math.sqrt(threshold)
        threshold = threshold + statistical_variation
    if 30 > avg_rad > 10:
        threshold = avg_rad * 5
        statistical_variation = 5 * math.sqrt(threshold)
        threshold = threshold + statistical_variation
    if 50 > avg_rad > 30:
        #threshold = avg_rad * 5
        statistical_variation = 5 * math.sqrt(threshold)
        threshold = threshold + statistical_variation
    if 100 > avg_rad > 50:
        statistical_variation = 5 * math.sqrt(avg_rad)
        threshold = threshold + statistical_variation
    if 100 < avg_rad < 200:
        statistical_variation = 3 * math.sqrt(avg_rad)
        threshold = threshold + statistical_variation
    if avg_rad > 200:
        threshold = avg_rad
    return threshold, avg_rad
