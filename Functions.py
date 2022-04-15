import numpy as np
import math
import PeakDetection
import Smoothing

ans = ''
peaks_info = ''
all_peaktimes = []


# finds time to compare for simultaneous peaks
def find_peaktimes(data, time, thres, peakAlgorithm):
    numCol = len(data.columns)
    times = []
    global all_peaktimes
    for columns in data.iloc[:, 2:numCol - 1]:
        height, peaktime = peak_time(data[columns], time, thres, peakAlgorithm)
        all_peaktimes = np.append(all_peaktimes, np.ceil(peaktime))
        times = np.append(times, np.ceil(peaktime))
    return times


def test_leak(data, inj, exit_time, thres, peakAlgorithm):
    global peaks_info
    global ans
    leak = 0
    for columns in data.iloc[:, 2:len(data.columns) - 1]:
        printed = 0
        height, ptimes = peak_time(data[columns], data['time'], thres, peakAlgorithm)
        ptimes = np.ceil(ptimes)
        times = find_peaktimes(data.loc[:, data.columns != columns], data['time'], thres, peakAlgorithm)
        for idx, x in np.ndenumerate(height):
            if x > thres and inj < ptimes[idx] < exit_time:
                simul_list = [ptimes[idx] - 2, ptimes[idx] - 1, ptimes[idx], ptimes[idx] + 1,
                              ptimes[idx] + 2]  # for simul peaks
                if len(set(simul_list).intersection(set(times))) <= 0:
                    if printed == 0:
                        ans += "\nLeakage present in " + columns
                        printed = 1
                    peaks_info += "\nPeak present in detector " + str(columns) + " at " + str(
                        ptimes[idx]) + " with value " + str(np.around(x, 3))
                    leak = 1
            if x < thres or ptimes[idx] < inj or ptimes[idx] > exit_time:
                peaks_info += "\nFalse peak present in detector " + str(columns) + " at " + str(
                    ptimes[idx]) + " with value " + str(np.around(x, 3))
    return leak


def peak_time(detector, time, thres, peakAlgorithm):
    if peakAlgorithm == 1:
        height, time = PeakDetection.peak(detector, time, thres)
    elif peakAlgorithm == 2:
        height, time = PeakDetection.cwt(detector, time, thres)
    elif peakAlgorithm == 3:
        height, time = PeakDetection.peakdet(detector, time, thres)
    elif peakAlgorithm == 4:
        height, time = PeakDetection.detect_peaks(detector, time, thres)
    else:
        height, time = PeakDetection.peak(detector, time, thres)
    # print(height,time)
    return height, time


def inj_and_exit_time(detector, time, thres, peakAlgorithm):
    if peakAlgorithm == 1:
        print("Find Peaks Python Selected")
        height, time = PeakDetection.peak(detector, time, thres)
    elif peakAlgorithm == 2:
        print("Continuous wavelet transform selected")
        height, time = PeakDetection.cwt(detector, time, thres)
    elif peakAlgorithm == 3:
        print("Maxima Method Selected")
        height, time = PeakDetection.peakdet(detector, time, thres)
    elif peakAlgorithm == 4:
        print("Tony Selected")
        height, time = PeakDetection.detect_peaks(detector, time, thres)
    else:
        print("Nothing selected")
        height, time = PeakDetection.peak(detector, time, thres)
    if height.size > 0:
        index = np.argmax(height)
        return height[index], time[index]
    return 0, 0


def find_bgrad(data):
    data_bg = data.iloc[:10]
    data_bg = data_bg.drop('time', axis=1)
    avg_rad = data_bg.stack().mean()
    return avg_rad


def data_leak(data, threshold, peakAlgorithm, smoothing):
    global ans
    global peaks_info
    global all_peaktimes
    ans = ''
    peaks_info = ''
    all_peaktimes = []
    thres, avgrad = find_threshold(data)
    if threshold != 0:
        thres = threshold
    if smoothing == 1:
        sav_data = Smoothing.smooth_data_np_average(data)
    elif smoothing == 2:
        sav_data = Smoothing.savgol(data)
    elif smoothing == 3:
        sav_data = Smoothing.convolve(data)
    elif smoothing == 4:
        sav_data = Smoothing.smooth_data_lowess(data)
    elif smoothing == 5:
        sav_data = Smoothing.exp_avg(data)
    else:
        sav_data = Smoothing.savgol(data)
    inj_height, inj_time = inj_and_exit_time(sav_data[sav_data.columns[1]], sav_data[sav_data.columns[0]], thres,
                                             peakAlgorithm)
    exit_height, exit_time = inj_and_exit_time(sav_data[sav_data.columns[-1]], sav_data[sav_data.columns[0]], thres,
                                               peakAlgorithm)
    find_peaktimes(sav_data, sav_data['time'], thres, peakAlgorithm)
    leak = test_leak(data, inj_time, exit_time, thres, peakAlgorithm)
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
        # threshold = avg_rad * 5
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
