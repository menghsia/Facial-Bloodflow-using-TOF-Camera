# compare two csv files
# 1. read two csv files
# 2. compare the two csv files
# 3. output the result to a new csv file

import csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def compare_csv(file1, file2):
    # read csv files
    data1 = read_csv(file1)
    data2 = read_csv(file2)
    # compare csv 
    # each line is like this: [227 202],[362 208],[359 246],[228 240]
    # compare the four points in each line
    # calculate the percentage of error in each point
    # print percentage of error
    # print the result to a new csv file
    result = []
    avg_error = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(len(data1)):
        # compare the four points in each line
        # print(data1[i])
        coord1 = [int(num) for coordinate in data1[i] for num in coordinate.replace("[", "").replace("]", "").split(" ")]
        coord2 = [int(num) for coordinate in data2[i] for num in coordinate.replace("[", "").replace("]", "").split(" ")]

        diff = np.array(coord1) - np.array(coord2)
        diff = np.abs(diff)
        
        percentage_error = diff / np.array(coord1)
        percentage_error *= 100.0
        
        avg_error += percentage_error
        
        result.append(percentage_error)


    avg_error /= len(data1)
    print(avg_error)
    # print the result to a new csv file
    with open('thanos_percent_error.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)
    
    return data1, data2

file1 = 'plot_results/good_bounding_box_corners.csv'
file2 = 'plot_results/bad_bounding_box_corners.csv'
data1, data2 = compare_csv(file1, file2)

