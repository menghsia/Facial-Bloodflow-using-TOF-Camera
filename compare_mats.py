# THIS DOES NOT CURRENTLY WORK. DO NOT USE.

import numpy as np
from scipy.io import loadmat
import os

# Load the first .mat file (skvs/mat/sk_automotive_20221003_164605.skv.mat)
# Get current working directory and append the path to the .mat file
cwd = os.getcwd()
# file1 = loadmat(cwd + "/skvs/mat/sk_automotive_20221003_164605.skv.mat")
file1 = loadmat(cwd + "/sk_automotive_20221003_164605_correct.skv.mat")

# Load the second .mat file (skvs/mat/sk_automotive_20221003_164625.skv.mat)
# Get current working directory and append the path to the .mat file
file2 = loadmat(cwd + "/sk_automotive_20221003_164605_out.skv.mat")

# Get the keys in both dictionaries
keys1 = set(file1.keys())
keys2 = set(file2.keys())

# print("file1: keys")
# print(keys1)

# print("file2: keys")
# print(keys2)

# # print keys1 __header__ and __version__ to see if they are the same
# print("file1: __header__")
# print(file1['__header__'])

# print("file2: __header__")
# print(file2['__header__'])

# print("file1: __globals__")
# print(file1['__globals__'])

# print("file2: __globals__")
# print(file2['__globals__'])

# print("file1: __version__")
# print(file1['__version__'])

# print("file2: __version__")
# print(file2['__version__'])

# Make sure the keys in the two dictionaries are identical
if keys1 != keys2:
    print("The files have different sets of variables.")
else:
    # Iterate over the keys and compare the values
    # Skip the __header__ key
    for key in keys1:
        if key != "__header__":
            # print(file1[key])
            if not np.array_equal(file1[key], file2[key]):
                print(f"The values for variable {key} are different.")
                break
    
    print("All variables have equivalent values.")


# import matlab.engine

# # Connect to the shared MATLAB engine
# eng = matlab.engine.connect_matlab()

# # eng.visdiff('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat');
# eng.eval("visdiff('sk_automotive_20221003_164605_correct.skv.mat', 'sk_automotive_20221003_164605_out.skv.mat')")

# # Disconnect from the MATLAB engine
# eng.quit()
