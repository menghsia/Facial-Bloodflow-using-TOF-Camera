import os
import sys
import subprocess
import shutil
import argparse
import time
from typing import Optional

from phase_two import PhaseTwo
import numpy as np

from processHR import ProcessHR as HRCalc

import csv

# Steps:
# - Use Automotive Suite to record a video clip (.skv file)
# - Copy skv file(s) to /skvs/
# - Open imx520_sample project in Visual Studio (use the long .exe profile)
# 	- We are debugging imx520_sample.exe (C:\Users\MNI Lab\Documents\GitHub\prgrm\facial-blood-ToF\imx520_sample\.build\output\imx520_sample.exe)
# - Run facial_skBF_facemeshTrak.py
# 	- Outputs to facial-blood-ToF/skv_mat/20221003-Sparsh_Fex_bfsig.mat
# - copy the bfsig output file to matlab_code
# 	- where??
# - open process_thermal.m
# - this outputs charts of the heart rate

# Record skv video file
#   - Open Automotive Suite to let user manually record a video clip (.skv file)

def record_skv() -> Optional[str]:
    """
    Launches Automotive Suite and waits until the user exits it.
    Once the user is done recording, they can close the program.
    After it is closed, moves all of the newly recorded files from the
    Automotive Suite recordings directory to the './skvs/' directory.
    
    Returns:
        str: The absolute path to the './skvs/' directory if new files were recorded,
        otherwise script terminates.
    
    NOTE: If no new files were recorded, terminates the script.
    """
    # Find the folder path that matches the pattern "./automotive_suite_recorder_viewer*"
    # print(os.getcwd())
    folder_name = next(filter(lambda x: x.startswith('automotive_suite_recorder_viewer'), os.listdir()), None)
    # print(f"folder_name: {folder_name}")
    
    if not folder_name:
        print('Automotive Suite not found at automotive_suite_recorder_viewer*/automotive_suite.exe')
        sys.exit()
    else:
        folder_path = os.path.join(os.getcwd(), folder_name)
        exe_path = os.path.join(folder_path, 'automotive_suite.exe')
        # print(f"folder_path: {folder_path}")
        # print(f"exe_path: {exe_path}")
    
        # Get set of files in ./automotive_suite_recorder_viewer*
        #   - Put all *.skv filenames and the datetime they were created in a set of tuples {("filename.skv", datetime_created))}
        #   - Ignore folders

        recordings_path = os.path.join(folder_path, 'RecordedMovies')

        skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(recordings_path, x))), os.listdir(recordings_path))))

        # skvs_before_recording = set()
        # for filename in os.listdir(recordings_path):
        #     print(f"filename: {filename}")
        #     if os.path.isfile(os.path.join(recordings_path, filename)) and filename.endswith('.skv'):
        #         # get the datetime the file was created
        #         datetime_created = os.path.getctime(os.path.join(recordings_path, filename))
        #         # add the filename and datetime_created to the set as a tuple
        #         skvs_before_recording.add((filename, datetime_created))




        # print("skvs before recording")
        # print(skvs_before_recording)
    
        # Launch the program and wait until the user exits it
        process = subprocess.run(exe_path, shell=True)
    
        skvs_after_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(recordings_path, x))), os.listdir(recordings_path))))
        # print("skvs after recording")
        # print(skvs_after_recording)
    
        # If sets are different, then new .skv file(s) were created
        # skvs_before_recording - skvs_after_recording = set of new .skv files
        if skvs_before_recording != skvs_after_recording:
            # Move the new .skv file(s) to ./skvs/
            new_skvs = skvs_after_recording - skvs_before_recording
            # print("new_skvs")
    
            # # Get absolute path to the new .skv file
            # new_skv_path = os.path.join(recordings_path, new_skv[0])

            for new_skv in new_skvs:
                # print(new_skv)
                new_skv_path = os.path.join(recordings_path, new_skv[0])
                # print(new_skv_path)
                shutil.move(new_skv_path, os.path.join(os.getcwd(), 'skvs'))
                print('Automotive Suite recorded new file: ' + new_skv_path)
            
            skv_dir = os.path.join(os.getcwd(), 'skvs')
    
            return skv_dir
        else:
            print('No new .skv file was recorded')
            sys.exit()

def check_for_skvs(skvs_dir: str):
    """
    Checks if there are any .skv files in ./skvs/.
    If there are no .skv files, terminates the script.

    Args:
        skvs_dir (str): absolute path to ./skvs/

    Returns:
        None
    """
    skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(skvs_dir, x))), os.listdir(skvs_dir))))

    # if skvs_before_recording empty, exit
    if not skvs_before_recording:
        print('No .skv files found in ./skvs/')
        sys.exit()
    
    return

def skv_to_bin(skvs_dir: str):
    """
    Convert each .skv file in skvs_dir to .bin file using imx520_sample.exe and save to ./skvs/mat/

    Args:
        skvs_dir (str): absolute path to ./skvs/

    Returns:
        None
    """
    ## For each .skv file in skvs_dir, convert to .mat file using imx520_sample.exe and save to ./skvs/mat/
    #for skv_filename in os.listdir(skvs_dir):
    #    # print("skv_filename:")
    #    # print(skv_filename)
    #    skv_path = os.path.join(skvs_dir, skv_filename)
    #    # print("skv_path:")
    #    # print(skv_path)
    #
    #    # Convert .skv video file into .mat file
    #    # Get absolute path to imx520_sample.exe
    #    imx520_sample_exe_path = os.path.join(os.getcwd(), "skv_to_mat/compiled_releases/r1/imx520_sample.exe")
    #    # Get absolute path to output .mat file
    #    output_mat_path = os.path.join(os.getcwd(), "skvs/mat/" + skv_filename + ".mat")
    #    # Run imx520_sample.exe
    #    process = subprocess.run([imx520_sample_exe_path, "-i", skv_path, "-o", output_mat_path], shell=True)

    # Convert all .skv video files in ./skvs/ into .mat files using imx520_sample.exe and save to ./skvs/mat/
    # Get absolute path to imx520_sample.exe
    imx520_sample_exe_path = os.path.join(os.getcwd(), "skv_to_mat/compiled_releases/r2_3/imx520_sample.exe")
    # Get absolute path to dir to save output .mat files to
    output_mat_dir = os.path.join(skvs_dir, "mat")
    # Run imx520_sample.exe
    # ./imx520_sample.exe -i ./skvs/ -o ./skvs/mat/ -d
    process = subprocess.run([imx520_sample_exe_path, "-i", skvs_dir, "-o", output_mat_dir, "-d"], shell=True)

    return

def bin_to_bfsig(skvs_dir: str, csvPath):
    """
    Take all .bin files in ./skvs/mat/ and convert to bloodflow signature .mat file using
    phase_two.py and save output to ./skvs/mat/

    Args:
        skvs_dir (str): absolute path to ./skvs/

    Returns:
        None
    """
    # Tag regions in face and generate bloodflow signature .mat file
    # myFaceMeshDetector = PhaseTwo(input_mats_dir="./skvs/mat/", output_bfsig_name="auto_bfsig")
    myPhaseTwo = PhaseTwo(input_dir=os.path.join(skvs_dir, "mat"), output_filename="auto_bfsig")
    myPhaseTwo.run(visualize_FaceMesh=False, visualize_ROIs=False)

    with open(csvPath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        firstCol = np.array(myPhaseTwo.chest_depth)
        firstCol = firstCol[:,0]
        firstCol = firstCol.tolist()
        firstCol = [[value] for value in firstCol]  # Convert each value to a single
        writer.writerows(firstCol)
    
    
    myPhaseTwo.clean_up()

    return

def bfsig_to_plot():
    """
    Plot bloodflow signature .mat file using process_thermal_SINGLE.m
    """
    # Run plotting matlab script
    # Create path to matlab script
    matlab_script_path = os.path.join(os.getcwd(), "auto_matlab/process_thermal_SINGLE.m")
    process = subprocess.run(["matlab", "-r", "run('" + matlab_script_path + "');"], shell=True)

    return

def process_args() -> argparse.Namespace:
    """
    Process command line arguments

    Returns:
        args (argparse.Namespace): parsed command line arguments

    Notes:
        If no command-line arguments are provided, all options are set to True by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--skv_to_bin', action='store_true', help='Convert .skv files to .bin files')
    parser.add_argument('--bin_to_bfsig', action='store_true', help='Take all .bin files and generate bloodflow signature .mat file')
    parser.add_argument('--bfsig_to_plot', action='store_true', help='Plot bloodflow signature .mat file')
    args = parser.parse_args()

    # If no args are provided, set all 3 bools to True
    if not any(vars(args).values()):
        args.skv_to_bin = True
        args.bin_to_bfsig = True
        args.bfsig_to_plot = True
    
    return args


def clear_directory(directory):

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_directory(file_path)
            os.rmdir(file_path)

def resetSkvs():
    skv = 'skvs'
    mat = os.path.join(skv, "mat")
    clear_directory(mat)
    for filename in os.listdir(skv):
        file_path = os.path.join(skv, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)



def run_main(csvPath):
    # TODO: Add checks to ensure that /skvs/ and /skvs/mat/ exist. If not, create them.
    
    print("beginning of main")
    main_start_time = time.time()
    
    args = process_args()

    # Get the path to the new .skv file
    # skvs_dir = record_skv()

    skvs_dir = os.path.join(os.getcwd(), 'skvs')

    check_for_skvs(skvs_dir)

    if args.skv_to_bin:
        print("SKV TO BIN")
        skv_to_bin(skvs_dir)
    
    if args.bin_to_bfsig:
        print("BIN TO BFSIG")
        start_time = time.time()
        bin_to_bfsig(skvs_dir, csvPath)
        end_time = time.time()
        print("mat_to_bfsig() took " + str(end_time - start_time) + " seconds to run")
    
    if args.bfsig_to_plot:
        print("BFSIG TO PLOT")
        bfsig_to_plot()

    print('Done!')


    main_end_time = time.time()
    print(f"run.py took {main_end_time - main_start_time} seconds to run")


def write_header_value_to_csv(header, value, filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([header, value])  # Writing the value


def process(inputDir, outputDir, calcHR = False):
    resetSkvs()
    outputFile = None
    if calcHR:
        outputFile = os.path.join(outputDir, "HRSummary.csv")
        with open(outputFile, 'w'):
            pass
    for directory in os.listdir(inputDir):
        for individualTest in os.listdir(os.path.join(inputDir, directory)):
            for file in os.listdir(os.path.join(inputDir, directory, individualTest)):
                fullFilePath = os.path.join(inputDir, directory, individualTest, file)
                extension = os.path.splitext(file)[1]
                destination = 'skvs'
               
                if extension == ".skv":
                    shutil.copy(fullFilePath, destination)

                csvPath = os.path.join(outputDir, individualTest + ".csv")
                run_main(csvPath)
                if calcHR:
                    outputImg = os.path.join(outputDir, individualTest + ".png")
                    HR = HRCalc.getHR(csvPath, outputImg)
                    write_header_value_to_csv(individualTest, HR, outputFile)
                resetSkvs()
                
                    






if __name__ == '__main__':
    process("7-6-23", "Data_7_6", calcHR = True)

"""if __name__ == '__main__':
    # TODO: Add checks to ensure that /skvs/ and /skvs/mat/ exist. If not, create them.
    
    print("beginning of main")
    main_start_time = time.time()
    
    args = process_args()

    # Get the path to the new .skv file
    # skvs_dir = record_skv()

    for directory in os.listdir("test_clips"):
        print(directory)

    # filepath = os.path.join(directory, filename)
    # if os.path.isfile(filepath):
    #     # Process the file
        
    #     #copy file from loadedFilesTo
    #     shutil.copy(filepath, destination)
    #     directInput(os.path.join("output", filename))

    skvs_dir = os.path.join(os.getcwd(), 'skvs')

    check_for_skvs(skvs_dir)

    if args.skv_to_bin:
        print("SKV TO BIN")
        skv_to_bin(skvs_dir)
    
    if args.bin_to_bfsig:
        print("BIN TO BFSIG")
        start_time = time.time()
        bin_to_bfsig(skvs_dir)
        end_time = time.time()
        print("mat_to_bfsig() took " + str(end_time - start_time) + " seconds to run")
    
    if args.bfsig_to_plot:
        print("BFSIG TO PLOT")
        bfsig_to_plot()

    print('Done!')


    main_end_time = time.time()
    print(f"run.py took {main_end_time - main_start_time} seconds to run")"""