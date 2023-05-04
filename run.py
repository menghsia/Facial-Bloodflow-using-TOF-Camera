import os
import sys
import subprocess
import shutil
import argparse
import time
# import tensorflow as tf

from face_mesh_detector import FaceMeshDetector

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

def record_skv():
    """
        Launch Automotive Suite and wait until the user exits it.
        Once the user is done recording, they will close the program.
        After it is closed, move all of the newly recorded files to ./skvs/ and
        return the absolute path to ./skvs.
        If no new files, terminate the script.
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

# Launch ./automotive_suite_recorder_viewer_v0.0.0/automotive_suite.exe
# Wait until the user exits the program (automotive_suite.exe) and then continue

# # Launch the program
# process = subprocess.Popen("./automotive_suite_recorder_viewer_v0.0.0/automotive_suite.exe")

# # Wait for the program to exit
# process.wait()


#   - Once the user is done recording, they will close the program
#   - After it is closed, the .skv file will be processed into a .mat file


# Convert .skv video file into .mat file
#   - imx520_sample.exe -i skvs/input_skv.skv -o skvs/mat/output.mat


# Generate bloodflow signature .mat file
#   - facial_skBF_facemeshTrak.py
#   - Outputs to facial-blood-ToF/skv_mat/20221003-Sparsh_Fex_bfsig.mat


# Generate plots of heart rate from bloodflow signature .mat file
#   - process_thermal.m
#   - Outputs plots of the heart rate

def check_for_skvs(skvs_dir):
    skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(skvs_dir, x))), os.listdir(skvs_dir))))

    # if skvs_before_recording empty, exit
    if not skvs_before_recording:
        print('No .skv files found in ./skvs/')
        sys.exit()

def skv_to_mat(skvs_dir):
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

def mat_to_bfsig(skvs_dir):
    # Tag regions in face and generate bloodflow signature .mat file
    # myFaceMeshDetector = FaceMeshDetector(input_mats_dir="./skvs/mat/", output_bfsig_name="auto_bfsig")
    myFaceMeshDetector = FaceMeshDetector(input_dir=os.path.join(skvs_dir, "mat"), output_filename="auto_bfsig")
    myFaceMeshDetector.run(visualize_ROI=False, visualize_FaceMesh=False)

def bfsig_to_plot():
    # Run plotting matlab script
    # Create path to matlab script
    matlab_script_path = os.path.join(os.getcwd(), "auto_matlab/process_thermal_SINGLE.m")
    process = subprocess.run(["matlab", "-r", "run('" + matlab_script_path + "');"], shell=True)

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skv_to_mat', action='store_true')
    parser.add_argument('--mat_to_bfsig', action='store_true')
    parser.add_argument('--bfsig_to_plot', action='store_true')
    args = parser.parse_args()

    # If no args are provided, set all 3 bools to True
    if not any(vars(args).values()):
        args.skv_to_mat = True
        args.mat_to_bfsig = True
        args.bfsig_to_plot = True
    
    return args

if __name__ == '__main__':
    main_start_time = time.time()
    
    args = process_args()

    # Get the path to the new .skv file
    # skvs_dir = record_skv()

    skvs_dir = os.path.join(os.getcwd(), 'skvs')

    check_for_skvs(skvs_dir)

    if args.skv_to_mat:
        skv_to_mat(skvs_dir)
    
    if args.mat_to_bfsig:
        # start_time = time.time()
        mat_to_bfsig(skvs_dir)
        # end_time = time.time()
        # print("mat_to_bfsig() took " + str(end_time - start_time) + " seconds to run")
    
    if args.bfsig_to_plot:
        bfsig_to_plot()

    # print(tf.config.list_physical_devices('GPU'))

    print('Done!')

    main_end_time = time.time()
    print(f"run.py took {main_end_time - main_start_time} seconds to run")