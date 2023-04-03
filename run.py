import os
import sys
import subprocess

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
        After it is closed, return the path to the new .skv file if it exists.
        Otherwise, terminate the script.
    """
    # Find the folder path that matches the pattern "./automotive_suite_recorder_viewer*"
    folder_name = next(filter(lambda x: x.startswith('./automotive_suite_recorder_viewer'), os.listdir()), None)
    
    if not folder_name:
        print('Automotive Suite not found at ./automotive_suite_recorder_viewer*')
        sys.exit()
    else:
        folder_path = os.path.join(os.getcwd(), folder_name)
        exe_path = os.path.join(folder_path, 'automotive_suite.exe')
    
        # Get list of files in ./automotive_suite_recorder_viewer_v*
        #   - Put all *.skv filenames and the datetime they were created in a set of tuples {("filename.skv", datetime_created))}
        #   - Ignore folders
        skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(folder_path, x))), os.listdir(folder_path))))
    
        # Launch the program and wait until the user exits it
        process = subprocess.run(exe_path)
    
        skvs_after_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(folder_path, x))), os.listdir(folder_path))))
    
        # If sets are different, then a new .skv file was created
        if skvs_before_recording != skvs_after_recording:
            # Get the new .skv file
            new_skv = next(iter(skvs_after_recording - skvs_before_recording))
    
            # file_name = new_skv[0]
            # file_datetime = new_skv[1]
    
            # Get absolute path to the new .skv file
            new_skv_path = os.path.join(folder_path, new_skv[0])
    
            print('Automotive Suite recorded new file: ' + new_skv_path)
    
            return new_skv_path
        else:
            print('No new .skv file was recorded')
            sys.exit()

# Launch ./automotive_suite_recorder_viewer_v0.0.0/automotive_suite.exe
# Wait until the user exits the program (automotive_suite.exe) and then continue

# Launch the program
process = subprocess.Popen("./automotive_suite_recorder_viewer_v0.0.0/automotive_suite.exe")

# Wait for the program to exit
process.wait()


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

if __name__ == '__main__':
    # Get the path to the new .skv file
    skv_path = record_skv()

    # Get filename of the .skv file without the path or extension
    skv_filename = os.path.basename(skv_path)[:-4]

    # Convert .skv video file into .mat file
    process = subprocess.run(["./skv_to_mat/compiled_releases/r1/imx520_sample.exe", "-i", skv_path, "-o", "./skvs/mat/" + skv_filename + ".mat"])

    # Run facial_skBF_facemeshTrak.py
    process = subprocess.run(["python", "./facial-skBF-facemeshTrak/facial_skBF_facemeshTrak.py"])

    # Run process_thermal.m
    process = subprocess.run(["matlab", "-r", "process_thermal"])

    print('Done!')