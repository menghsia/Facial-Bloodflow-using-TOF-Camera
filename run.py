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