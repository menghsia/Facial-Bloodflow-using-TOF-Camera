import os
import sys

# Check that two input arguments were provided
if len(sys.argv) != 3:
    print("Usage: python compare_mats.py file1.mat file2.mat")
    sys.exit(1)

# Get the input file names from the command line arguments
file1 = sys.argv[1]
file2 = sys.argv[2]

# Get the absolute paths of the input files
abs_file1 = os.path.abspath(file1)
abs_file2 = os.path.abspath(file2)

# Print the absolute paths for debugging purposes
# print("Absolute path to file 1: %s" % abs_file1)
# print("Absolute path to file 2: %s" % abs_file2)

print(f"Comparing {abs_file1} and {abs_file2}...")

# Check that the input files exist
if not os.path.isfile(abs_file1):
    print("Error: File %s not found" % abs_file1)
    sys.exit(1)

if not os.path.isfile(abs_file2):
    print("Error: File %s not found" % abs_file2)
    sys.exit(1)

# Compare the input files
# matlab -nosplash -nodesktop -wait -r "visdiff('C:\Users\Muhsinun\Desktop\Muhsinun\Repositories\GitHub\facial-bloodflow-tof_live\sk_automotive_20221003_164605_correct.skv.mat', 'C:\Users\Muhsinun\Desktop\Muhsinun\Repositories\GitHub\facial-bloodflow-tof_live\sk_automotive_20221003_164605_out.skv.mat');"
# Run the visdiff command
command = 'matlab -nosplash -nodesktop -wait -r "visdiff(\'%s\', \'%s\');"' % (abs_file1, abs_file2)
os.system(command)