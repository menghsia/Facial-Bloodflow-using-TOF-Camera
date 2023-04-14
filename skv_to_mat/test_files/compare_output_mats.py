import os

correct_path = "./mat_out_correct/"
out_path = "./mat_out/"

for correct_file in os.listdir(correct_path):
    if correct_file.endswith(".mat"):
        out_file = correct_file
        if os.path.isfile(os.path.join(out_path, out_file)):
            cmd = "python ../../compare_mats.py " + os.path.join(correct_path, correct_file) + " " + os.path.join(out_path, out_file)
            os.system(cmd)
        else:
            print("Matching out file not found for " + correct_file)
