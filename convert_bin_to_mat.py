import numpy as np
from scipy.io import savemat
import os
import sys

def read_binary_file(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read a binary file containing x, y, z coordinates, and confidence values.

        Args:
            filepath: The path to the binary file to be read.

        Returns:
            A tuple containing four NumPy arrays: x_all, y_all, z_all, and confidence_all.
            - x_all: An (n,d) array of x-coordinates.
            - y_all: An (n,d) array of y-coordinates.
            - z_all: An (n,d) array of z-coordinates.
            - confidence_all: An (n,d) array of confidence values.

        This method reads a binary file and extracts x, y, z coordinates, and confidence values.
        The binary file is assumed to have a specific structure where each array is stored sequentially.

        Note: This method assumes that the binary file is properly formatted and contains valid data.

        Example usage:
            x, y, z, confidence = read_binary_file('data.bin')
        """
        x_all, y_all, z_all, confidence_all = None, None, None, None

        with open(filepath, 'rb') as binary_file:
            x_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            y_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            z_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            confidence_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()

        return x_all, y_all, z_all, confidence_all

def save_to_mat_file(x_all: np.ndarray, y_all: np.ndarray, z_all: np.ndarray, confidence_all: np.ndarray,
                          output_dir_path: str, filename: str) -> None:
        """
        Save the provided arrays to a MATLAB (.mat) file.

        Args:
            x_all: (n,d) array containing X-coordinate data.
            y_all: (n,d) array containing Y-coordinate data.
            z_all: (n,d) array containing Z-coordinate data.
            confidence_all: (n,d) array containing confidence values.
            output_dir_path: Path to the output directory.
            filename: Name of the output file (without extension).

        Returns:
            None
        """
        # mdic = {"Depth": D_signal, 'I_raw': I_signal, 'EAR': EAR} # EAR: eye aspect ratio
        # savemat(os.path.join(matpath, matname + '.mat'), mdic)

        mat_dict = {'x_all': x_all, 'y_all': y_all, 'z_all': z_all, 'confidence_all': confidence_all}
        savemat(os.path.join(output_dir_path, filename + '.mat'), mat_dict)
        # hdf5storage.write(mat_dict, output_dir_path, filename + '.mat', matlab_compatible=True)

        # Save using Matlab 7.3 to use HDF5 format (code does not currently work)
        # # make a dictionary to store the MAT data in
        # matfiledata = {}
        # # *** u prefix for variable name = unicode format, no issues thru Python 3.5
        # # advise keeping u prefix indicator format based on feedback despite docs ***
        # matfiledata[u'x_all'] = x_all
        # matfiledata[u'y_all'] = y_all
        # matfiledata[u'z_all'] = z_all
        # matfiledata[u'gray_all'] = gray_all
        # hdf5storage.write(matfiledata, output_dir_path, filename + '.mat', matlab_compatible=True)
        return

def main():
    # Check that an input argument was provided
    if len(sys.argv) != 2:
        print("Usage: python convert_bin_to_mat.py file_in.bin")
        sys.exit(1)

    # Get the input file name from the command line arguments
    file_in = sys.argv[1]
    # Set output file name to input file name without any extension
    file_out = os.path.splitext(os.path.basename(file_in))[0]

    # Get the absolute paths of the files
    abs_file_in = os.path.abspath(file_in)
    # Set the path of the output file to be in the same directory as the input file
    abs_file_out_dir = os.path.dirname(abs_file_in)

    # Print the absolute paths for debugging purposes
    # print("Absolute path to file_in: %s" % abs_file_in)
    # print("Absolute path to file_out_dir: %s" % abs_file_out_dir)
    # print(f"file_out: {file_out}")

    print(f"Converting {abs_file_in} to {os.path.normpath(os.path.join(abs_file_out_dir, file_out + '.mat'))}")

    # Check that the input file exists
    if not os.path.isfile(abs_file_in):
        print("Error: File %s not found" % abs_file_in)
        sys.exit(1)

    # Convert the input .bin file to a .mat file

    # Read the binary file
    x_all, y_all, z_all, confidence_all = read_binary_file(abs_file_in)

    # Save the data to a .mat file
    save_to_mat_file(x_all, y_all, z_all, confidence_all, abs_file_out_dir, file_out)
    return

if __name__ == '__main__':
    main()