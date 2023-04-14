#include <iostream>
#include "iu456\iu456.h"
#include <cloudify.h>
#include "skv_wrapper.h"
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <time.h>
#include <cassert>
#include <numeric>
#include <iterator>
#include <stdio.h>
#include <string.h> /* For memcpy() */
#include <stdlib.h> /* For EXIT_FAILURE, EXIT_SUCCESS */
#include <mat.h>
#include <windows.h>
#include <chrono>
#include <ctime>
#include <string>
#include <cstring>
#include <filesystem>
#include <dirent.h>
#define BUFSIZE 256

#include "main.h"

using namespace std;

int main(int argc, char* argv[]) {
	// Get the input and output file paths. These paths are represented as follows:
	// C:\Users\user\Documents\input_file.skv
	std::tuple<pathMode, std::filesystem::path, std::filesystem::path> options = getOptions(argc, argv);
	pathMode path_mode = std::get<0>(options);
	const std::filesystem::path input_path = std::get<1>(options);
	const std::filesystem::path output_path = std::get<2>(options);

	// Save the input and output file paths as strings. We need to convert the backslashes to forward slashes.
	// If file mode, current form: C:\path\to\file\input_file.skv
	// If directory mode, current form: C:\path\to\dir\input_dir
	std::string input_path_string_forward_slash = input_path.string();
	std::string output_path_string_forward_slash = output_path.string();
	std::replace(input_path_string_forward_slash.begin(), input_path_string_forward_slash.end(), '\\', '/');
	std::replace(output_path_string_forward_slash.begin(), output_path_string_forward_slash.end(), '\\', '/');
	// If file mode, final form: C:/path/to/file/input_file.skv
	// If directory mode, final form: C:/path/to/dir/input_dir (no forward slash at the end)

	// File mode
	std::string input_file_name = "";
	std::string output_file_name = "";

	if (path_mode == file) {
		// Save the input and output file names as strings
		input_file_name = input_path.filename().string();
		output_file_name = output_path.filename().string();
	}

	iu456_error_details_t error;

	///************************************************************************/
	///* SKV DATA			                                                    */
	///************************************************************************/
	/// 
	HANDLE hFind;
	WIN32_FIND_DATA FindFileData;

	// skvs is a vector of strings that contains the names of all the .skv files in the input directory
	std::vector<std::string> skvs;

	// mats is a vector of strings that contains the names of all the .mats files that will be generated in the output directory
	std::vector<std::string> mats;
	
	// Fill skvs with the names of all the .skv files in the input directory
	std::string search_path = input_path_string_forward_slash + "/*.skv";

	hFind = FindFirstFile(search_path.c_str(), &FindFileData);

	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			skvs.push_back(string(FindFileData.cFileName));
		} while (FindNextFile(hFind, &FindFileData));

		FindClose(hFind);
	}

	std::cout << "Found " << skvs.size() << " .skv files" << std::endl;

	// Fill list of output file names
	if (path_mode == file) {
		mats.push_back(output_file_name);
	}
	else if (path_mode == directory) {
		for (size_t skv_i = 0; skv_i < skvs.size(); ++skv_i) {
			mats.push_back(skvs[skv_i] + ".mat");
		}
	}

	std::string movie_path = "";

	// Loop through all .skv files and process them
	for (size_t skv_i = 0; skv_i < skvs.size(); ++skv_i) {
		// Set output file name (in directory mode this is "file_name.skv.mat")
		char* matfile_name_c = const_cast<char*>(mats[skv_i].c_str());

		// Create skv handler (dynamically allocated in memory) to interface with each skv file
		std::unique_ptr<depthsense::skv::helper::skv> skv_reader(new depthsense::skv::helper::skv());

		if (path_mode == file) {
			// "path/to/input.skv"
			movie_path = input_path_string_forward_slash;
		}
		else {
			movie_path = input_path_string_forward_slash + "/" + skvs[skv_i];
		}

		// Attempt to open the skv movie
		try {
			skv_reader->open(movie_path);
		}
		catch (...) {
			std::cout << "ERROR: Failed to open skv movie." << std::endl;
			return 1;
		}

		// read and save camera intrinsics
		depthsense::skv::helper::camera_intrinsics camera_parameters = skv_reader->get_camera_intrinsic_parameters();
		
		// Get number of frames in skv file
		// Pretty sure I can use size_t here instead of int
		int num_frames = skv_reader->get_frame_count();

		// Print skv file info
		std::cout << "**** skv file info:" << std::endl;
		std::cout << "path			: " << movie_path << std::endl;
		std::cout << "frames count		: " << skv_reader->get_frame_count() << std::endl;
		std::cout << "frames dimensions	: width=" << camera_parameters.width <<
			" height=" << camera_parameters.height << std::endl;

		std::cout << "intrinsics		: fx=" << camera_parameters.focal_x <<
			" fy=" << camera_parameters.focal_y <<
			" cx=" << camera_parameters.central_x <<
			" cy=" << camera_parameters.central_y << std::endl;

		std::cout << "Lens			: k1=" << camera_parameters.k1 <<
			" k2=" << camera_parameters.k2 <<
			" k3=" << camera_parameters.k3 <<
			" p1=" << camera_parameters.p1 <<
			" p2=" << camera_parameters.p2 << std::endl;

		std::cout << "****************" << std::endl << std::endl << std::endl;

		// Initialize Cloudify library(?)
		cloudify_error_details error_details;
		const char* version;
		cloudify_get_version(&version, &error_details);
		std::cout << "[CLOUDIFY_SAMPLE] cloudify version : " << version << std::endl;

		// Load the intrinsics from the skv files

		// sk_automotive_20221003_164605.skv:
		// example_intrinsics.width: 640
		// example_intrinsics.height : 480
		// example_intrinsics.cx : 316.56
		// example_intrinsics.cy : 238.73
		// example_intrinsics.fx : 370.024
		// example_intrinsics.fy : 370.024
		// example_intrinsics.k1 : -0.0767163
		// example_intrinsics.k2 : -0.0198108
		// example_intrinsics.k3 : 0.00441911
		// example_intrinsics.k4 : 0

		// sk_automotive_20221003_164625.skv:
		// example_intrinsics.width: 640
		// example_intrinsics.height : 480
		// example_intrinsics.cx : 316.56
		// example_intrinsics.cy : 238.73
		// example_intrinsics.fx : 370.024
		// example_intrinsics.fy : 370.024
		// example_intrinsics.k1 : -0.0767163
		// example_intrinsics.k2 : -0.0198108
		// example_intrinsics.k3 : 0.00441911
		// example_intrinsics.k4 : 0

		cloudify_intrinsic_parameters example_intrinsics = 
		{
			//example_intrinsics.width: 640
			camera_parameters.width,
			//example_intrinsics.height : 480
			camera_parameters.height,

			//example_intrinsics.cx : 316.56
			camera_parameters.central_x,
			//example_intrinsics.cy : 238.73
			camera_parameters.central_y,

			//example_intrinsics.fx : 370.024
			camera_parameters.focal_x,
			//example_intrinsics.fy : 370.024 (why is this focal_x again? is it a bug?)
			camera_parameters.focal_x,

			//example_intrinsics.k1 : -0.0767163
			camera_parameters.k1,
			//example_intrinsics.k2 : -0.0198108
			camera_parameters.k2,
			//example_intrinsics.k3 : 0.00441911
			camera_parameters.k3,
			//example_intrinsics.k4 : 0
			0.f
		};

		cloudify_handle* handle;
		cloudify_error_details err;

		cloudify_init(&example_intrinsics, &handle, &err);

		if (err.code != cloudify_success) {
			std::cout << err.message << std::endl;
			return -1;
		}

		std::cout << "[CLOUDIFY_SAMPLE] library intialized" << std::endl;

		// example_intrinsics.width * example_intrinsics.height = 307200
		// 640 * 480 = 307200 pixels per frame
		// int16_t: 2 bytes
		// 2 bytes * 640 * 480 = 2457600 bytes = 2.3438 MB per frame
		// 2457600 bytes per frame * 600 frames = 1474560000 bytes per 600-frame skv file = 1.3733 GB per 600-frame skv file

		static int16_t cloudify_pt_x[600][307200];
		static int16_t cloudify_pt_y[600][307200];
		static int16_t cloudify_pt_z[600][307200];
		static int16_t Confidence[600][307200];

		// Still have to test if vectors will be plug-and-play
		/*std::vector<std::vector<int16_t>> cloudify_pt_x(num_frames, std::vector<int16_t>(307200));
		std::vector<std::vector<int16_t>> cloudify_pt_y(num_frames, std::vector<int16_t>(307200));
		std::vector<std::vector<int16_t>> cloudify_pt_z(num_frames, std::vector<int16_t>(307200));
		std::vector<std::vector<int16_t>> Confidence(num_frames, std::vector<int16_t>(307200));*/

		// Initialnize a buffer, save all depth data of a frame in 1d array
		std::vector<int16_t> depth_map_radial(example_intrinsics.width * example_intrinsics.height);
		std::vector<int16_t> confidence_map_radial(example_intrinsics.width * example_intrinsics.height);

		int return_val = 0;

		// Loop through each frame of this skv file
		for (size_t frame_num = 0; frame_num < num_frames; ++frame_num) {
			process_frame(frame_num, num_frames, skv_reader, depth_map_radial, confidence_map_radial, example_intrinsics, handle, err, cloudify_pt_x, cloudify_pt_y, cloudify_pt_z, Confidence);
		}

		cloudify_release(&handle, &err);
		if (err.code != cloudify_success) {
			std::cout << err.message << std::endl;
			return -1;
		}
		std::cout << "[CLOUDIFY_SAMPLE] Library released" << std::endl;

		skv_reader->close();
		std::cout << "[CLOUDIFY_SAMPLE] SKV closed" << std::endl;




		// Save data to matlab .mat file

		MATFile* pmat;
		mxArray* pa1, * pa2, * pa3, * pa4, * pa5;

		int16_t data[9] = { 9, 9, 9, 9, 4, 3, 2, 1, 10000 };

		// Combine output_path with mats[skv_i], convert to c_str
		std::string out_file = output_path_string_forward_slash + "/" + mats[skv_i];
		const char* file = const_cast<char*>(out_file.c_str());

		char str[BUFSIZE];
		int status;

		printf("Creating file %s...\n\n", file);
		pmat = matOpen(file, "w");

		if (pmat == NULL) {
			printf("Error creating file %s\n", file);
			printf("(Do you have write permission in this directory?)\n");
			return(EXIT_FAILURE);
		}

		pa1 = mxCreateNumericMatrix(307200, 600, mxINT16_CLASS, mxREAL);
		if (pa1 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa1)), (void*)&Confidence[0], 2 * 600 * 307200);
		status = matPutVariable(pmat, "grayscale", pa1);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}

		mxDestroyArray(pa1);

		pa2 = mxCreateNumericMatrix(307200, 600, mxINT16_CLASS, mxREAL);
		if (pa2 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa2)), (void*)&cloudify_pt_x[0], 2 * 600 * 307200);

		status = matPutVariableAsGlobal(pmat, "x_value", pa2);
		if (status != 0) {
			printf("Error using matPutVariableAsGlobal\n");
			return(EXIT_FAILURE);
		}


		mxDestroyArray(pa2);

		pa3 = mxCreateNumericMatrix(307200, 600, mxINT16_CLASS, mxREAL);
		if (pa3 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa3)), (void*)&cloudify_pt_y[0], 2 * 600 * 307200);

		status = matPutVariable(pmat, "y_value", pa3);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}
		mxDestroyArray(pa3);


		pa4 = mxCreateNumericMatrix(307200, 600, mxINT16_CLASS, mxREAL);
		if (pa4 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa4)), (void*)&cloudify_pt_z[0], 2 * 600 * 307200);

		status = matPutVariable(pmat, "z_value", pa4);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}
		mxDestroyArray(pa4);

		/*pa5 = mxCreateString(ctime(&end_time));
		if (pa5 == NULL) {
			printf("%s :  Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create string mxArray.\n");
			return(EXIT_FAILURE);
		}

		status = matPutVariable(pmat, "start_time", pa5);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}*/



		/* clean up */


		/*mxDestroyArray(pa5);
		std::cout << ii;*/

		if (matClose(pmat) != 0) {
			printf("Error closing file %s\n", file);
			return(EXIT_FAILURE);
		}

		std::cout << "Finished" << std::endl;
	}
}
