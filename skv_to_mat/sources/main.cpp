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
	std::filesystem::path input_path = std::get<1>(options);
	std::filesystem::path output_path = std::get<2>(options);

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

		/*std::vector<std::array<int16_t, 307200>> cloudify_pt_x;
		std::vector<std::array<int16_t, 307200>> cloudify_pt_y;
		std::vector<std::array<int16_t, 307200>> cloudify_pt_z;
		std::vector<std::array<int16_t, 307200>> Confidence;*/

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

		// Loop through each frame of this skv file
		for (size_t i = 0; i < num_frames; ++i) {
			std::cout << "Processing frame " << i << "..." << std::endl;

			// Get depth and confidence data of this frame (640x480=307200 pixels per frame)
			skv_reader->get_depth_image(i, depth_map_radial.data());
			skv_reader->get_confidence_image(i, confidence_map_radial.data());

			// std::cout << "example_intrinsics.width: " << example_intrinsics.width << std::endl; // 640
			// std::cout << "example_intrinsics.height: " << example_intrinsics.height << std::endl; // 480
			// std::cout << "example_intrinsics.cx: " << example_intrinsics.cx << std::endl;
			// std::cout << "example_intrinsics.cy: " << example_intrinsics.cy << std::endl;
			// std::cout << "example_intrinsics.fx: " << example_intrinsics.fx << std::endl;
			// std::cout << "example_intrinsics.fy: " << example_intrinsics.fy << std::endl;
			// std::cout << "example_intrinsics.k1: " << example_intrinsics.k1 << std::endl;
			// std::cout << "example_intrinsics.k2: " << example_intrinsics.k2 << std::endl;
			// std::cout << "example_intrinsics.k3: " << example_intrinsics.k3 << std::endl;
			// std::cout << "example_intrinsics.k4: " << example_intrinsics.k4 << std::endl;

			// Loop over each pixel of this frame (width then height)
			for (size_t j = 0; j < example_intrinsics.width; ++j) {
				//std::cout << "j: " << j << std::endl;
				for (size_t k = 0; k < example_intrinsics.height; ++k) {
					// We are looking at a single pixel

					size_t example_index[] = { j, k };
					size_t idx = example_index[0] + example_index[1] * example_intrinsics.width;
					float radial_input = depth_map_radial[idx];
					float cartesian_output;
					cloudify_compute_radial_to_cartesian_depth(handle, example_index[0], example_index[1], radial_input, &cartesian_output, &err);
					if (err.code != cloudify_success) {
						std::cout << err.message << std::endl;
						return -1;
					}

					cloudify_position_3d position;
					cloudify_compute_3d_point(handle, example_index[0], example_index[1], cartesian_output, &position, &err);
					if (err.code != cloudify_success) {
						std::cout << err.message << std::endl;
						return -1;
					}

					//std::cout << "[CLOUDIFY_SAMPLE] Cloudified frame #" << i << "\t @[" << example_index[0] << ", " << example_index[1] << ", " << radial_input << "] --> \t @[" << position.x << ", " << position.y << ", " << position.z << "]" << std::endl;
					cloudify_pt_x[i][idx] = (int16_t)position.x;
					cloudify_pt_y[i][idx] = (int16_t)position.y;
					cloudify_pt_z[i][idx] = (int16_t)position.z;
					Confidence[i][idx] = (int16_t)confidence_map_radial[idx];

				}
			}
			// This is: i / (frames_count = skv_reader->get_frame_count())
			//std::cout << i << std::endl;
			//std::cout << i << "\n";
		}

		cloudify_release(&handle, &err);
		if (err.code != cloudify_success) {
			std::cout << err.message << std::endl;
			return -1;
		}
		std::cout << "[CLOUDIFY_SAMPLE] Library released" << std::endl;

		skv_reader->close();
		std::cout << "[CLOUDIFY_SAMPLE] SKV closed" << std::endl;


		// ******************************
		//  Initialize camera and driver
		// ******************************
		/*int number_of_measurement = 10;
		cout << "measurement started" << endl;
		int ii = 0;
		while (ii <= number_of_measurement)
		{
			bool continue_processing = true;
			get time stamp
			auto end = std::chrono::system_clock::now();
			std::time_t end_time = std::chrono::system_clock::to_time_t(end);
			std::cout << "measurement started at " << std::ctime(&end_time);
			string participant_name = "kaiwen_2_sitting_0707_";
			string participant_count = to_string(ii);
			string matfile_name = participant_name + participant_count + ".mat";
			int n = matfile_name.length();
			char* matfile_name_c = const_cast<char*>(matfile_name.c_str());
				initialize library
			if (!iu456_initialize(nullptr, nullptr, nullptr, nullptr, &error))
			{
				std::cout << "[error] failed to initialize library (" << error.message << ")" << std::endl;
				return 1;
			}

				get version
			iu456_version_t version = iu456_get_version();

				get list of attached devices
			const iu456_device_t** devices;
			size_t num_devices;

			if (!iu456_get_device_list(&devices, &num_devices, &error))
			{
				std::cout << "[error] failed to find devices (" << error.message << ")" << std::endl;
				return 1;
			}

			if (num_devices == 0)
			{
				std::cout << "[error] no devices found" << std::endl;
				return 1;
			}

				create a handle with the first device in the list (devices[0])
			iu456_handle_t* handle;

			if (!iu456_create(&handle, devices[0], nullptr, nullptr, &error))
			{
				std::cout << "[error] failed to create device (" << error.message << ")" << std::endl;
				return 1;
			}

				get a list of configurations for this camera
				this configuration contains the mode uid which defines, among other things, playback speed (fps)
			const iu456_configuration_info_t** configuration_list;
			size_t configuration_list_size;

			if (!iu456_get_configuration_list(handle, &configuration_list, &configuration_list_size, &error))
			{
				std::cout << "[error] failed to get configuration list (" << error.message << ")" << std::endl;
				return 1;
			}


											-----------------
				set the desired camera mode as advised by sds
											-----------------

				examples:

				int camera_mode = 0x2d03;                       -> explicit camera mode, as given by sds
				int camera_mode = configuration_list[0]->uid    -> camera mode retrieved from configuration list

			int camera_mode = 0x2d01;
			if (!iu456_set_configuration_uid(handle, camera_mode, &error))
			{
				std::cout << "[error] failed to find mode (" << error.message << ")" << std::endl;
				return 1;
			}

			iu456_set_auto_exposure(handle, camera_mode, iu456_property_state_enabled, nullptr);

				the list consists of three exposure time, minimum, typical, and maximum.

			size_t exposure_list_size;
			const int* exposure_range_values;
			if (!iu456_get_exposure_time_list(handle, camera_mode, &exposure_range_values, &exposure_list_size, &error))
			{
				std::cout << "[error] failed to get exposure time list = (" << error.message << ")" << std::endl;
				return 1;
			}

			int minimum_allowed_exposure_time = exposure_range_values[0];
			cout << minimum_allowed_exposure_time << endl;

			int typical_allowed_exposure_time = exposure_range_values[1];
			cout << typical_allowed_exposure_time << endl;

			int maximum_allowed_exposure_time = exposure_range_values[2];
			cout << maximum_allowed_exposure_time << endl;

				adjust the exposure time of the currently running mode.
				this function only works when the device is in a streaming state.
				this value should be in the range returned by iu456_get_exposure_time_list()
				exposure times are expressed in microseconds.




				turn off filtering
			if (!iu456_set_filtering(handle, iu456_property_state_disabled, &error))
			{
				std::cout << "[error] failed to disable filtering (" << error.message << ")" << std::endl;
				return 1;
			}

				get camera configuration, including intrinsic lens parameters
			const iu456_configuration_info_t* config;

			if (!iu456_get_current_configuration(handle, &config, &error))
			{
				std::cout << "[error] failed to retrieve device configuration (" << error.message << ")" << std::endl;
				return 1;
			}*/




			//// **************
			////  Start camera
			//// **************
			//if (!iu456_start(handle, &error))
			//{
			//	std::cout << "[ERROR] Failed to start device (" << error.message << ")" << std::endl;
			//	return 1;
			//}

			//int desired_exposure_time = 280;
			//if (desired_exposure_time >= minimum_allowed_exposure_time &&
			//	desired_exposure_time <= maximum_allowed_exposure_time) //checking the exposure time is acceptable according camera mode
			//{
			//	if (!iu456_set_exposure_time(handle, desired_exposure_time, &error))
			//	{
			//		std::cout << "[ERROR] Failed to set desired exposure time = (" << error.message << ")" << std::endl;
			//		system("pause");
			//		return 1;
			//	}
			//}



			// **********
			//  Get data
			// **********

			//Define vectors to store frame values

			//int minimum_vector_allocated_size = 1000; // number of element to pre-allocate in memory to avoid automatic reallocation by the system

			//std::vector<int> Time_in_mus;
			//std::vector<std::array<int16_t, 307200>> ConfLvl;
			//std::vector<std::array<int16_t, 307200>> Depth;
			//std::vector<std::array<int16_t, 307200>> pt_x;
			//std::vector<std::array<int16_t, 307200>> pt_y;

			//Time_in_mus.reserve(minimum_vector_allocated_size);
			//ConfLvl.reserve(minimum_vector_allocated_size);
			//Depth.reserve(minimum_vector_allocated_size);
			//pt_x.reserve(minimum_vector_allocated_size);
			//pt_y.reserve(minimum_vector_allocated_size);

			//std::array<int16_t, 307200> local_image;

			// Run a loop to retrieve frames.
			// To avoid blocking interactive applications, this should run in a separate processing thread. (see Note below).


			//changing integration time
			// section to defined exposure time
			// Get a list of supported exposure time range for a given configuration.

			//std::vector<std::array<int16_t, 307200>> cloudify_pt_x;
			//std::vector<std::array<int16_t, 307200>> cloudify_pt_y;
			//std::vector<std::array<int16_t, 307200>> cloudify_pt_z;




			//while (continue_processing)
			//{
			//	const iu456_frame_t* frame;

			//	// Note: This is a blocking call. It will wait until a frame is ready before it returns.
			//	if (!iu456_get_last_frame(handle, &frame, -1, &error))
			//	{
			//		std::cout << "[iu456] Failed to get frame from device (" << error.message << ")" << std::endl;
			//		return 1;
			//	}

			//	// 'frame' now has all frame data available, such as 
			//	//   - frame->timestamp    (timestamp of the processed frame)
			//	//   - frame->depth        (pointer to a depth pixel buffer)
			//	//   - frame->confidence   (pointer to a confidence pixel buffer)
			//	//
			//	// Example usage:

			//	//std::cout << "Timestamp: " << frame->timestamp << std::endl;
			//	//std::cout << "First pixel depth value: " << frame->depth[0] << std::endl;
			//	//std::cout << "First pixel confidence value: " << frame->confidence[0] << std::endl;
			//	//std::cout << "First pixel x value: " << frame->x[0] << std::endl;
			//	//std::cout << "First pixel x value: " << frame->y[0] << std::endl;
			//	//std::cout << std::endl;


			//	

			//	// Store values per frame into vectors

			//	Time_in_mus.push_back(frame->timestamp);

			//	std::copy(frame->depth, frame->depth + 307200, std::begin(local_image));
			//	Depth.push_back(local_image);

			//	std::copy(frame->confidence, frame->confidence + 307200, std::begin(local_image));
			//	ConfLvl.push_back(local_image);

			//	std::copy(frame->x, frame->x + 307200, std::begin(local_image));
			//	pt_x.push_back(local_image);

			//	std::copy(frame->y, frame->y + 307200, std::begin(local_image));
			//	pt_y.push_back(local_image);

				////std::cout << "frame id:" << frame->frame_id << std::endl;
				//if (frame->timestamp > 25000000)
				//{
				//	std::cout << "Acquisition Finished" << std::endl;
				//	
				//	break;
				//}


				// At some point, processing should stop. In this example, we only process one frame.

		/*	std::cout << "numbr of frames: " << Time_in_mus.size() << std::endl;*/
			// Write vectors into csv files


		MATFile* pmat;
		mxArray* pa1, * pa2, * pa3, * pa4, * pa5;

		int16_t data[9] = { 9, 9, 9, 9, 4, 3, 2, 1, 10000 };
		//const char* file = matfile_name_c;
		const char* file = const_cast<char*>(output_path_string_forward_slash.c_str());
		char str[BUFSIZE];
		int status;

		//printf("Creating file %s...\n\n", file);
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


		//ii = ii + 1;

		//// *************
		////  Stop camera
		//// *************
		//if (!iu456_stop(handle, nullptr))
		//{
		//	return 1;
		//	std::cout << "destroy wrong ";

		//}
	}
}
