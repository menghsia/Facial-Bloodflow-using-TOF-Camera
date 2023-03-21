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
#include "mat.h"
#include<windows.h>+
#include <chrono>
#include <ctime>  
#include<string>  
#include <cstring>
#define BUFSIZE 256


using namespace std;

std::array<int16_t, 307200> x_buffer;
std::array<int16_t, 307200> y_buffer;
std::array<int16_t, 307200> z_buffer;

int main()
{

	iu456_error_details_t error;
	
	///************************************************************************/
	///* SKV DATA			                                                    */
	///************************************************************************/
	//std::string movie_path("D:\Dropbox\On semicondcutor proposal\ToF matlab\20210302-automotive_suite-v0.0.0-recorder_viewer-win64\Cloudify-v0.1.0-win64\cloudify_sample\src\sk_automotive_20210430_165528.skv"
	//);
	//std::unique_ptr<depthsense::skv::helper::skv> skv_reader(new depthsense::skv::helper::skv());  //

	//try
	//{
	//	skv_reader->open(movie_path);
	//}
	//catch (...)
	//{
	//	std::cout << "ERROR: Failed to open skv movie." << std::endl;
	//	return 1;
	//}

	// ******************************
	//  Initialize camera and driver
	// ******************************
	int number_of_measurement = 10;
	cout << "MEASUREMENT STARTED" << endl;
	int ii = 0;
	while (ii <= number_of_measurement)
	{
		bool continue_processing = true;
		//Get time stamp
		auto end = std::chrono::system_clock::now();
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		std::cout << "measurement started at " << std::ctime(&end_time);
		string participant_name = "Kaiwen_2_sitting_0707_";
		string participant_count = to_string(ii);
		string matfile_name = participant_name + participant_count + ".mat";
		int n = matfile_name.length();
		char* matfile_name_c = const_cast<char*>(matfile_name.c_str());
		// Initialize library
		if (!iu456_initialize(nullptr, nullptr, nullptr, nullptr, &error))
		{
			std::cout << "[ERROR] Failed to initialize library (" << error.message << ")" << std::endl;
			return 1;
		}

		// Get version
		iu456_version_t version = iu456_get_version();

		// Get list of attached devices
		const iu456_device_t** devices;
		size_t num_devices;

		if (!iu456_get_device_list(&devices, &num_devices, &error))
		{
			std::cout << "[ERROR] Failed to find devices (" << error.message << ")" << std::endl;
			return 1;
		}

		if (num_devices == 0)
		{
			std::cout << "[ERROR] No devices found" << std::endl;
			return 1;
		}

		// Create a handle with the first device in the list (devices[0])
		iu456_handle_t* handle;

		if (!iu456_create(&handle, devices[0], nullptr, nullptr, &error))
		{
			std::cout << "[ERROR] Failed to create device (" << error.message << ")" << std::endl;
			return 1;
		}

		// Get a list of configurations for this camera
		// This configuration contains the mode UID which defines, among other things, playback speed (fps)
		const iu456_configuration_info_t** configuration_list;
		size_t configuration_list_size;

		if (!iu456_get_configuration_list(handle, &configuration_list, &configuration_list_size, &error))
		{
			std::cout << "[ERROR] Failed to get configuration list (" << error.message << ")" << std::endl;
			return 1;
		}


		//                             -----------------
		// Set the desired camera mode AS ADVISED BY SDS
		//                             -----------------
		//
		// Examples:
		//
		// int camera_mode = 0x2D03;                       -> explicit camera mode, as given by SDS
		// int camera_mode = configuration_list[0]->uid    -> camera mode retrieved from configuration list

		int camera_mode = 0x2D01;
		if (!iu456_set_configuration_uid(handle, camera_mode, &error))
		{
			std::cout << "[ERROR] Failed to find mode (" << error.message << ")" << std::endl;
			return 1;
		}

		iu456_set_auto_exposure(handle, camera_mode, iu456_property_state_enabled, nullptr);

		// The list consists of three exposure time, minimum, typical, and maximum.

		size_t exposure_list_size;
		const int* exposure_range_values;
		if (!iu456_get_exposure_time_list(handle, camera_mode, &exposure_range_values, &exposure_list_size, &error))
		{
			std::cout << "[ERROR] Failed to get exposure time list = (" << error.message << ")" << std::endl;
			return 1;
		}

		int minimum_allowed_exposure_time = exposure_range_values[0];
		//cout << minimum_allowed_exposure_time << endl;

		int typical_allowed_exposure_time = exposure_range_values[1];
		//cout << typical_allowed_exposure_time << endl;

		int maximum_allowed_exposure_time = exposure_range_values[2];
		//cout << maximum_allowed_exposure_time << endl;

		// Adjust the exposure time of the currently running mode.
		// This function only works when the device is in a streaming state.
		// This value should be in the range returned by iu456_get_exposure_time_list()
		// Exposure times are expressed in microseconds.

		


		// Turn off filtering
		if (!iu456_set_filtering(handle, iu456_property_state_disabled, &error))
		{
			std::cout << "[ERROR] Failed to disable filtering (" << error.message << ")" << std::endl;
			return 1;
		}

		// Get camera configuration, including intrinsic lens parameters
		const iu456_configuration_info_t* config;

		if (!iu456_get_current_configuration(handle, &config, &error))
		{
			std::cout << "[ERROR] Failed to retrieve device configuration (" << error.message << ")" << std::endl;
			return 1;
		}

		/*cloudify_intrinsic_parameters intrinsics =
		{
			config->width,
			config->height,

			config->intrinsics->cx,
			config->intrinsics->cy,

			config->intrinsics->fx,
			config->intrinsics->fy,

			config->intrinsics->k1,
			config->intrinsics->k2,
			config->intrinsics->k3,
			0.f
		};

		cloudify_handle* handle_cloudify;
		cloudify_error_details err;
		cloudify_init(&intrinsics, &handle_cloudify, &err);
		if (err.code != cloudify_success)
		{
			std::cout << err.message << std::endl;
			return -1;
		}
		std::cout << "[CLOUDIFY_SAMPLE] library intialized" << std::endl;*/


		// **************
		//  Start camera
		// **************
		if (!iu456_start(handle, &error))
		{
			std::cout << "[ERROR] Failed to start device (" << error.message << ")" << std::endl;
			return 1;
		}

		int desired_exposure_time = 280;
		if (desired_exposure_time >= minimum_allowed_exposure_time &&
			desired_exposure_time <= maximum_allowed_exposure_time) //checking the exposure time is acceptable according camera mode
		{
			if (!iu456_set_exposure_time(handle, desired_exposure_time, &error))
			{
				std::cout << "[ERROR] Failed to set desired exposure time = (" << error.message << ")" << std::endl;
				system("pause");
				return 1;
			}
		}




		// **********
		//  Get data
		// **********

		//Define vectors to store frame values

		int minimum_vector_allocated_size = 1000; // number of element to pre-allocate in memory to avoid automatic reallocation by the system

		std::vector<int> Time_in_mus;
		std::vector<std::array<int16_t, 307200>> ConfLvl;
		std::vector<std::array<int16_t, 307200>> Depth;
		std::vector<std::array<int16_t, 307200>> pt_x;
		std::vector<std::array<int16_t, 307200>> pt_y;

		Time_in_mus.reserve(minimum_vector_allocated_size);
		ConfLvl.reserve(minimum_vector_allocated_size);
		Depth.reserve(minimum_vector_allocated_size);
		pt_x.reserve(minimum_vector_allocated_size);
		pt_y.reserve(minimum_vector_allocated_size);

		std::array<int16_t, 307200> local_image;

		// Run a loop to retrieve frames.
		// To avoid blocking interactive applications, this should run in a separate processing thread. (see Note below).


		//changing integration time
		// section to defined exposure time
		// Get a list of supported exposure time range for a given configuration.

		//std::vector<std::array<int16_t, 307200>> cloudify_pt_x;
		//std::vector<std::array<int16_t, 307200>> cloudify_pt_y;
		//std::vector<std::array<int16_t, 307200>> cloudify_pt_z;




		while (continue_processing)
		{
			const iu456_frame_t* frame;

			// Note: This is a blocking call. It will wait until a frame is ready before it returns.
			if (!iu456_get_last_frame(handle, &frame, -1, &error))
			{
				std::cout << "[iu456] Failed to get frame from device (" << error.message << ")" << std::endl;
				return 1;
			}

			// 'frame' now has all frame data available, such as 
			//   - frame->timestamp    (timestamp of the processed frame)
			//   - frame->depth        (pointer to a depth pixel buffer)
			//   - frame->confidence   (pointer to a confidence pixel buffer)
			//
			// Example usage:

			//std::cout << "Timestamp: " << frame->timestamp << std::endl;
			//std::cout << "First pixel depth value: " << frame->depth[0] << std::endl;
			//std::cout << "First pixel confidence value: " << frame->confidence[0] << std::endl;
			//std::cout << "First pixel x value: " << frame->x[0] << std::endl;
			//std::cout << "First pixel x value: " << frame->y[0] << std::endl;
			//std::cout << std::endl;


			

			// Store values per frame into vectors

			Time_in_mus.push_back(frame->timestamp);

			std::copy(frame->depth, frame->depth + 307200, std::begin(local_image));
			Depth.push_back(local_image);

			std::copy(frame->confidence, frame->confidence + 307200, std::begin(local_image));
			ConfLvl.push_back(local_image);

			std::copy(frame->x, frame->x + 307200, std::begin(local_image));
			pt_x.push_back(local_image);

			std::copy(frame->y, frame->y + 307200, std::begin(local_image));
			pt_y.push_back(local_image);

			//for (size_t y_idx = 0; y_idx < intrinsics.height; ++y_idx)
			//{
			//	for (size_t x_idx = 0; x_idx < intrinsics.width; ++x_idx)
			//	{
			//		size_t pixel_idx = x_idx + y_idx * intrinsics.width;
			//		float radial_input = frame->depth[pixel_idx];
			//		float cartesian_output;
			//		cloudify_compute_radial_to_cartesian_depth(handle_cloudify, x_idx, y_idx, radial_input, &cartesian_output, &err);
			//		if (err.code != cloudify_success)
			//		{
			//			std::cout << err.message << std::endl;
			//			return -1;
			//		}
			//		//std::cout << "[CLOUDIFY_SAMPLE] radial_input @[" << example_index[0] << ", " << example_index[1] << "] : " << radial_input << std::endl;
			//		//std::cout << "[CLOUDIFY_SAMPLE] cartesian_output @[" << example_index[0] << ", " << example_index[1] << "] : " << cartesian_output << std::endl;


			//		cloudify_position_3d position;
			//		cloudify_compute_3d_point(handle_cloudify, x_idx, y_idx, cartesian_output, &position, &err);
			//		if (err.code != cloudify_success)
			//		{
			//			std::cout << err.message << std::endl;
			//			return -1;
			//		}
			//		//std::cout << "[CLOUDIFY_SAMPLE] Cloudified \t @[" << example_index[0] << ", " << example_index[1] << ", " << radial_input << "] --> \t @[" << position.x << ", " << position.y << ", " << position.z << "]" << std::endl;
			//		x_buffer[pixel_idx] = static_cast<int16_t>(position.x);
			//		y_buffer[pixel_idx] = static_cast<int16_t>(position.y);
			//		z_buffer[pixel_idx] = static_cast<int16_t>(position.z);

			//	}
			//}

			//cloudify_pt_x.push_back(x_buffer);
			//cloudify_pt_y.push_back(y_buffer);
			//cloudify_pt_z.push_back(z_buffer);

			//std::cout << "frame id:" << frame->frame_id << std::endl;
			if (frame->timestamp > 25000000)
			{
				std::cout << "Acquisition Finished" << std::endl;
				
				break;
			}



			// At some point, processing should stop. In this example, we only process one frame.
		}
		std::cout << "numbr of frames: " << Time_in_mus.size() << std::endl;
		// Write vectors into csv files


		MATFile* pmat;
		mxArray* pa1, * pa2, * pa3, * pa4, * pa5;
		std::vector<int> myInts;
		myInts.push_back(1);
		myInts.push_back(2);
		printf("Accessing a STL vector: %d\n", myInts[1]);

		int16_t data[9] = { 9, 9, 9, 9, 4, 3, 2, 1, 10000 };
		std::cout << "numbr of frames: " << sizeof(data) << std::endl;
		const char* file = matfile_name_c;
		char str[BUFSIZE];
		int status;

		printf("Creating file %s...\n\n", file);
		pmat = matOpen(file, "w");
		if (pmat == NULL) {
			printf("Error creating file %s\n", file);
			printf("(Do you have write permission in this directory?)\n");
			return(EXIT_FAILURE);
		}

		pa1 = mxCreateNumericMatrix(307200, Time_in_mus.size(), mxINT16_CLASS, mxREAL);
		if (pa1 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa1)), (void*)&ConfLvl[0], 2 * Time_in_mus.size() * 307200);
		status = matPutVariable(pmat, "grayscale", pa1);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}

		mxDestroyArray(pa1);

		pa2 = mxCreateNumericMatrix(307200, Time_in_mus.size(), mxINT16_CLASS, mxREAL);
		if (pa2 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa2)), (void*)&Depth[0], 2 * Time_in_mus.size() * 307200);

		status = matPutVariableAsGlobal(pmat, "distance", pa2);
		if (status != 0) {
			printf("Error using matPutVariableAsGlobal\n");
			return(EXIT_FAILURE);
		}

		
		mxDestroyArray(pa2);

		pa3 = mxCreateNumericMatrix(307200, Time_in_mus.size(), mxINT16_CLASS, mxREAL);
		if (pa3 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa3)), (void*)&pt_x[0], 2 * Time_in_mus.size() * 307200);

		status = matPutVariable(pmat, "x_value", pa3);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}
		mxDestroyArray(pa3);
		

		pa4 = mxCreateNumericMatrix(307200, Time_in_mus.size(), mxINT16_CLASS, mxREAL);
		if (pa4 == NULL) {
			printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create mxArray.\n");
			return(EXIT_FAILURE);
		}

		memcpy((void*)(mxGetPr(pa4)), (void*)&pt_y[0], 2 * Time_in_mus.size() * 307200);

		status = matPutVariable(pmat, "y_value", pa4);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}
		mxDestroyArray(pa4);
	
		pa5 = mxCreateString(ctime(&end_time));
		if (pa5 == NULL) {
			printf("%s :  Out of memory on line %d\n", __FILE__, __LINE__);
			printf("Unable to create string mxArray.\n");
			return(EXIT_FAILURE);
		}

		status = matPutVariable(pmat, "start_time", pa5);
		if (status != 0) {
			printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
			return(EXIT_FAILURE);
		}

		
	
		/* clean up */
		
		
		mxDestroyArray(pa5);
		std::cout << ii;

		if (matClose(pmat) != 0) {
			printf("Error closing file %s\n", file);
			return(EXIT_FAILURE);
		}


		ii = ii + 1;

		// *************
		//  Stop camera
		// *************
		if (!iu456_stop(handle, nullptr))
		{
			return 1;
			std::cout << "destroy wrong ";

		}

		// ******************
		//  Shut down camera
		// ******************

		// Destroy handle
		if (handle)
		{
			iu456_destroy(handle, nullptr);
		}
		std::cout << "destroy ok ";

		// Clear memory
		if (devices)
		{
			for (size_t i = 0; i < num_devices; ++i)
				iu456_release_device(devices[i]);
			iu456_release_device_list(devices);
		}
		std::cout << "mem clear ok ";

		// Shut down driver
		iu456_shutdown(nullptr);
		std::cout << "shutdown ok ";

	}
	
	}
