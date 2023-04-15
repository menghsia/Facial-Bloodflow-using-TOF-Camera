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
#include <thread>
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

		// Initialize matlab .mat output file
		MATFile* pmat;
		//mxArray* pa1, * pa2, * pa3, * pa4, * pa5;

		//int16_t data[9] = { 9, 9, 9, 9, 4, 3, 2, 1, 10000 };

		// Combine output_path with mats[skv_i], convert to c_str
		std::string out_file = output_path_string_forward_slash + "/" + mats[skv_i];
		const char* file = const_cast<char*>(out_file.c_str());

		//char str[BUFSIZE];
		//int status;

		printf("Creating file %s...\n\n", file);
		pmat = matOpen(file, "w");

		if (pmat == NULL) {
			printf("Error creating file %s\n", file);
			printf("(Do you have write permission in this directory?)\n");
			return(EXIT_FAILURE);
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

		const int num_threads = 16;
		std::mutex mutex_skv_reader;
		std::mutex mutex_cloudify_pt_x;
		std::mutex mutex_cloudify_pt_y;
		std::mutex mutex_cloudify_pt_z;

		// Create a vector to hold the thread objects
		std::vector<std::thread> threads;
		threads.reserve(num_threads);

		// Loop through each frame of this skv file
		for (size_t frame_num = 0; frame_num < num_frames; ++frame_num) {
			//std::cout << "frame_num: " << frame_num << std::endl;

			// Wait for a thread to finish if we have too many running
			/* TODO: There is a better way to manage the threads.
			* Right now, we are just waiting for that first thread to finish before starting a new one.
			* If that first thread takes a long time to finish, another one may have already finished,
			* but we would still be waiting for the first one, which means we might not be utilizing
			* all available threads.
			* 
			* Instead, we should use a thread pool that manages the threads for us. This way, we can
			* start a new thread as soon as one finishes, and we can also keep track of how many/which threads
			* are available to help us decide when to start a new one.
			*/
			if (threads.size() >= num_threads) {
				threads.front().join();
				threads.erase(threads.begin());
			}

			// Create a new thread and start it running the thread_function
			//threads.emplace_back(process_frame, frame_num, num_frames, skv_reader, depth_map_radial, confidence_map_radial, example_intrinsics, handle, err, cloudify_pt_x, cloudify_pt_y, cloudify_pt_z, Confidence);
			threads.emplace_back(std::thread(process_frame, frame_num, std::ref(num_frames),
				std::ref(mutex_skv_reader), std::ref(skv_reader),
				std::ref(example_intrinsics), std::ref(handle), std::ref(err),
				std::ref(mutex_cloudify_pt_x), std::ref(cloudify_pt_x),
				std::ref(mutex_cloudify_pt_y), std::ref(cloudify_pt_y),
				std::ref(mutex_cloudify_pt_z), std::ref(cloudify_pt_z),
				std::ref(Confidence)));

			//process_frame(frame_num, num_frames, skv_reader, depth_map_radial, confidence_map_radial, example_intrinsics, handle, err, cloudify_pt_x, cloudify_pt_y, cloudify_pt_z, Confidence);
			//threads[i] = std::thread(process_frame, frame_num, num_frames, skv_reader, depth_map_radial, confidence_map_radial, example_intrinsics, handle, err, cloudify_pt_x, cloudify_pt_y, cloudify_pt_z, Confidence);
		}

		// Wait for any remaining threads to finish before continuing with the rest of the program
		for (auto& thread : threads) {
			thread.join();
		}

		// Clear the vector of threads
		threads.clear();

		cloudify_release(&handle, &err);
		if (err.code != cloudify_success) {
			std::cout << err.message << std::endl;
			return -1;
		}
		std::cout << "[CLOUDIFY_SAMPLE] Library released" << std::endl;

		skv_reader->close();
		std::cout << "[CLOUDIFY_SAMPLE] SKV closed" << std::endl;

		// Save data to matlab .mat file

		std::list<std::tuple<int16_t(*)[600][307200], std::string, bool>> vars;
		vars.emplace_back(&Confidence, "grayscale", false);
		vars.emplace_back(&cloudify_pt_x, "x_value", true);
		vars.emplace_back(&cloudify_pt_y, "y_value", false);
		vars.emplace_back(&cloudify_pt_z, "z_value", false);

		const int num_threads_mat = 4;
		std::mutex mutex_pmat;

		// Loop through each variable to save
		for (const auto& var_i : vars) {

			if (threads.size() >= num_threads_mat) {
				threads.front().join();
				threads.erase(threads.begin());
			}

			threads.emplace_back(std::thread(write_to_mat_file, std::ref(mutex_pmat), std::ref(pmat), std::ref(*std::get<0>(var_i)), std::get<1>(var_i), std::get<2>(var_i)));

			//write_to_mat_file(pmat, *std::get<0>(var_i), std::get<1>(var_i), std::get<2>(var_i));
		}

		// Wait for any remaining threads to finish before continuing with the rest of the program
		for (auto& thread : threads) {
			thread.join();
		}

		if (matClose(pmat) != 0) {
			printf("Error closing file %s\n", file);
			return(EXIT_FAILURE);
		}

		std::cout << "Finished" << std::endl;
	}
}
