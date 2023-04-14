// main.h

#ifndef MAIN_H
#define MAIN_H

#include <filesystem>
#include <iostream>
#include <string>
// Needed to get the command line arguments
#include "getopt.h"

/* FUNCTION DECLARATIONS */

enum pathType {
	//unspecified,
	file,
	directory
};

void printHelp(char* argv[]);
void process_flag_i(const pathType& path_mode, const std::string& input_arg, std::filesystem::path& input_path, std::string& input_path_str_forward_slash, bool& input_dir_contains_SKV);
void process_flag_o(const pathType& path_mode, const std::string& output_arg, std::filesystem::path& output_path, std::string& output_path_str_forward_slash);
std::tuple<std::filesystem::path, std::filesystem::path, pathType> getOptions(int argc, char* argv[]);



/* FUNCTION DEFINITIONS */

inline void printHelp(char* argv[]) {
	std::cout << "Usage: " << argv[0] << " <-i:input input.skv or input_dir/>|<-o:output output.mat or output_dir/>|[-h]\n";
	std::cout << "This program takes in an input .skv file (or dir of .skv files) and converts it into an output .mat file (or fills a dir with .mat files).\n";
}

/*
* Sets input_path to the absolute path of the input file/directory.
*/
inline void process_flag_i(const pathType& path_mode, const std::string& input_arg, std::filesystem::path& input_path, std::string& input_path_str_forward_slash, bool& input_dir_contains_SKV) {
	// Convert input_path to a std::filesystem::path object.
	input_path = std::filesystem::path(input_arg);

	// Check if input_path is a relative path. If it is, convert it to an absolute path.
	input_path = std::filesystem::absolute(input_path);

	/* NOTE: Because I wanted to use std::filesystem to check if files exist, I had
	to upgrade from C++14 to C++20 (filesystem requires C++17 or higher). */

	input_path_str_forward_slash = input_path.string();
	std::replace(input_path_str_forward_slash.begin(), input_path_str_forward_slash.end(), '\\', '/');

	if (path_mode == file) {
		// file mode

		// Check if input_path exists. If not, print an error message with full path and exit.
		if (!std::filesystem::exists(input_path)) {
			std::cerr << "Input file does not exist\n";
			std::cerr << "  Input file specified: " << input_path_str_forward_slash
				<< "\n";
			std::exit(1);
		}

		// Check if input_path is a .skv file. If not, exit.
		if (input_path.extension() != ".skv") {
			std::cerr << "Input file must be .skv file\n";
			std::cerr << "  Input file specified: " << input_path_str_forward_slash
				<< "\n";
			std::exit(1);
		}
	}
	else {
		// directory mode
		
		// Check if input_path exists. If not, print an error message with full path and exit.
		if (!std::filesystem::exists(input_path)) {
			std::cerr << "Input directory does not exist\n";
			std::cerr << "  Input directory specified: " << input_path_str_forward_slash
				<< "\n";
			std::exit(1);
		}

		// Check if directory contains any .skv files. If not, print an error message with full path and exit.
		input_dir_contains_SKV = false;

		for (const auto& instance_file : std::filesystem::directory_iterator(input_path)) {
			if (instance_file.path().extension() == ".skv") {
				input_dir_contains_SKV = true;
				break;
			}
		}

		if (!input_dir_contains_SKV) {
			std::cerr << "Input directory does not contain any .skv files\n";
			std::cerr << "  Input directory specified: " << input_path_str_forward_slash
				<< "\n";
			std::exit(1);
		}
	}
}

/*
* Sets output_path to the absolute path of the output file/directory.
*/
inline void process_flag_o(const pathType& path_mode, const std::string& output_arg, std::filesystem::path& output_path, std::string& output_path_str_forward_slash) {
	// Convert output_path to a std::filesystem::path object.
	output_path = std::filesystem::path(output_arg);

	// Check if output_path is a relative path. If it is, convert it to an absolute path.
	output_path = std::filesystem::absolute(output_path);

	output_path_str_forward_slash = output_path.string();
	std::replace(output_path_str_forward_slash.begin(), output_path_str_forward_slash.end(), '\\', '/');

	//// Check if output_path is a directory or a file.
	//if (std::filesystem::is_directory(output_path)) {
	//	// output_path is a directory.
	//	path_mode = directory;

	//	// Check if directory is empty.
	//	if (std::filesystem::is_empty(output_path)) {
	//		std::cerr << "Output directory is empty\n";
	//		std::cerr << "  Output directory specified: " << output_path_str_forward_slash
	//			<< "\n";
	//		exit(1);
	//	}
	//}
	//else {
	//	// output_path is a file.
	//	path_mode = file;

	//	// Check if output_path exists. If it does, notify the user that it will be overwritten.
	//	if (std::filesystem::exists(output_path)) {
	//		std::cout << "Output file already exists. It will be overwritten.\n";
	//		std::cout << "  Output file specified: " << output_path_str_forward_slash
	//			<< "\n";
	//	}

	//	// Check if output_path is a .mat file. If not, exit.
	//	if (output_path.extension() != ".mat") {
	//		std::cerr << "Output file must be .mat file\n";
	//		std::cerr << "  Output file specified: " << output_path_str_forward_slash
	//			<< "\n";
	//		exit(1);
	//	}
	//}

	// Check if output_path already exists
	if (std::filesystem::exists(output_path)) {
		// output_path already exists. We don't know if it is a file or directory.

		// Check if output_path is a file or directory
		if (std::filesystem::is_regular_file(output_path)) {
			// output_path is a file
			std::cout << "output_path exists and is a file";
		}
		else if (std::filesystem::is_directory(output_path)) {
			// output_path is a directory
			std::cout << "output_path exists and is a directory";
		}
		else {
			// output_path is neither a file nor a directory.
			std::cerr << "Output path exists, but is neither a file nor a directory\n";
			std::cerr << "  Output path specified: " << output_path_str_forward_slash
				<< "\n";
			exit(1);
		}
	}
	else {
		// output_path does not exist.

		// Check if output_path is a directory or a file.
		std::cout << "output_path does not exist";
	}
}

/**
 * @param argc The number of command line arguments.
 * @param argv An array of C-style strings containing the command line arguments.
 * @return A tuple (path_mode, input_path, output_path)
 */
inline std::tuple<pathType, std::filesystem::path, std::filesystem::path> getOptions(int argc, char* argv[]) {
	std::string input_arg;
	std::filesystem::path input_path;
	std::string input_path_str_forward_slash;
	bool input_specified = false;

	std::string output_arg;
	std::filesystem::path output_path;
	std::string output_path_str_forward_slash;
	bool output_specified = false;

	// Assume we are working with individual files by default, unless the user specifies -d flag (directory mode)
	pathType path_mode = file;
	//pathType outputType;

	bool input_dir_contains_SKV = false;

	// These are used with getopt_long()
	// Let us handle all error output for command line options
	opterr = false;
	int choice;
	int option_index = 0;
	option long_options[] = {
		/*{ "stack", no_argument, nullptr, 's' },
		{ "queue", no_argument, nullptr, 'q' },
		{ "output", required_argument, nullptr, 'o' },*/
		{ "input", required_argument, nullptr, 'i' },
		{ "output", required_argument, nullptr, 'o' },
		{ "directory-mode", no_argument, nullptr, 'd' },
		{ "help", no_argument, nullptr, 'h' },
		{ nullptr, 0,                 nullptr, '\0' }
	};

	// Fill in the double quotes, to match the mode and help options.
	while ((choice = getopt_long(argc, argv, "i:o:dh", long_options, &option_index)) != -1) {
		switch (choice) {
		case 'i':
			if (!input_specified) {
				input_specified = true;
			}
			else {
				std::cerr << "Input path was specified more than once.\n";
				std::exit(1);
			}
			input_arg = optarg;

			process_flag_i(path_mode, input_arg, input_path, input_path_str_forward_slash, input_dir_contains_SKV);

			break;

		case 'o':
			if (!output_specified) {
				output_specified = true;
			}
			else {
				std::cerr << "Output path was specified more than once.\n";
				std::exit(1);
			}
			output_arg = optarg;

			process_flag_o(path_mode, output_arg, output_path, output_path_str_forward_slash);

			break;

		case 'd':
			path_mode = directory;

			break;

		case 'h':
			printHelp(argv);
			std::exit(0);

		default:
			std::cerr << "Error: invalid option\n";
			std::exit(1);
		}
	}

	if (!input_specified) {
		std::cerr << "No input file specified\n";
		std::exit(1);
	}

	if (!output_specified) {
		std::cerr << "No output file specified\n";
		std::exit(1);
	}

	return std::make_tuple(path_mode, input_path, output_path);
}



#endif /* MAIN_H */
