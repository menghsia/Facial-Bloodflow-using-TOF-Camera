// main.h

#ifndef MAIN_H
#define MAIN_H

#include <filesystem>
#include <iostream>
#include <string>
// Needed to get the command line arguments
#include "getopt.h"

/* FUNCTION DECLARATIONS */

enum pathMode {
	//unspecified,
	file,
	directory
};

void printHelp(char* argv[]);
void process_flag_i(const pathMode& path_mode, const std::string& input_arg, std::filesystem::path& input_path, std::string& input_path_str_forward_slash, bool& input_dir_contains_SKV);
void process_flag_o(const pathMode& path_mode, const std::string& output_arg, std::filesystem::path& output_path, std::string& output_path_str_forward_slash, bool& dir_created);
std::tuple<pathMode, std::filesystem::path, std::filesystem::path> getOptions(int argc, char* argv[]);



/* FUNCTION DEFINITIONS */

inline void printHelp(char* argv[]) {
	std::cout << "Usage: " << argv[0] << " <-i:input [input.skv]|[path/to/input_dir/]>|<-o:output [output.mat]|[path/to/output_dir/]>|[-d]|[-h]\n";
	std::cout << "This program takes in an input .skv file (or dir of .skv files) and converts it into an output .mat file (or fills a dir with .mat files).\n";
	std::cout << "By default, the program assumes file input-output mode. Add -d to enable directory input-output mode.\n";
	std::cout << "Both the input and output must be files or directories, depending on the input-output mode. They must match type.\n";
}

/*
* Sets input_path to the absolute path of the input file/directory.
*/
inline void process_flag_i(const pathMode& path_mode, const std::string& input_arg, std::filesystem::path& input_path, std::string& input_path_str_forward_slash, bool& input_dir_contains_SKV) {
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
		if (!std::filesystem::is_regular_file(input_path)) {
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
		if (!std::filesystem::is_directory(input_path)) {
			std::cerr << "Input directory does not exist\n";
			std::cerr << "  Input directory specified: " << input_path_str_forward_slash << "\n";
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
inline void process_flag_o(const pathMode& path_mode, const std::string& output_arg, std::filesystem::path& output_path, std::string& output_path_str_forward_slash, bool& dir_created) {
	// Convert output_path to a std::filesystem::path object.
	output_path = std::filesystem::path(output_arg);

	// Check if output_path is a relative path. If it is, convert it to an absolute path.
	output_path = std::filesystem::absolute(output_path);

	output_path_str_forward_slash = output_path.string();
	std::replace(output_path_str_forward_slash.begin(), output_path_str_forward_slash.end(), '\\', '/');

	if (path_mode == file) {
		// file mode code

		// Check if file exists. If it does, notify the user that it will be overwritten.
		if (std::filesystem::is_regular_file(output_path)) {
			std::cout << "WARNING: Output file already exists. It will be overwritten.\n";
			std::cout << "  Output file specified: " << output_path_str_forward_slash << "\n";
		}

		// Check if outputFile is a .mat file. If not, print an error message with full path and exit.
		if (output_path.extension() != ".mat") {
			std::cerr << "Output file must be .mat file\n";
			std::cerr << "  Output file specified: " << output_path_str_forward_slash << "\n";
			std::exit(1);
		}
	}
	else {
		// directory mode code

		// Check if output_path string contains a ".". If so, print an error message with full path and exit.
		if (output_path_str_forward_slash.find(".") != std::string::npos) {
			std::cerr << "Output must be a directory when path-mode is directory\n";
			std::cerr << "  Output file specified: " << output_path_str_forward_slash << "\n";
			std::exit(1);
		}

		if (std::filesystem::exists(output_path) && !std::filesystem::is_directory(output_path)) {
			// output_path exists, but is not a directory. Maybe it's a file that has no extension?
			std::cerr << "Output specified is unknown (maybe it is a file with no extension?)\n";
			std::cerr << "  Output specified: " << output_path_str_forward_slash << "\n";
			std::exit(1);
		}

		// Check if directory exists. If not, create it. If directory cannot be created (maybe user lacks permission?), print an error message with full path and exit.
		if (!std::filesystem::exists(output_path)) {
			// Create directory at output_path
			dir_created = std::filesystem::create_directory(output_path);

			// Check if the directory was created successfully
			if (dir_created) {
				std::cout << "New directory created: " << output_path_str_forward_slash << std::endl;
			}
			else {
				std::cerr << "Failed to create directory: " << output_path_str_forward_slash << std::endl;
				std::exit(1);
			}
		}
	}
}

/**
 * @param argc The number of command line arguments.
 * @param argv An array of C-style strings containing the command line arguments.
 * @return A tuple (pathType path_mode, filesystem::path input_path, filesystem::path output_path)
 */
inline std::tuple<pathMode, std::filesystem::path, std::filesystem::path> getOptions(int argc, char* argv[]) {
	// Assume we are working with individual files by default, unless the user specifies -d flag (directory mode)
	pathMode path_mode = file;

	std::string input_arg = "";
	std::filesystem::path input_path(".");
	std::string input_path_str_forward_slash = "";
	bool input_specified = false;
	bool input_dir_contains_SKV = false;

	std::string output_arg = "";
	std::filesystem::path output_path(".");
	std::string output_path_str_forward_slash = "";
	bool output_specified = false;
	bool dir_created = false;

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
				std::cerr << "Input arg (-i) was specified more than once.\n";
				std::exit(1);
			}
			input_arg = optarg;

			break;

		case 'o':
			if (!output_specified) {
				output_specified = true;
			}
			else {
				std::cerr << "Output arg (-o) was specified more than once.\n";
				std::exit(1);
			}
			output_arg = optarg;

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
		std::cerr << "No input file/path specified\n";
		std::exit(1);
	}

	if (!output_specified) {
		std::cerr << "No output file/path specified\n";
		std::exit(1);
	}

	process_flag_i(path_mode, input_arg, input_path, input_path_str_forward_slash, input_dir_contains_SKV);
	process_flag_o(path_mode, output_arg, output_path, output_path_str_forward_slash, dir_created);

	if (path_mode == file) {
		std::cout << "Input file: " << input_path_str_forward_slash << "\n";
		std::cout << "Output file: " << output_path_str_forward_slash << "\n";
	}
	else {
		std::cout << "Input directory: " << input_path_str_forward_slash << "\n";
		std::cout << "Output directory: " << output_path_str_forward_slash << "\n";
	}

	return std::make_tuple(path_mode, input_path, output_path);
}



#endif /* MAIN_H */
