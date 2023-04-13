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
	unspecified,
	file,
	directory
};

void printHelp(char* argv[]);
std::tuple<std::filesystem::path, std::filesystem::path, pathType> getOptions(int argc, char* argv[]);



/* FUNCTION DEFINITIONS */

void printHelp(char* argv[]) {
	std::cout << "Usage: " << argv[0] << " <-i:input input.skv or input_dir/>|<-o:output output.mat or output_dir/>|[-h]\n";
	std::cout << "This program takes in an input .skv file (or dir of .skv files) and converts it into an output .mat file (or fills a dir with .mat files).\n";
}

/**
 * @brief This function takes in the command line arguments (argc, argv), parses them, and returns a tuple of input and output filepaths.
 * The function uses the getopt_long function to parse the command line arguments and sets the input and output file paths accordingly.
 * Input and output filepaths are both required. If the file cannot be opened (input file does not exist or input/output file have
 * incorrect extension), the function terminates the program with exit code 1.
 *
 * @param argc The number of command line arguments.
 * @param argv An array of C-style strings containing the command line arguments.
 * @return A tuple containing the input filepath and output filepath.
 *         Filepath includes path, filename, and extension.
 *         Example: C:\\path\\to\\file\\sk_automotive_20221003_164605.skv
 * 
 * Example usage:
 * std::tuple<std::filesystem::path, std::filesystem::path> options = getOptions(argc, argv);
 * std::filesystem::path input_path = std::get<0>(options);
 * std::filesystem::path output_path = std::get<1>(options);
 */
std::tuple<std::filesystem::path, std::filesystem::path, pathType> getOptions(int argc, char* argv[]) {
	std::string input_arg;
	std::filesystem::path input_path;
	std::string input_path_str_forward_slash;
	bool input_specified = false;

	std::string output_arg;
	std::filesystem::path output_path;
	std::string output_path_str_forward_slash;
	bool output_specified = false;

	pathType path_mode = unspecified;
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
		{ "help", no_argument, nullptr, 'h' },
		{ nullptr, 0,                 nullptr, '\0' }
	};

	// Fill in the double quotes, to match the mode and help options.
	while ((choice = getopt_long(argc, argv, "i:o:h", long_options, &option_index)) != -1) {
		switch (choice) {
		case 'i':
			input_specified = true;
			input_arg = optarg;

			// Convert input_path to a std::filesystem::path object.
			input_path = std::filesystem::path(input_arg);

			// Check if input_path is a relative path. If it is, convert it to an absolute path.
			input_path = std::filesystem::absolute(input_path);

			/* NOTE: Because I wanted to use std::filesystem to check if files exist, I had
			to upgrade from C++14 to C++20 (filesystem requires C++17 or higher). */

			input_path_str_forward_slash = input_path.string();
			std::replace(input_path_str_forward_slash.begin(), input_path_str_forward_slash.end(), '\\', '/');

			// Check if input_path is a directory or a file.
			if (std::filesystem::is_directory(input_path)) {
				// input_path is a directory.
				path_mode = directory;

				// Check if directory contains any .skv files.
				// If it does not, print an error message and exit.
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
					exit(1);
				}
			}
			else {
				// input_path is a file.
				path_mode = file;
				
				// Check if file exists. If not, print an error message with full path and exit.
				if (!std::filesystem::exists(input_path)) {
					std::cerr << "Input file does not exist\n";
					std::cerr << "  Input file specified: " << input_path_str_forward_slash
						<< "\n";
					exit(1);
				}

				// Check if file is a .skv file. If not, exit.
				if (input_path.extension() != ".skv") {
					std::cerr << "Input file must be .skv file\n";
					std::cerr << "  Input file specified: " << input_path_str_forward_slash
						<< "\n";
					exit(1);
				}
			}

			break;

		case 'o':
			output_specified = true;
			output_arg = optarg;

			// Convert output_path to a std::filesystem::path object.
			output_path = std::filesystem::path(output_arg);

			// Check if output_path is a relative path. If it is, convert it to an absolute path.
			output_path = std::filesystem::absolute(output_path);

			output_path_str_forward_slash = output_path.string();
			std::replace(output_path_str_forward_slash.begin(), output_path_str_forward_slash.end(), '\\', '/');

			// Check if output_path is a directory or a file.
			if (std::filesystem::is_directory(output_path)) {
				// output_path is a directory.
				path_mode = directory;

				// Check if directory is empty.
				if (std::filesystem::is_empty(output_path)) {
					std::cerr << "Output directory is empty\n";
					std::cerr << "  Output directory specified: " << output_path_str_forward_slash
						<< "\n";
					exit(1);
				}
			}
			else {
				// output_path is a file.
				path_mode = file;

				// Check if output_path exists. If it does, notify the user that it will be overwritten.
				if (std::filesystem::exists(output_path)) {
					std::cout << "Output file already exists. It will be overwritten.\n";
					std::cout << "  Output file specified: " << output_path_str_forward_slash
						<< "\n";
				}

				// Check if output_path is a .mat file. If not, exit.
				if (output_path.extension() != ".mat") {
					std::cerr << "Output file must be .mat file\n";
					std::cerr << "  Output file specified: " << output_path_str_forward_slash
						<< "\n";
					exit(1);
				}
			}

			break;

		case 'h':
			printHelp(argv);
			exit(0);

		default:
			std::cerr << "Error: invalid option\n";
			exit(1);
		}
	}

	if (!input_specified) {
		std::cerr << "No input file specified\n";
		exit(1);
	}

	if (!output_specified) {
		std::cerr << "No output file specified\n";
		exit(1);
	}

	return std::make_tuple(input_path, output_path, path_mode);
}



#endif /* MAIN_H */
