#pragma once


#include <iostream>
#include <string>
#include <mutex>

#pragma warning( push )
#pragma warning( disable : 4458)
#include "softkinetic/skv.h"
#pragma warning( pop )

namespace depthsense
{
namespace skv
{
namespace helper
{
struct camera_intrinsics
{
	size_t width;
	size_t height;
	
	float central_x;
	float central_y;
	
	float focal_x;
	float focal_y;
	
	float k1;
	float k2;
	float k3;
	float p1;
	float p2;
};

class skv 
{
public:
	bool open(const std::string& path) 
	{
		// Open file
		skv_error error;
		if (skv_open_file(&handle, path.c_str(), skv_read_only, &error) != skv_error_code_success)
		{
			std::cout << "[skv] Failed to open file (" << error.message << ")" << std::endl;
			return false;
		}

		// Find depth and confidence streams
		uint32_t stream_count;
		skv_get_stream_count(handle, &stream_count, nullptr);
		for (uint32_t stream_id = 0; stream_id < stream_count; ++stream_id)
		{
			skv_stream_type stream_type;
			skv_get_stream_type(handle, stream_id, &stream_type, nullptr);
			if (stream_type != skv_stream_type_image)
				continue;

			skv_image_stream_info info;
			skv_get_image_stream_info(handle, stream_id, &info, nullptr);

			if (std::string(info.name) == std::string("depth_0") || std::string(info.name) == std::string("Depth stream"))
				depth_stream_id = stream_id;

			if (std::string(info.name) == std::string("confidence_0") || std::string(info.name) == std::string("Confidence stream"))
				confidence_stream_id = stream_id;
		}

		skv_get_stream_frame_count(handle, depth_stream_id, &frame_count, nullptr);

		uint64_t first_timestamp, last_timestamp;
		skv_get_frame_timestamp(handle, depth_stream_id, 0, &first_timestamp, nullptr);
		skv_get_frame_timestamp(handle, depth_stream_id, static_cast<uint32_t>(get_frame_count() - 1), &last_timestamp, nullptr);
		average_fps = static_cast<float>(get_frame_count() - 1) / static_cast<float>(last_timestamp - first_timestamp) * 1000.f * 1000.f;

		filepath = path;

		return true;
	}

	void close() 
	{
		if (handle)
		{
			skv_close_file(handle);
		}
		depth_stream_id = static_cast<uint32_t>(-1);
		confidence_stream_id = static_cast<uint32_t>(-1);
	}

	camera_intrinsics get_camera_intrinsic_parameters() const 
	{
		camera_intrinsics parameters;

		skv_image_stream_info info;
		skv_get_image_stream_info(handle, depth_stream_id, &info, nullptr);
		parameters.width = info.width;
		parameters.height = info.height;

		bool has_pinhole_model;
		skv_has_pinhole_model(handle, depth_stream_id, &has_pinhole_model, nullptr);
		if (has_pinhole_model)
		{
			skv_pinhole_model model;
			skv_get_pinhole_model(handle, depth_stream_id, &model, nullptr);

			if (model.cx <= 1.f && model.cy <= 1.f)
			{
				parameters.central_x = model.cx * (float)parameters.width;
				parameters.central_y = model.cy * (float)parameters.height;
			}
			else
			{
				parameters.central_x = model.cx;
				parameters.central_y = model.cy;
			}
		}

		bool has_distortion_model;
		skv_has_distortion_model(handle, depth_stream_id, &has_distortion_model, nullptr);
		if (has_distortion_model)
		{
			skv_distortion_model model;
			skv_get_distortion_model(handle, depth_stream_id, &model, nullptr);

			if (model.fx <= 1.f && model.fy <= 1.f)
			{
				parameters.focal_x = model.fx * (float)parameters.width;
				parameters.focal_y = model.fy * (float)parameters.height;
			}
			else
			{
				parameters.focal_x = model.fx;
				parameters.focal_y = model.fy;
			}
			parameters.k1 = model.k1;
			parameters.k2 = model.k2;
			parameters.k3 = model.k3;
			parameters.p1 = model.p1;
			parameters.p2 = model.p2;
		}

		return parameters;
	}

	bool is_open() const 
	{
		return handle != nullptr;
	}

	const std::string& get_filepath() const 
	{
		return filepath;
	}

	
	const std::string get_camera_vendor() const 
	{
		bool has_buffer = false;
		skv_has_custom_buffer(handle, "vendor_0", &has_buffer, nullptr);
		if (!has_buffer) return std::string();

		size_t bytecount;
		skv_get_custom_buffer_byte_count(handle, "vendor_0", &bytecount, nullptr);
		char* data = new char[bytecount + 1];
		skv_get_custom_buffer_data(handle, "vendor_0", data, nullptr);
		data[bytecount] = 0;
		std::string result(data);
		delete[] data;
		return result;
	}

	const std::string get_camera_model() const 
	{
		bool has_buffer = false;
		skv_has_custom_buffer(handle, "model_0", &has_buffer, nullptr);
		if (!has_buffer) return std::string();

		size_t bytecount;
		skv_get_custom_buffer_byte_count(handle, "model_0", &bytecount, nullptr);
		char* data = new char[bytecount + 1];
		skv_get_custom_buffer_data(handle, "model_0", data, nullptr);
		data[bytecount] = 0;
		std::string result(data);
		delete[] data;
		return result;
	}

	const std::string get_camera_serial() const 
	{
		bool has_serial_number = false;
		skv_has_custom_buffer(handle, "serial_number_0", &has_serial_number, nullptr);
		if (!has_serial_number) return std::string();

		size_t bytecount;
		skv_get_custom_buffer_byte_count(handle, "serial_number_0", &bytecount, nullptr);
		char *data = new char[bytecount + 1];
		skv_get_custom_buffer_data(handle, "serial_number_0", data, nullptr);
		data[bytecount] = 0;
		std::string serial(data);
		delete[] data;
		return serial;
	}

	int32_t get_driver_software_id() const
	{
		bool has_buffer = false;
		skv_has_custom_buffer(handle, "sid_0", &has_buffer, nullptr);
		if (!has_buffer) return -1;

		int32_t data;
		skv_get_custom_buffer_data(handle, "sid_0", &data, nullptr);
		return data;
	}

	int32_t get_driver_mode_id() const
	{
		bool has_buffer = false;
		skv_has_custom_buffer(handle, "uid_0", &has_buffer, nullptr);
		if (!has_buffer) return -1;

		int32_t data;
		skv_get_custom_buffer_data(handle, "uid_0", &data, nullptr);
		return data;
	}

	const std::string get_recorder_version() const
	{
		bool has_buffer = false;
		skv_has_custom_buffer(handle, "recorder_version", &has_buffer, nullptr);
		if (!has_buffer) return std::string();

		size_t bytecount;
		skv_get_custom_buffer_byte_count(handle, "recorder_version", &bytecount, nullptr);
		char* data = new char[bytecount + 1];
		skv_get_custom_buffer_data(handle, "recorder_version", data, nullptr);
		data[bytecount] = 0;
		std::string result(data);
		delete[] data;
		return result;
	}

	size_t get_frame_count() const 
	{
		return frame_count;
	}

	float get_average_fps() const 
	{
		return average_fps;
	}

	void get_timestamp(size_t index, uint64_t& timestamp) const 
	{
		skv_get_frame_timestamp(handle, depth_stream_id, static_cast<uint32_t>(index), &timestamp, nullptr);
	}

	void get_depth_image(size_t index, int16_t* depth_image) const 
	{
		skv_get_frame_data(handle, depth_stream_id, static_cast<uint32_t>(index), depth_image, nullptr);
	}

	void get_confidence_image(size_t index, int16_t* confidence_image) const 
	{
		uint64_t timestamp;
		get_timestamp(index, timestamp);

		skv_get_frame_data_by_timestamp(handle, confidence_stream_id, timestamp, confidence_image, nullptr);
	}

private:
	skv_handle* handle;
	std::string filepath;
	uint32_t depth_stream_id;
	uint32_t confidence_stream_id;

	float average_fps;
	uint32_t frame_count;
};

} // namespace helper
} // namespace skv
} // namespace depthsense
