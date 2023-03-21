// COPYRIGHT AND CONFIDENTIALITY NOTICE
// SONY DEPTHSENSING SOLUTIONS CONFIDENTIAL INFORMATION
//
// All rights reserved to Sony Depthsensing Solutions SA/NV, a
// company incorporated and existing under the laws of Belgium, with
// its principal place of business at Boulevard de la Plainelaan 11,
// 1050 Brussels (Belgium), registered with the Crossroads bank for
// enterprises under company number 0811 784 189
//
// This file is part of PROJECT-NAME, which is proprietary
// and confidential information of Sony Depthsensing Solutions SA/NV.


// Copyright (c) 2018 Sony Depthsensing Solutions SA/NV

#if !defined(SOFTKINETIC_SKV_FILE_INC)
#define SOFTKINETIC_SKV_FILE_INC

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <tuple>
#include <cassert>
#include <softkinetic/skv/core.h>
#include <softkinetic/skv/skv_exception.h>
#include <softkinetic/skv/stream.h>
#include <softkinetic/skv/custom_buffer.h>

/**
 * \namespace softkinetic Root namespace.
 */
namespace softkinetic
{
/**
 * \namespace skv SKV namespace.
 */
namespace skv
{
class ref_count
{
public:
	ref_count() : count(1) {}
	void increase() { ++count; }
	unsigned int release() { --count; return count; }

private:
	unsigned int count;
};

/**
 * \class file file.h <softkinetic/skv/file.h>
 * \brief This class allows reading and writing streams and custom data to SKV files.
 */
class file
{
private:

	friend file create_file(const std::string&);
	friend file open_file(const std::string&, skv_file_mode mode);

    file() : handle(nullptr), ref(new ref_count()) {}

	void close_file()
	{
		if (handle)
		{
			bool is_open = false;
			auto ec = skv_is_open(handle, &is_open, nullptr);
			assert(ec == skv_error_code_success);
			_unused(ec);
			if (is_open)
			{
				skv_close_file(handle);
			}
			handle = nullptr;
		}
	}

public:

	file(const file& other) :
		handle(other.handle),		
		streams(other.streams),
		custom_buffers(other.custom_buffers),
		ref(other.ref)
	{
		ref->increase();
	}

	file& operator = (const file& other)
	{
		if (ref->release() == 0)
		{
			delete ref;
			ref = nullptr;
			close_file();
		}

		handle = other.handle;
		ref = other.ref;
		ref->increase();
		return *this;
	}

	/**
	 * \brief Move constructor.
	 *
	 * \param[in] other The other file object.
	 */
	file(file&& other) : 
		handle(std::move(other.handle)),
		streams(std::move(other.streams)),
		custom_buffers(std::move(other.custom_buffers)),
		ref(std::move(other.ref))
	{
		other.handle = nullptr;
		other.ref = nullptr;
	}

	/**
	 * \brief Move assignment operator.
	 *
	 * \param[in] other The other file object.
	 */
	file& operator=(file&& other)
	{
		file tmp(std::move(other));
		swap(tmp);
		return *this;
	}

	/**
	 * \brief Destructor.
	 */
	~file()
	{
		if (ref && ref->release() == 0)
		{
			delete ref;
			ref = nullptr;
			close_file();
		}
	}

	void swap(file& other)
	{
		std::swap(handle, other.handle);
		std::swap(streams, other.streams);
		std::swap(custom_buffers, other.custom_buffers);
		std::swap(ref, other.ref);
	}

	/**
	 * \brief Gets the underlying handle.
	 *
	 * \returns A pointer to the underlying SKV handle.
	 */
	skv_handle* get_handle()
	{
		return handle;
	}

	/**
	 * \brief Gets the format version of this file.
	 *
	 * \returns The version containing the major, minor, and patch values.
	 * \throws skv_exception
	 * - There was an internal error
	 */
	std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> get_format_version() const
	{
		std::uint32_t major = 0;
		std::uint32_t minor = 0;
		std::uint32_t patch = 0;
		skv_get_format_version(handle, &major, &minor, &patch, throw_on_error());
		return std::make_tuple(major, minor, patch);
	}
	
	/**
	* \brief Tests if the file uses the legacy internal format (SKF).
	*
	* \returns \c true if the file was recorded using the legacy format, \c false otherwise.
	* \throws file_error
	* - There was an internal error
	*/
	bool is_legacy_format() const
	{
		bool is_legacy_format = false;
		skv_is_legacy_format(handle, &is_legacy_format, throw_on_error());
		return is_legacy_format;
	}

	/**
	 * \brief Tests if the file contains a device info structure.
	 *
	 * \returns \c true if the file contains a device info, \c false otherwise.
	 * \throws skv_exception
	 * - There was an internal error
	 */
	bool has_device_info() const
	{
		bool has_device_info = false;
		skv_has_device_info(handle, &has_device_info, throw_on_error());
		return has_device_info;
	}

	/**
	 * \brief Sets the vendor name and camera model for this SKV file.
	 *
	 * \param[in] vendor_name The vendor name.
	 * \param[in] camera_model The camera model.
	 * \throws skv_exception
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 */
	void set_device_info(const std::string& vendor_name, const std::string& camera_model)
	{
		skv_device_info info;
		skv_assign_device_info(&info, vendor_name.c_str(), camera_model.c_str());
		skv_set_device_info(handle, &info, throw_on_error());
	}

	/**
	 * \brief Gets the vendor name and camera model for this SKV file.
	 *
	 * \returns The vendor name and camera model.
	 * \throws skv_exception
	 * - The device info wasn't set
	 * - There was an internal error
	 */
	std::tuple<std::string, std::string> get_device_info() const
	{
		skv_device_info info;
		skv_get_device_info(handle, &info, throw_on_error());
		return std::make_tuple(info.vendor_name, info.camera_model);
	}

	/**
	 * \brief Gets the total number of image and custom streams in this file.
	 *
	 * \returns The total number of streams.
	 * \throws skv_exception
	 * - There was an internal error
	 */
	std::uint32_t get_stream_count() const
	{
		std::uint32_t stream_count = 0;
		skv_get_stream_count(handle, &stream_count, throw_on_error());
		return stream_count;
	}

	/**
	 * \brief Gets the stream with the given ID.
	 *
	 * \param[in] id The stream ID.
	 * \returns A reference to the stream.
	 * \throws skv_exception
	 * - The stream doesn't exist
	 * - There was an internal error
	 */
	stream& get_stream_by_id(std::uint32_t id) const
	{
		auto it = std::find_if(
			std::begin(streams), std::end(streams),
			[id](const std::shared_ptr<stream>& s) { return s->get_id() == id;  });
		if (it == std::end(streams))
		{
			throw skv_exception(skv_error_code_stream_does_not_exist, "The file doesn't have a stream with that ID.");
		}

		return **it;
	}

	/**
	* \brief Gets the stream with the given name.
	*
	* \param[in] name The stream name.
	* \returns A reference to the stream.
	* \throws skv_exception
	* - The stream doesn't exist
	* - There was an internal error
	*/
	stream& get_stream_by_name(const std::string& name) const
	{
		auto it = std::find_if(
			std::begin(streams), std::end(streams),
			[name](const std::shared_ptr<stream>& s) { return s->get_name() == name;  });
		if (it == std::end(streams))
		{
			throw skv_exception(skv_error_code_stream_does_not_exist, "The file doesn't have a stream with that name.");
		}

		return **it;
	}

	/**
	* \brief Gets the list of names of the streams.
	*
	* \returns A list containing the names of the streams.
	* \throws skv_exception
	* - There was an internal error
	*/
	std::vector<std::string> get_stream_names() const
	{
		std::vector<std::string> stream_names;		
		for (auto const& s: streams)
		{
			stream_names.push_back(s->get_name());		
		}

		return stream_names;
	}
	
	/**
	* \brief Gets the list of names of the custom buffers.
	*
	* \returns A list containing the names of the custom buffers.
	* \throws skv_exception
	* - There was an internal error
	*/
	std::vector<std::string> get_custom_buffer_names() const
	{
		std::vector<std::string> custom_buffer_names;
		for (auto const& cb : custom_buffers)
		{
			custom_buffer_names.push_back(cb->get_name());
		}

		return custom_buffer_names;
	}

	/**
	 * \brief Adds an image stream to the file.
	 *
	 * \param[in] name The name of the image stream.
	 * \param[in] type The type of image to save.
	 * \param[in] resolution The image resolution.
	 * \param[in] compression The type of compression to use.
	 * \returns A reference to the stream.
	 * \throws skv_exception
	 * - The stream already exists
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 */
	stream& add_image_stream(const std::string& name, skv_image_type type, std::tuple<std::uint32_t, std::uint32_t> resolution,
	                         skv_compression compression = skv_compression_none)
	{
		skv_image_stream_info info;
		std::uint32_t id = 0;
		std::uint32_t width = 0, height = 0;
		std::tie(width, height) = resolution;
		
		skv_assign_image_stream_info(&info, name.c_str(), type, compression, width, height);
		skv_add_image_stream(handle, &info, &id, throw_on_error());

		streams.emplace_back(new stream(handle, id));
		return *streams.back();
	}

	/**
	 * \brief Add a stream of custom data to the file.
	 *
	 * \param[in] name The name of the stream.
	 * \param[in] compression The type of compression to use.
     * \param[in] frame_size The size of the frame in bytes.
	 * \returns A reference to the stream.
	 * \throws skv_exception
	 * - The stream already exists
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 */
    stream& add_custom_stream(const std::string& name, const size_t frame_size, skv_compression compression = skv_compression_none)
	{
		skv_custom_stream_info info;
		std::uint32_t id = 0;

		skv_assign_custom_stream_info(&info, name.c_str(), compression, frame_size);
		skv_add_custom_stream(handle, &info, &id, throw_on_error());

		streams.emplace_back(new stream(handle, id));
		return *streams.back();
	}

	/**
	 * \brief Adds a custom data buffer to the file.
	 *
	 * \param[in] name The name of the custom data.
	 * \param[in] raw_pointer The address of the contiguous block of data to save.
	 * \param[in] byte_count The number of bytes to write.
	 * \param[in] compression The type of compression to use.
	 * \throws skv_exception
	 * - The buffer already exists
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 */
	custom_buffer& add_custom_buffer(const std::string& name, const void* raw_pointer, std::size_t byte_count,
	                       skv_compression compression = skv_compression_none)
	{
		std::uint32_t id = get_custom_buffer_count();

		skv_add_custom_buffer(handle, name.c_str(), raw_pointer, byte_count, compression, throw_on_error());

		custom_buffers.emplace_back(new custom_buffer(handle, name, id));
		return *custom_buffers.back();
	}

	/**
	 * \brief Adds a custom data buffer to the file.
	 *
	 * \tparam ContiguousData A type that represents a contiguous block of data.
	 * \param[in] name The name of the custom data.
	 * \param[in] data A contiguous block of data to save.
	 * \param[in] compression The type of compression to use.
	 * \throws skv_exception
	 * - The buffer already exists
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 */
	template <class ContiguousData>
	custom_buffer& add_custom_buffer(const std::string& name, const ContiguousData& data,
	                       skv_compression compression = skv_compression_none)
	{
		return add_custom_buffer(name, get_raw_pointer(data), get_byte_count(data), compression);
	}

	/**
	 * \brief Gets the total number of custom buffers in this file.
	 *
	 * \returns The total number of custom buffers.
	 * \throws skv_exception
	 * - There was an internal error
	 */
	std::uint32_t get_custom_buffer_count() const
	{
		std::uint32_t custom_buffer_count = 0;
		skv_get_custom_buffer_count(handle, &custom_buffer_count, throw_on_error());
		return custom_buffer_count;
	}

	/**
	 * \brief Gets the custom buffer with the given name.
	 *
	 * \param[in] name The custom buffer name.
	 * \returns A reference to the custom buffer.
	 * \throws skv_exception
	 * - The buffer doesn't exist
	 * - There was an internal error
	 */
	custom_buffer& get_custom_buffer_by_name(const std::string& name) const
	{
		auto it = std::find_if(
			std::begin(custom_buffers), std::end(custom_buffers),
			[name](const std::shared_ptr<custom_buffer>& b) { return b->get_name() == name;  });
		if (it == std::end(custom_buffers))
		{
			throw skv_exception(skv_error_code_custom_buffer_does_not_exist, "The file doesn't have a custom buffer with that name.");
		}

		return **it;
	}

	/**
	 * \brief Gets the custom buffer with the given ID.
	 *
	 * \param[in] id The custom buffer ID.
	 * \returns A reference to the custom buffer.
	 * \throws skv_exception
	 * - The buffer doesn't exist
	 * - There was an internal error
	 */
	custom_buffer& get_custom_buffer_by_id(std::uint32_t id) const
	{
		auto it = std::find_if(
				std::begin(custom_buffers), std::end(custom_buffers),
				[id](const std::shared_ptr<custom_buffer>& b) { return b->get_id() == id; });
		if (it == std::end(custom_buffers))
		{
			throw skv_exception(skv_error_code_custom_buffer_does_not_exist, "The file doesn't have a custom buffer with that ID.");
		}

		return **it;
	}

	/**
	 * \brief Seeks to the next frame in the file, regardless of the stream.
	 *
	 * \returns true if a new frame is available, false otherwise (typically at end-of-file)
	 * \throws skv_exception
	 * - The stream doesn't exist, or there are no streams
	 * - There was an internal error
	 */
	bool seek_next_frame()
	{
		try
		{
			skv_seek_next_frame(handle, throw_on_error());
		}
		catch (skv_exception& e)
		{
			if (e.get_error_code() == skv_error_code_frame_does_not_exist)
				return false;

			throw;
		}

		return true;
	}

	/**
	 * \brief Seeks to a frame in the specified stream.
	 *
	 * \param[in] stream_id The stream id.
	 * \param[in] index The frame index.
	 * \throws skv_exception
	 * - The stream with given id does not exist
	 * - The frame at \c index does not exist
	 * - There was an internal error
	 */
	void seek_frame_by_index(std::uint32_t stream_id, std::uint32_t index)
	{
		skv_seek_frame_by_index(handle, stream_id, index, throw_on_error());
	}

	/**
	 * \brief Seeks to a frame in the specified stream.
	 *
	 * \param[in] stream_id The stream id.
	 * \param[in] timestamp The frame timestamp.
	 * \throws skv_exception
	 * - The stream with given id does not exist
	 * - The frame at \c timestamp does not exist
	 * - There was an internal error
	 */
	void seek_frame_by_timestamp(std::uint32_t stream_id, std::uint64_t timestamp)
	{
		skv_seek_frame_by_timestamp(handle, stream_id, timestamp, throw_on_error());
	}

	/**
	* \brief Gets the current frame timestamp.
	*
	* \returns The current frame timestamp.
	* \throws skv_exception
	* - The seek function was not called prior to this function
	* - There was an internal error
	*/
	std::uint32_t get_current_frame_index() const
	{
		std::uint32_t index = 0;
		skv_get_current_frame_index(handle, &index, throw_on_error());
		return index;
	}

	/**
	* \brief Gets the current frame index.
	*
	* \returns The current frame index.
	* \throws skv_exception
	* - The seek function was not called prior to this function
	* - There was an internal error
	*/
	std::uint64_t get_current_frame_timestamp() const
	{
		std::uint64_t timestamp = 0;
		skv_get_current_frame_timestamp(handle, &timestamp, throw_on_error());
		return timestamp;
	}

	/**
	* \brief Gets the number of bytes in the current frame.
	*
	* \returns The current frame byte count.
	* \throws skv_exception
	* - The seek function was not called prior to this function
	* - There was an internal error
	*/
	std::size_t get_current_frame_byte_count() const
	{
		std::size_t byte_count = 0;
		skv_get_current_frame_byte_count(handle, &byte_count, throw_on_error());
		return byte_count;
	}

	/**
	* \brief Copies the current frame into a buffer at the given address.
	*
	* \param[out] raw_pointer The address of a contiguous data buffer, already allocated, to which to copy the frame
	*             data.
	* \throws skv_exception
	* - The seek function was not called prior to this function
	* - There was an internal error
	*/
	void get_current_frame_data(void* raw_pointer) const
	{
		skv_get_current_frame_data(handle, raw_pointer, throw_on_error());
	}

	/**
	* \brief Copies the current frame into a buffer.
	*
	* \tparam ContiguousData A type that represents a contiguous block of data.
	* \param[out] buffer The data buffer, already allocated, to which to copy the frame data.
	* \throws skv_exception
	* - The seek function was not called prior to this function
	* - There was an internal error
	*/
	template <class ContiguousData>
	void get_current_frame_data(ContiguousData& buffer) const
	{
		assert(get_current_frame_byte_count() == get_byte_count(buffer));
		get_current_frame_data(get_raw_pointer(buffer));
	}

	/**
	 * \brief Gets the stream for the current frame.
	 *
	 * \returns A reference to the stream.
	 * \throws skv_exception
	 * - The seek function was not called prior to this function
	 * - There was an internal error
	 */
	stream& get_stream_for_current_frame()
	{
		std::uint32_t stream_id = 0;
		skv_get_current_frame_stream_id(handle, &stream_id, throw_on_error());

		return get_stream_by_id(stream_id);
	}

private:
	skv_handle* handle;
	mutable std::vector<std::shared_ptr<stream>> streams;
	mutable std::vector<std::shared_ptr<custom_buffer>> custom_buffers;

	ref_count* ref;
};

/**
 * \brief Tests that a file exists.
 *
 * \param[in] file_name The file name.
 * \returns \c true if the file exists, \c false otherwise.
 */
inline bool file_exists(const std::string& file_name)
{
	return skv_file_exists(file_name.c_str());
}

/**
* \brief Creates a new SKV file.
*
* \param file_name The name of the SKV file.
* \throws skv_exception
* - The file name was not valid
* - The file already exists
* - The file doesn't exist and could not be created
* - There was an internal error
*/
inline file create_file(const std::string& file_name)
{
	file f;
	skv_create_file(&(f.handle), file_name.c_str(), throw_on_error());
	return f;
}

/**
* \brief Opens an existing SKV file.
*
* \param file_name The name of the SKV file.
* \param mode The mode in which the file will be open (r/w or read only).
* \throws skv_exception
* - The file name was not valid
* - The file doesn't exist
* - The file was not a valid SKV file
* - There was an internal error
*/
inline file open_file(const std::string& file_name, skv_file_mode mode)
{
	file f;
    skv_open_file(&(f.handle), file_name.c_str(), mode, throw_on_error());

	std::uint32_t stream_count = f.get_stream_count();
	for (std::uint32_t i = 0; i < stream_count; ++i)
	{
		f.streams.emplace_back(std::shared_ptr<stream>(new stream(f.handle, i)));
	}

	std::uint32_t custom_buffer_count = f.get_custom_buffer_count();
	for (std::uint32_t i = 0; i < custom_buffer_count; ++i)
	{
		char name[256];
		skv_get_custom_buffer_name(f.get_handle(), i, name, 255, throw_on_error());
		f.custom_buffers.emplace_back(std::shared_ptr<custom_buffer>(new custom_buffer(f.handle, name, i)));
	}


	return f;
}

inline std::tuple<std::uint32_t, std::uint32_t> resolution(std::uint32_t width, std::uint32_t height)
{
	return std::make_tuple(width, height);
}
} // namespace skv
} // namespace softkinetic

#endif // SOFTKINETIC_SKV_FILE_INC
