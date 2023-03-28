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

#if !defined(SOFTKINETIC_SKV_STREAM_INC)
#define SOFTKINETIC_SKV_STREAM_INC

#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <type_traits>
#include <softkinetic/skv/core.h>
#include <softkinetic/skv/skv_exception.h>

namespace softkinetic
{
namespace skv
{
namespace details
{
template <class T>
struct is_primitive : public std::integral_constant<bool, std::is_arithmetic<T>::value || std::is_pod<T>::value>
{};
} // namespace details

template <class T>
inline const void* get_raw_pointer(const T& data, typename std::enable_if<details::is_primitive<T>::value>::type* = 0)
{
	return static_cast<const void*>(&data);
}

template <class T>
inline void* get_raw_pointer(T& data, typename std::enable_if<details::is_primitive<T>::value>::type* = 0)
{
	return static_cast<void*>(&data);
}

template <class T>
inline const void* get_raw_pointer(const T* data, typename std::enable_if<details::is_primitive<T>::value>::type* = 0)
{
	return static_cast<const void*>(data);
}

template <class T>
inline void* get_raw_pointer(T* data, typename std::enable_if<details::is_primitive<T>::value>::type* = 0)
{
	return static_cast<void*>(data);
}

template <class T>
inline std::size_t get_byte_count(const T&, typename std::enable_if<details::is_primitive<T>::value>::type* = 0)
{
	return sizeof(T);
}

template <class T, class Allocator>
inline
const void *get_raw_pointer(const std::vector<T, Allocator>& data)
{
	return static_cast<const void *>(data.data());
}

template <class T, class Allocator>
inline
void *get_raw_pointer(std::vector<T, Allocator>& data)
{
	return static_cast<void *>(data.data());
}

template <class T, class Allocator>
inline
typename std::vector<T, Allocator>::size_type get_byte_count(const std::vector<T, Allocator>& data)
{
	return data.size() * sizeof(typename std::vector<T, Allocator>::value_type);
}

/**
 * \class stream stream.h <softkinetic/skv/stream.h>
 * \brief Encapsulates the operations on an image or custom stream.
 */
class stream
{
public:

	// \cond DOXYGEN_SHOULD_SKIP_THIS
	stream(skv_handle* handle, std::uint32_t id)
	    : handle(handle),
	      id(id),
	      image_type(skv_image_type_unknown),
	      width(0),
	      height(0),
	      compression(skv_compression_none)
	{
		skv_get_stream_type(handle, id, &type, throw_on_error());

		if (type == skv_stream_type_custom)
		{
			skv_custom_stream_info info;
			skv_get_custom_stream_info(handle, id, &info, throw_on_error());
			name = info.name;
			compression = info.compression;
		}
		else if (type == skv_stream_type_image)
		{
			skv_image_stream_info info;
			skv_get_image_stream_info(handle, id, &info, throw_on_error());
			name = info.name;
			image_type = info.type;
			width = info.width;
			height = info.height;
			compression = info.compression;
		}
	}

	stream(const stream& other) :
		handle(other.handle),
		id(other.id),
		type(other.type),
		name(other.name),
		image_type(other.image_type),
		width(other.width),
		height(other.height),
		compression(other.compression)
	{
	}

	stream(stream&& other) :
		handle(other.handle),
		id(other.id),
		type(other.type),
		name(other.name),
		image_type(other.image_type),
		width(other.width),
		height(other.height),
		compression(other.compression)
	{
	}

	stream& operator=(const stream&) = delete;
	stream& operator=(stream&&) = delete;
	// \endcond

	/**
	 * \brief Destructor.
	 */
	~stream() = default;

	/**
	 * \brief Gets the stream ID.
	 *
	 * \returns The stream ID.
	 */
	std::uint32_t get_id() const { return id; }

	/**
	 * \brief Gets the stream type i.e. image stream or custom data stream.
	 *
	 * \returns The stream type.
	 */
	skv_stream_type get_type() const { return type; }

	/**
	 * \brief Gets the stream name.
	 *
	 * \returns The stream name.
	 */
	std::string get_name() const
	{
		return name;
	}

	/**
	 * \brief Gets the image stream image type.
	 *
	 * \returns The image type.
	 * \throws skv_exception
	 * - This stream is a custom stream
	 */
	skv_image_type get_image_type() const
	{
		if (get_type() != skv_stream_type_image)
		{
			throw skv_exception(skv_error_code_custom_stream_does_not_support_field, "A custom stream does not have an image type.");
		}
		return image_type;
	}

	/**
	* \brief Gets the image resolution.
	*
	* \returns The image resolution.
	* \throws skv_exception
	* - This stream is a custom stream.
	*/
	std::tuple<std::uint32_t, std::uint32_t> get_resolution() const
	{
		if (get_type() != skv_stream_type_image)
		{
			throw skv_exception(skv_error_code_custom_stream_does_not_support_field, "A custom stream does not have a resolution.");
		}
		return std::make_tuple(width, height);
	}

	/**
	 * \brief Gets the stream compression type.
	 *
	 * \returns The compression type.
	 */
	skv_compression get_compression() const
	{
		return compression;
	}

	/**
	 * \brief Checks if the image stream has a pinhole model.
	 *
	 * \returns \c true if the image stream has a pinhole model, \c false otherwise.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	bool has_pinhole_model() const
	{
		bool has_pinhole_model = false;
		skv_has_pinhole_model(handle, id, &has_pinhole_model, throw_on_error());
		return has_pinhole_model;
	}

	/**
	 * \brief Gets the image stream's pinhole model.
	 *
	 * \returns The pinhole model.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	skv_pinhole_model get_pinhole_model() const
	{
		skv_pinhole_model model;
		skv_get_pinhole_model(handle, id, &model, throw_on_error());
		return model;
	}

	/**
	 * \brief Sets the image stream's pinhole model.
	 *
	 * \param[in] model The pinhole model.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	void set_pinhole_model(const skv_pinhole_model& model)
	{
		skv_set_pinhole_model(handle, id, &model, throw_on_error());
	}

	/**
	 * \brief Checks if the image stream has a distortion model.
	 * 
	 * \returns \c true if the image stream has a distortion model, \c false otherwise.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	bool has_distortion_model() const
	{
		bool has_distortion_model = false;
		skv_has_distortion_model(handle, id, &has_distortion_model, throw_on_error());
		return has_distortion_model;
	}

	/**
	 * \brief Gets the image stream's distortion model.
	 *
	 * \returns The distortion model.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	skv_distortion_model get_distortion_model() const
	{
		skv_distortion_model model;
		skv_get_distortion_model(handle, id, &model, throw_on_error());
		return model;
	}

	/**
	 * \brief Sets the image stream's distortion model.
	 *
	 * \param[in] model The distortion model.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	void set_distortion_model(const skv_distortion_model& model)
	{
		skv_set_distortion_model(handle, id, &model, throw_on_error());
	}

	/**
	 * \brief Checks if the image stream has a stereo transform.
	 *
	 * \returns \c true if the image stream has a stereo transform, \c false otherwise.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	bool has_stereo_transform() const
	{
		bool has_stereo_transform = false;
		skv_has_stereo_transform(handle, id, &has_stereo_transform, throw_on_error());
		return has_stereo_transform;
	}

	/**
	 * \brief Gets the image stream's stereo transform.
	 *
	 * \returns The stereo transform.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	skv_stereo_transform get_stereo_transform() const
	{
		skv_stereo_transform transform;
		skv_get_stereo_transform(handle, id, &transform, throw_on_error());
		return transform;
	}

	/**
	 * \brief Sets the image stream's stereo transform.
	 *
	 * \param[in] transform The stereo transform.
	 * \throws file_error
	 * - Models are only valid for image streams
	 * - There was an internal error
	 */
	void set_stereo_transform(const skv_stereo_transform& transform)
	{
		skv_set_stereo_transform(handle, id, &transform, throw_on_error());
	}

	/**
	 * \brief Gets the number of frames in the stream.
	 *
	 * \returns The frame count.
	 */
	std::uint32_t get_frame_count() const
	{
		std::uint32_t frame_count;
		skv_get_stream_frame_count(handle, id, &frame_count, throw_on_error());

		return frame_count;
	}

	/**
	 * \brief Adds a frame to the stream.
	 *
	 * \param[in] timestamp The timestamp.
	 * \param[in] raw_pointer The address of the contiguous block of data to save.
	 * \param[in] byte_count The number of bytes to write.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - Can't add a frame whose byte_count does not match with the expected size of the image stream.
	 * - There was an internal error
	 */
	void add_frame(std::uint64_t timestamp, const void* raw_pointer, std::size_t byte_count)
	{
		skv_add_frame(handle, id, timestamp, raw_pointer, byte_count, throw_on_error());
	}

	/**
	 * \brief Adds a frame to the stream.
	 *
	 * \tparam ContiguousData A type that represents a contiguous block of data.
	 * \param[in] timestamp The timestamp.
	 * \param[in] buffer A contiguous block of data to save.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - Can't add a frame whose byte_count does not match with the expected size of the image stream.
	 * - There was an internal error
	 */
	template <class ContiguousData>
	void add_frame(std::uint64_t timestamp, const ContiguousData& buffer)
	{
		add_frame(timestamp, get_raw_pointer(buffer), get_byte_count(buffer));
	}

	/**
	 * \brief Adds multiple contiguous, chronologically ordered frames to the stream.
	 * All frames must be chronologically ordered and fit between two existing frames.
	 *
	 * \param[in] timestamps An array of timestamps.
	 * \param[in] raw_pointers An array of addresses of contiguous blocks of data to save.
	 * \param[in] frames_count The number of contiguous blocks of data to save.
	 * \param[in] byte_count The number of bytes to write per contiguous block of data.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - Can't add a frame whose byte_count does not match with the expected size of the image stream.
	 * - The timestamps are not chronological
	 * - The timestamps don't fit between two existing timestamps (i.e. they span over one or more existing frames)
	 * - There was an internal error
	 */
	void add_frames(const std::uint64_t* timestamps, const void* const* raw_pointers, std::size_t frames_count, std::size_t byte_count)
	{
		skv_add_frames(handle, id, timestamps, raw_pointers, frames_count, byte_count, throw_on_error());
	}

	/**
	 * \brief Adds multiple contiguous, chronologically ordered frames to the stream.
	 * All frames must be chronologically ordered and fit between two existing frames.
	 *
	 * \param[in] timestamps A vector of timestamps.
	 * \param[in] buffers A vector of addresses of contiguous blocks of data to save.
	 * \param[in] byte_count The number of bytes to write per contiguous block of data.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - Can't add a frame whose byte_count does not match with the expected size of the image stream.
	 * - The timestamps are not chronological
	 * - The timestamps don't fit between two existing timestamps (i.e. they span over one or more existing frames)
	 * - There was an internal error
	 */
	void add_frames(std::vector<std::uint64_t>& timestamps, const std::vector<const void*>& buffers, std::size_t byte_count)
	{
		add_frames(timestamps.data(), buffers.data(), buffers.size(), byte_count);
	}

	/**
	 * \brief Adds multiple contiguous, chronologically ordered frames to the stream.
	 * All frames must be chronologically ordered and fit between two existing frames.
	 *
	 * \tparam ContiguousData A type that represents a contiguous block of data.
	 * \param[in] timestamps A vector of timestamps.
	 * \param[in] buffers A vector of contiguous blocks of data to save.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - The timestamps are not chronological
	 * - The timestamps don't fit between two existing timestamps (i.e. they span over one or more existing frames)
	 * - There was an internal error
	 */
	template <class ContiguousData>
	void add_frames(std::vector<std::uint64_t>& timestamps, const std::vector<ContiguousData>& buffers)
	{
		typedef typename std::vector<ContiguousData>::const_iterator buffer_iterator;
		std::vector<const void*> buffers_ptr;
		for (buffer_iterator it = buffers.begin(), it_end = buffers.end(); it != it_end; ++it)
			buffers_ptr.push_back(get_raw_pointer(*it));

		add_frames(timestamps, buffers_ptr, get_byte_count(buffers[0]));
	}


	/**
	* \brief Gets the number of bytes in the specified frame.
	*
	* \param[in] frame_index The index of the frame that we want to retrieve.
	* \returns The byte count in the specified frame.
	* \throws file_error
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	std::size_t get_frame_byte_count(std::uint32_t frame_index) const
	{
		std::size_t byte_count = 0;
		skv_get_frame_byte_count(handle, id, frame_index, &byte_count, throw_on_error());
		return byte_count;
	}

	/**
	* \brief Copies the data from the frame specified by frame_index into a buffer at the given address.
	*
	* \param[in] frame_index The index of the frame that we want to retrieve.
	* \param[out] raw_pointer The address of a contiguous data buffer, already allocated, to which to copy the frame
	*             data.
	* \throws file_error
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	void get_frame_data(std::uint32_t frame_index, void* raw_pointer) const
	{
		skv_get_frame_data(handle, id, frame_index, raw_pointer, throw_on_error());
	}

	/**
	* \brief Copies the data from the frame specified by frame_index into a buffer.
	*
	* \tparam ContiguousData A type that represents a contiguous block of data.
	* \param[in] frame_index The index of the frame that we want to retrieve.
	* \param[out] buffer The data buffer, already allocated, to which to copy the frame data.
	* \throws file_error
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	template <class ContiguousData>
	void get_frame_data(std::uint32_t frame_index, ContiguousData& buffer) const
	{
		assert(get_frame_byte_count(frame_index) == get_byte_count(buffer));
		get_frame_data(frame_index, get_raw_pointer(buffer));
	}
	
	/**
	* \brief Gets the number of bytes in the specified frame.
	*
	* \param[in] time_stamp The timestamp of the frame that we want to retrieve.
	* \returns The byte count in the specified frame.
	* \throws file_error
	* - The frame at \c timestamp does not exist
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	std::size_t get_frame_byte_count_by_timestamp(std::uint64_t time_stamp) const
	{
		std::size_t byte_count = 0;
		skv_get_frame_byte_count_by_timestamp(handle, id, time_stamp, &byte_count, throw_on_error());
		return byte_count;
	}

	/**
	* \brief Copies the data from the frame specified by time_stamp into a buffer at the given address.
	*
	* \param[in] time_stamp The timestamp of the frame that we want to retrieve.
	* \param[out] raw_pointer The address of a contiguous data buffer, already allocated, to which to copy the frame
	*             data.
	* \throws file_error
	* - The frame at \c timestamp does not exist
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	void get_frame_data_by_timestamp(std::uint64_t time_stamp, void* raw_pointer) const
	{
		skv_get_frame_data_by_timestamp(handle, id, time_stamp, raw_pointer, throw_on_error());
	}

	/**
	* \brief Copies the data from the frame specified by frame_index into a buffer.
	*
	* \tparam ContiguousData A type that represents a contiguous block of data.
	* \param[in] time_stamp The timestamp of the frame that we want to retrieve.
	* \param[out] buffer The data buffer, already allocated, to which to copy the frame data.
	* \throws file_error
	* - The frame at \c timestamp does not exist
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	template <class ContiguousData>
	void get_frame_data_by_timestamp(std::uint64_t time_stamp, ContiguousData& buffer) const
	{
		assert(get_frame_byte_count_by_timestamp(time_stamp) == get_byte_count(buffer));
		get_frame_data_by_timestamp(time_stamp, get_raw_pointer(buffer));
	}
	
	/**
	* \brief Gets the frame_index of the frame specified by its timestamp.
	*
	* \param[in] time_stamp The timestamp of the frame that we want to retrieve.
	* \returns The index of the specified frame.
	* \throws file_error
	* - The frame at \c timestamp does not exist
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	std::uint32_t get_frame_index(std::uint64_t time_stamp) const
	{
		std::uint32_t frame_index = 0;
		skv_get_frame_index(handle, id, time_stamp, &frame_index, throw_on_error());
		return frame_index;
	}

	/**
	* \brief Gets the timestamp in the frame specified by its frame_index.
	*
	* \param[in] frame_index The index of the frame that we want to retrieve.
	* \returns The timestamp the specified frame.
	* \throws file_error
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	*/
	std::uint64_t get_frame_timestamp(std::uint32_t frame_index) const
	{
		std::uint64_t  timestamp = 0;
		skv_get_frame_timestamp(handle, id, frame_index, &timestamp, throw_on_error());
		return timestamp;
	}

	/**
	 * \brief Removes the stream from the file.
	 * \warning Don't use a stream object after removal.
	 * All queries or operations on a removed stream have undefined behavior.
	 *
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - The stream was already removed
	 * - There was an internal error
	 */
	void remove()
	{
		skv_remove_stream(handle, id, throw_on_error());

		handle = nullptr;
		id = std::numeric_limits<std::uint32_t>::max();
		type = skv_stream_type_unknown;
		name = "___REMOVED___";
		image_type = skv_image_type_unknown;
		width = 0;
		height = 0;
		compression = skv_compression_none;
	}

	/**
	 * \brief Renames the stream.
	 *
	 * \param new_name The new name for the stream.
	 *
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - The new name already exists
	 * - There was an internal error
	 */
	void rename(const std::string& new_name)
	{
		skv_rename_stream(handle, id, new_name.c_str(), throw_on_error());

		name = new_name;
	}

	/**
	 * \brief Modifies the data of a frame.
	 *
 	 * \warning The size of the data buffer must be the same as what is in the file. Providing data with a smaller or
 	 * larger buffer size will result in unexpected behavior.
 	 *
	 * \param[in] frame_index The frame index.
	 * \param[in] raw_pointer The address of the contiguous block of data to save.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 */
	void modify_frame_data(std::uint32_t frame_index, const void* raw_pointer)
	{
		skv_modify_frame_data(handle, id, frame_index, raw_pointer, throw_on_error());
	}

	/**
	 * \brief Modifies the data of a frame.
	 *
 	 * \warning The size of the data buffer must be the same as what is in the file. Providing data with a smaller or
 	 * larger buffer size will result in unexpected behavior.
 	 *
	 * \tparam ContiguousData A type that represents a contiguous block of data.
	 * \param[in] frame_index The frame index.
	 * \param[in] buffer A contiguous block of data to save.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 * - The frame referred to frame_index does not exist
	 */
	template <class ContiguousData>
	void modify_frame_data(std::uint32_t frame_index, const ContiguousData& buffer)
	{
		modify_frame_data(frame_index, get_raw_pointer(buffer));
	}

	/**
	 * \brief Modifies the timestamps of a contiguous set of frames.
	 *
	 * \param[in] frame_index The frame index to change the timestamp for.
	 * \param[in] timestamp A new timestamp.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - the new timestamp is not chronological
	 */
	void modify_timestamps(std::uint32_t frame_index, uint64_t timestamp)
	{
		std::vector<uint64_t> timestamps(1, timestamp);
		modify_timestamps(frame_index, timestamps);
	}

	/**
	 * \brief Modifies the timestamps of a contiguous set of frames.
	 *
	 * \param[in] start_frame_index The first frame index to change the timestamps for.
	 * \param[in] timestamps A vector of new timestamps.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - the new timestamps are not chronological
	 */
	void modify_timestamps(std::uint32_t start_frame_index, std::vector<std::uint64_t>& timestamps)
	{
		std::uint32_t end_frame_index = start_frame_index + static_cast<std::uint32_t>(timestamps.size()) - 1;
		skv_modify_frame_timestamps(handle, id, start_frame_index, end_frame_index, timestamps.data(), throw_on_error());
	}

	/**
	 * \brief Removes a frame from the stream
	 *
	 * \param[in] frame_index_begin The frame index of the first frame to be removed.
	 * \param[in] frame_index_end The frame index of the last frame to be removed.
	 * \throws file_error
	 * - Can't modify a file recorded using the legacy file format
	 * - There was an internal error
	 * - The frame referred to frame_index does not exist
	 */
	void remove_frames(std::uint32_t frame_index_begin, uint32_t frame_index_end)
	{		
		skv_remove_frames(handle, id, frame_index_begin, frame_index_end, throw_on_error());
	}
	
	/**
	* \brief Removes a frame from the stream by timestamp
	*
	* \param[in] timestamp_begin The timestamp of the first frame to be removed.
	* \param[in] timestamp_end The timestamp of the last frame to be removed.
	* \throws file_error
	* - Can't modify a file recorded using the legacy file format
	* - There was an internal error
	* - The frame referred to frame_index does not exist
	* - The timestamp is out of range.
	*/
	void remove_frames_by_timestamp(std::uint64_t timestamp_begin, uint64_t timestamp_end)
	{
		skv_remove_frames_by_timestamp(handle, id, timestamp_begin, timestamp_end, throw_on_error());
	}

private:

	skv_image_stream_info get_image_stream_info() const
	{
		skv_image_stream_info info;
		skv_get_image_stream_info(handle, id, &info, throw_on_error());
		return info;
	}

	skv_custom_stream_info get_custom_stream_info() const
	{
		skv_custom_stream_info info;
		skv_get_custom_stream_info(handle, id, &info, throw_on_error());
		return info;
	}

	skv_handle* handle;
	std::uint32_t id;
	skv_stream_type type;
	std::string name;
	skv_image_type image_type;
	std::uint32_t width, height;
	skv_compression compression;
};
}  // namespace skv
}  // namespace softkinetic

#endif  // SOFTKINETIC_SKV_STREAM_INC
