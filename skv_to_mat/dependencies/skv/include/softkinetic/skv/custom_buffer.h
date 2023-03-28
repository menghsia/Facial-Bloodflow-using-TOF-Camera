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

#if !defined(SOFTKINETIC_SKV_CUSTOM_BUFFER_INC)
#define SOFTKINETIC_SKV_CUSTOM_BUFFER_INC

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
/**
 * \class custom_buffer custom_buffer.h <softkinetic/skv/custom_buffer.h>
 * \brief Encapsulates the operations on a custom buffer.
 */
class custom_buffer
{
public:

	// \cond DOXYGEN_SHOULD_SKIP_THIS
	custom_buffer(skv_handle* handle, const std::string& name, std::uint32_t id)
		: handle(handle),
		  id(id),
		  name(name),
		  compression(skv_compression_none),
		  byte_count(0)
	{
		skv_get_custom_buffer_byte_count(handle, name.c_str(), &byte_count, throw_on_error());
		skv_get_custom_buffer_compression(handle, name.c_str(), &compression, throw_on_error());
	}

	custom_buffer(const custom_buffer& other) :
		handle(other.handle),
		id(other.id),
		name(other.name),
		compression(other.compression),
		byte_count(other.byte_count)
	{
	}

	custom_buffer(custom_buffer&& other) :
		handle(other.handle),
		id(other.id),
		name(other.name),
		compression(other.compression),
		byte_count(other.byte_count)
	{
	}

	custom_buffer& operator=(const stream&) = delete;
	custom_buffer& operator=(stream&&) = delete;
	// \endcond
	
	/**
	 * \brief Destructor.
	 */
	~custom_buffer() = default;

	/**
	 * \brief Gets the custom buffer ID.
	 *
	 * \returns The custom buffer ID.
	 */
	std::uint32_t get_id() const { return id; }

	/**
	 * \brief Gets the custom buffer name.
	 *
	 * \returns The custom buffer name.
	 */
	const std::string& get_name() const { return name; }

	/**
	 * \brief Gets the custom buffer compression type.
	 *
	 * \returns The compression type.
	 */
	skv_compression get_compression() const { return compression; }

	/**
	 * \brief Gets the number of bytes in the custom buffer.
	 *
	 * \returns The custom buffer's byte count.
	 */
	std::size_t get_byte_count() const { return byte_count; }

	/**
	 * \brief Copies the custom buffer data into a buffer at the given address.
	 *
	 * \param[out] raw_pointer The address of a contiguous data buffer, already allocated, to which to copy the custom
	 *             buffer data.
	 * \throws skv_exception
	 * - There was an internal error
	 */
	void get_data(void* raw_pointer) const
	{
		skv_get_custom_buffer_data(handle, name.c_str(), raw_pointer, throw_on_error());
	}

	/**
	 * \brief Copeis the custom buffer data into a buffer.
	 *
	 * \tparam ContiguousData A type that represents a contiguous block of data.
	 * \param[out] buffer The data buffer, already allocated, to which to copy the frame data.
	 * \throws skv_exception
	 * - There was an internal error
	 */
	template <class ContiguousData>
	void get_data(ContiguousData& buffer) const
	{
		assert(byte_count == softkinetic::skv::get_byte_count(buffer));
		get_data(get_raw_pointer(buffer));
	}

	/**
	 * \brief Removes the custom buffer.
	 * \warning Don't use a custom buffer object after removal.
	 * All queries or operations on a removed custom buffer have undefined behavior.
	 *
	 * \throws skv_exception
	 * - The custom buffer doesn't exist
	 * - Can't modify a file recorded using the legacy format
	 * - There was an internal error
	 */
	void remove()
	{
		skv_remove_custom_buffer(handle, name.c_str(), throw_on_error());

		handle = nullptr;
		name = "__REMOVED__";
		compression = skv_compression_none;
		byte_count = 0;
	}

	/**
	 * \brief Renames the custom buffer.
	 *
	 * \param[in] new_name The new name of the custom buffer.
	 * \throws skv_exception
	 * - Another custom buffer already exists with the new name
	 * - Can't modify a file recorded using the legacy format
	 * - There was an internal error
	 */
	void rename(const std::string& new_name)
	{
		skv_rename_custom_buffer(handle, name.c_str(), new_name.c_str(), throw_on_error());

		name = new_name;
	}

	/**
	 * \brief Modifies the data of a custom buffer.
	 *
 	 * \warning The buffer byte count must be the same as what is in the file. Providing data with a smaller or
 	 * larger buffer byte count will result in unexpected behavior.
 	 *
	 * \param[in] raw_pointer The address of the contiguous block of data to save
	 * \param[in] buffer_byte_count The number of bytes to write.
	 * \throws skv_exception
	 * - Can't modify a file recorded using the legacy format
	 * - There was an internal error
	 */
	void modify(const void* raw_pointer, std::size_t buffer_byte_count)
	{
		skv_modify_custom_buffer(handle, name.c_str(), raw_pointer, buffer_byte_count, throw_on_error());
	}

	/**
	 * \brief Modifies the data of a custom buffer.
	 *
 	 * \warning The byte count must be the same as what is in the file. Providing data with a smaller or
 	 * larger buffer size will result in unexpected behavior.
 	 *
	 * \tparam ContiguousData A type that represents a contiguous block of data.
	 * \param[in] buffer A contiguous block of data to save.
	 * \throws skv_exception
	 * - Can't modify a file recorded using the legacy format
	 * - There was an internal error
	 */
	template <class ContiguousData>
	void modify(const ContiguousData& buffer)
	{
		modify(get_raw_pointer(buffer), softkinetic::skv::get_byte_count(buffer));
	}

private:

	skv_handle* handle;
	std::uint32_t id;
	std::string name;
	skv_compression compression;
	std::size_t byte_count;
};

}  // namespace skv
}  // namespace softkinetic
#endif  // SOFTKINETIC_SKV_CUSTOM_BUFFER_INC
