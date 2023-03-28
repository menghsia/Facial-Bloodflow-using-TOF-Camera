/***************************************************************************************/
//  SoftKinetic SKV library
//  Project Name      : SKV
//  Module Name	      : SKV API v2 
//  Description       : C API v2 for reading/writing skv files
//
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
/****************************************************************************************/

/**
 * \file core.h
 * \brief Contains the functions for creating, opening, reading, and
 *        writing data to SKV files.
 *
 * The functions contained in this file cover all generic operations
 * for using SKV files. More operations are provided as with the C++
 * or Python bindings, which support more specific image or custom
 * data.
 */

#if !defined(SOFTKINETIC_SKV_CORE_INC)
#define SOFTKINETIC_SKV_CORE_INC

#pragma once

#include <softkinetic/skv/core/platform.h>
#include <softkinetic/skv/core/types.h>
#include <softkinetic/skv/core/version.h>
#include <exception>

#define _unused(x) ((void)(x))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief A type that is used to handle SKV files.
 */
typedef struct skv_handle skv_handle;

/**
 * \brief Enumerates a set of error codes for reading and writing
 *        SKV files.
 */
typedef enum
{
	skv_error_code_success = 0,
	/*!< Indicates that the operation succeeded. */

	skv_error_code_invalid_file_name,
	/*!< Indicates that the file name is not valid. */
	skv_error_code_invalid_file,
	/*!< Indicates that the file could not be interpreted as an SKV file. */
	skv_error_code_file_already_exists,
	/*!< Indicates that the file already exists. */
	skv_error_code_unable_to_create_file,
	/*!< Indicates that the file could not be created. */
	skv_error_code_file_does_not_exist,
	/*!< Indicates that the file could not be opened because it doesn't exist. */

	skv_error_code_file_is_closed,
	/*!< Indicates that the file is not open, and can't be accessed. */

	skv_error_code_cant_read_device_info,
	/*!< Indicates that the device info could not be read. */

	skv_error_code_custom_stream_does_not_support_field,
	/*!< Indicates that custom streams don't support the data. */
	skv_error_code_stream_does_not_exist,
	/*!< Indicates that the stream doesn't exist. */
	skv_error_code_stream_already_exists,
	/*!< Indicates that the stream already exists. */
	skv_error_code_frame_does_not_exist,
	/*!< Indicates that the frame doesn't exist. */
	skv_error_code_seek_not_initiated,
	/*!< Indicates that seeking frames is not initiated. */
	skv_error_code_timestamp_out_of_range,
	/*!< Indicates that the timestamp is out of range. */

	skv_error_code_custom_buffer_does_not_exist,
	/*!< Indicates that the custom buffer doesn't exist. */
	skv_error_code_custom_buffer_already_exists,
	/*!< Indicates that the custom buffer already exists. */
	skv_error_code_byte_count_does_not_match,
	/*!< Indicates that the bytecount of the added frame does not match with the expected size of the image or custom stream. */

	skv_error_code_cannot_modify_readonly_file,
	/*!< Indicates that a call to a function that modifies the file won't succeed when the file was opened in read-only mode. */
	skv_error_code_cannot_modify_legacy_format,
	/*!< Indicates that a call to a function that modifies the file won't succeed when the file uses the SKF format
	internally (e.g. pre-1.5) */
	skv_error_code_compression_method_deprecated,
	/*!< Indicates that the selected compression method is no longer supported for this version. */
	
	skv_error_code_internal_error
	/*!< Indicates an internal file format error. */

} skv_error_code;

/**
 * \brief Holds additional information about errors.
 */
typedef struct
{
	/**
	 * \brief The error code.
	 */
	skv_error_code code;

	/**
	 * \brief A useful feedback message
	 */
	const char* message;
} skv_error;

/**
 * \brief Contains information about the device used to record the data in the SKV file.
 */
typedef struct
{
	/**
	 * \brief The device vendor name.
	 */
	char vendor_name[256];
	/**
	 * \brief The device camera model.
	 */
	char camera_model[256];
} skv_device_info;

/**
 * \brief Enumerates the types of streams.
 */
typedef enum
{
	skv_stream_type_unknown = 0,
	/*!< Indiciates that the stream is not yet known. */
	skv_stream_type_image,
	/*!< Indicates that the stream contains image data. */
	skv_stream_type_custom,
	/*!< Indicates that the stream contains custom data. */
} skv_stream_type;

/**
 * \brief Enumerates the types of image data that are stored inside
 *        image streams.
 */
typedef enum
{
	skv_image_type_unknown = 0,
	/*!< Indicates that the image type is not yet known. */
	skv_image_type_int8 = 1,
	/*!< Indicates that the image is made up of int8 values. */
	skv_image_type_uint8 = 2,
	/*!< Indicates that the image is made up of uint8 values. */
	skv_image_type_int16 = 3,
	/*!< Indicates that the image is made up of int16 values. */
	skv_image_type_uint16 = 4,
	/*!< Indicates that the image is made up of uin16 values. */
	skv_image_type_int32 = 5,
	/*!< Indicates that the image is made up of int32 values. */
	skv_image_type_uint32 = 6,
	/*!< Indicates that the image is made up of uint32 values. */
	skv_image_type_bgr24 = 7,
	/*!< Indicates that the image is made up of bgr24 values. */
	skv_image_type_yuv16 = 8,
	/*!< Indicates that the image is made up of yuv16 values. */
	skv_image_type_float = 9,
	/*!< Indicates that the image is made up of float values. */
	skv_image_type_rgb24 = 10,
	/*!< Indicates that the image is made up of rgb24 values. */
	skv_image_type_bgra32 = 11,
	/*!< Indicates that the image is made up of bgra32 values. */
	skv_image_type_rgba32 = 12,
	/*!< Indicates that the image is made up of rgba32 values. */
	skv_image_type_double = 13
	/*!< Indicates that the image is made up of double values. */
} skv_image_type;

/**
 * \brief Enumerates the types of compression available for data streams and custom buffers.
 */
typedef enum
{
	skv_compression_none = 0,
	/*!< Indicates that the data was not compressed. */
	skv_compression_snappy = 1,
	/*!< Indicates that the data was compressed using Snappy. */
	skv_compression_zlib = 3,
	/*!< Indicates that the data was compressed using ZLib. */
	skv_compression_lz4 = 5
	/*!< Indicates that the data was compressed using LZ4. */
} skv_compression;

/**
* \brief Enumerates the different modes in which to open the files.
*/
typedef enum
{
	/*!< Open a file with read and write capabilities. */
	skv_read_write = 0,
	/*!< Open a file with read only capabilities. */
	skv_read_only = 1
} skv_file_mode;


/**
 * \brief Contains information about a particular image stream.
 */
typedef struct
{
	/**
	 * \brief The image stream name.
	 */
	char name[256];
	/**
	 * \brief The image value type.
	 */
	skv_image_type type;
	/**
	 * \brief The image compression.
	 */
	skv_compression compression;
	/**
	 * \brief The image width in pixels.
	 */
	uint32_t width;
	/**
	 * \brief The image height in pixels.
	 */
	uint32_t height;
} skv_image_stream_info;

/**
 * \brief Contains information identifying custom stream data.
 */
typedef struct
{
	/**
	 * \brief The name of the custom data.
	 */
	char name[256];
	/**
	 * \brief The custom data compression.
	 */
	skv_compression compression;
    /**
    * \brief The size of a custom frame in bytes.
    */
    size_t frame_size;
} skv_custom_stream_info;

/**
 * \brief The camera pinhole model.
 */
typedef struct
{
	/**
	* \name Field-of-view
	* \brief The field-of-view.
	 * @{
	 */
	float fovx;
	float fovy;
	/**
	 * @}
	 */

	/**
	 * \name Central point
	 * \brief The central point is expressed as a ratio of the image width and height on each axis.
	 * @{
	 */
	float cx;
	float cy;
	/**
	 * @}
	 */
} skv_pinhole_model;

/**
* \brief The camera distortion model.
*/
typedef struct
{
	/**
	 * \name Focal length
	 * \brief The focal length.
	 * @{
	 */
	float fx;
	float fy;
	/**
	 * @}
	 */

	/**
	 * \name Radial distortion
	 * \brief The radial distortion co-efficients.
	 */
	float k1;
	float k2;
	float k3;
	float k4;
	/**
	 * @}
	 */

	/**
	 * \name Tangential distortion
	 * \brief The tangential distortion co-efficients.
	 */
	float p1;
	float p2;
	/**
	 * @}
	 */
} skv_distortion_model;

/**
* \brief The stereo transform, including a rotation and translation.
*/
typedef struct
{
	/**
	 * \name Rotation
	 * \brief The 3x3 rotation matrix.
	 * @{
	 */
	float r11;
	float r12;
	float r13;
	float r21;
	float r22;
	float r23;
	float r31;
	float r32;
	float r33;
	/**
	 * @}
	 */

	/**
	 * \name Translation
	 * \brief The 3-dimensional transformation vector.
	 * @{
	 */
	float t1;
	float t2;
	float t3;
	/**
	 * @}
	 */

} skv_stereo_transform;

/**
 * \brief Returns the library (DLL) version as a string in the format
 *        MAJOR.MINOR.PATCH[-STAGE].
 * \returns the library version e.g. "2.0.0"
 */
SKV_API const char* SKV_SDK_DECL skv_library_version();

/**
 * \brief Returns the library major version.
 * \returns The library major version.
 */
SKV_API int SKV_SDK_DECL skv_library_version_major();

/**
* \brief Returns the library minor version.
* \returns The library minor version.
*/
SKV_API int SKV_SDK_DECL skv_library_version_minor();

/**
* \brief Returns the library patch version.
* \returns The library patch version.
*/
SKV_API int SKV_SDK_DECL skv_library_version_patch();

/**
 * \brief Returns an string corresponding to an error code.
 * \param[in] ec The error code value.
 * \returns A short string giving a description of the error.
 */
SKV_API const char* SKV_SDK_DECL skv_error_message(skv_error_code ec);

/**
 * \brief Tests if the file exists.
 *
 * \param[in] file_name The file name.
 * \returns \c true if the file exists, \c false otherwise.
 */
SKV_API bool SKV_SDK_DECL skv_file_exists(const char* file_name);

/**
 * \brief Creates a file and associates it with the uninitialized given file handle,
 *        with the given file name in the current directory. If the file doesn't
 *        exist, it is created.
 *
 * \param[out] handle Address to an uninitialized SKV handle.
 * \param[in] file_name The file name.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_already_exists</tt>: The file already exists.
 * - <tt>skv_error_code_unable_to_create_file</tt>: Unable to create file.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \sa skv_close_file
 */
SKV_API skv_error_code SKV_SDK_DECL skv_create_file(skv_handle** handle, const char* file_name, skv_error* error);

/**
 * \brief Opens a file and associates it with the uninitialized given file handle,
 *        with the given file name in the current directory.
 *
 * \param[out] handle Address to an uninitialized SKV handle.
 * \param[in] file_name The file name.
 * \param[in] mode The mode in which the file will be open (r/w or read only).
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_does_not_exist</tt>: The file does not exist.
 * - <tt>skv_error_code_invalid_file</tt>: The file was not a valid SKV file.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \sa skv_close_file
 */
SKV_API skv_error_code SKV_SDK_DECL skv_open_file(skv_handle** handle, const char* file_name, skv_file_mode mode, skv_error* error);

/**
 * \brief Closes a file and destroys its pre-allocated handle that has been previously been initialized with
 *        skv_open_file or skv_create_file. After closure, the output of skv_is_open will be false.
 *
 * \param[in] handle An SKV handle.
 * \pre before closing <tt>bool is_open = false; skv_is_open(handle, &is_open); assert(is_open);</tt>
 * \post after closing<tt>bool is_open = false; skv_is_open(handle, &is_open); assert(!is_open);</tt>
 */
SKV_API void SKV_SDK_DECL skv_close_file(skv_handle* handle);

/**
 * \brief Tests if the file has been opened with skv_open_file.
 *
 * \param[in] handle An SKV handle.
 * \param[out] is_open Set to \c true if the handle has been opened
 *                     with skv_open_file.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_is_open(skv_handle* handle, bool* is_open, skv_error* error);

/**
 * \brief Tests if the file has been opened in read-only access mode.
 *
 * \param[in] handle An SKV handle.
 * \param[out] is_readonly Set to \c true if the handle has been opened
 *                     with read-only access mode. Set to \c false if the
 *                     handle has been opened with read-write access mode.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_is_readonly(skv_handle* handle, bool* is_readonly, skv_error* error);

/**
 * \brief Gets the version of the library that was used to create the file.
 *
 * \param[in] handle An SKV handle.
 * \param[out] major The SKV major version that was used to write this file.
 * \param[out] minor The SKV minor version that was used to write this file.
 * \param[out] patch The SKV patch version that was used to write this file.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \pre <tt>bool is_open = false; skv_is_open(handle, &is_open); assert(!is_open);</tt>
 * \sa skv_dll_version
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_format_version(skv_handle* handle, uint32_t* major, uint32_t* minor,
                                                           uint32_t* patch, skv_error* error);

/**
* \brief Tests if the file uses the legacy internal format (SKF, all versions prior to 1.5).
*
* \param[in] handle An SKV handle.
* \param[out] is_legacy_format \c true if the file was recorded using the legacy format, \c false otherwise.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
* \pre <tt>bool is_open = false; skv_is_open(handle, &is_open); assert(!is_open);</tt>
*/
SKV_API skv_error_code SKV_SDK_DECL skv_is_legacy_format(skv_handle* handle, bool* is_legacy_format, skv_error* error);

/**
 * \brief Tests if the file contains a device info structure.
 *
 * \param[in] handle An SKV handle.
 * \param[out] has_device_info \c true if the file contains the device info, \c false otherwise.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \pre <tt>bool is_open = false; skv_is_open(handle, &is_open); assert(!is_open);</tt>
 * \sa skv_get_device_info
 * \sa skv_set_device_info
 */
SKV_API skv_error_code SKV_SDK_DECL skv_has_device_info(skv_handle* handle, bool* has_device_info, skv_error* error);

/**
 * \brief Gets the device info from the SKV file.
 *
 * \param[in] handle An SKV handle.
 * \param[out] info The device info.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_cant_read_device_info</tt>: The device info does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \pre <tt>bool is_open = false; skv_is_open(handle, &is_open); assert(!is_open);</tt>
 * \sa skv_has_device_info
 * \sa skv_set_device_info
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_device_info(skv_handle* handle, skv_device_info* info, skv_error* error);

/**
 * \brief Convenience function that sets the vendor name and camera
 *        model fields in the device info.
 *
 * \param[in] info A device info.
 * \param[in] vendor_name The vendor name.
 * \param[in] camera_model The camera model.
 */
SKV_API void SKV_SDK_DECL skv_assign_device_info(skv_device_info* info, const char* vendor_name,
                                                 const char* camera_model);

/**
 * \brief Sets the device info in the SKV file.
 *
 * \param[in] handle An SKV handle.
 * \param[out] info A device info object.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
 * - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \pre <tt>bool is_open = false; skv_is_open(handle, &is_open); assert(!is_open);</tt>
 * \sa skv_has_device_info
 * \sa skv_get_device_info
 */
SKV_API skv_error_code SKV_SDK_DECL skv_set_device_info(skv_handle* handle, const skv_device_info* info, skv_error* error);

/**
 * \brief Gets the number of streams in the SKV file.
 *
 * \param[in] handle An SKV handle.
 * \param[out] stream_count The number of streams.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_stream_count(skv_handle* handle, uint32_t* stream_count, skv_error* error);

/**
* \brief Gets the name of a stream in the SKV file.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The id of the stream.
* \param[out] stream_name The name of the stream.
* \param[in] name_size The size of the buffer to store the stream name.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_stream_name(skv_handle* handle, uint32_t stream_id, char* stream_name, uint32_t name_size, skv_error* error);

/**
* \brief Gets the id of a stream in the SKV file.
*
* \param[in] handle An SKV handle.
* \param[in] stream_name The name of the stream.
* \param[out] stream_id The id of the stream.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_stream_id(skv_handle* handle, const char* stream_name, uint32_t* stream_id, skv_error* error);

/**
 * \brief Gets the information of an image stream with the given stream ID.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id An ID identifying the stream.
 * \param[out] info The image stream information.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_image_stream_info(skv_handle* handle, uint32_t stream_id,
                                                              skv_image_stream_info* info, skv_error* error);

/**
 * \brief Convenience function that sets the fields in the image
 *        stream info.
 *
 * \param[in] info An image stream info.
 * \param[in] name The image stream name - maximum 255 characters.
 * \param[in] type The image stream type.
 * \param[in] compression The image stream compression.
 * \param[in] width The image width.
 * \param[in] height The image height.
 */
SKV_API void SKV_SDK_DECL skv_assign_image_stream_info(skv_image_stream_info* info, const char* name,
                                                       skv_image_type type, skv_compression compression, uint32_t width,
                                                       uint32_t height);

/**
 * \brief Gets the custom stream information, given the stream ID.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The custom stream ID.
 * \param[out] info The custom stream info.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_custom_stream_info(skv_handle* handle, uint32_t stream_id,
                                                               skv_custom_stream_info* info, skv_error* error);

/**
 * \brief Convenience function that sets the fields in the image
 *        stream info.
 *
 * \param[in] info A custom stream info.
 * \param[in] name The custom stream name - maximum 255 characters.
 * \param[in] compression The custom stream compression.
 * \param[in] frame_size  The size of a custom frame in bytes.
 */
SKV_API void SKV_SDK_DECL skv_assign_custom_stream_info(skv_custom_stream_info* info, const char* name,
                                                        skv_compression compression, size_t frame_size);

/**
 * \brief Adds a new image stream to the SKV file.
 *
 * \param[in] handle An SKV handle.
 * \param[in] info An image stream info.
 * \param[out] stream_id The new stream ID.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_already_exists</tt>: A stream with this name already exists.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
 * - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_add_image_stream(skv_handle* handle, const skv_image_stream_info* info,
                                                         uint32_t* stream_id, skv_error* error);

/**
 * \brief Adds a new custom stream to the SKV file.
 *
 * \param[in] handle An SKV handle.
 * \param[in] info A custom stream info.
 * \param[out] stream_id The new stream ID.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_already_exists</tt>: A stream with this name already exists.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
 * - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_add_custom_stream(skv_handle* handle, const skv_custom_stream_info* info,
                                                          uint32_t* stream_id, skv_error* error);

/**
* \brief Renames an existing stream in the SKV file.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The stream ID.
* \param[in] name The new name for this stream.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_stream_already_exists</tt>: A stream with this name already exists.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
*/
SKV_API skv_error_code SKV_SDK_DECL skv_rename_stream(skv_handle* handle, uint32_t stream_id, const char* name, skv_error* error);

/**
 * \brief Removes an images stream from the SKV file.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
 * - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_remove_stream(skv_handle* handle, uint32_t stream_id, skv_error* error);

/**
 * \brief Gets the stream type (either an image stream or custom stream) given a stream ID.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id A stream ID.
 * \param[out] stream_type The stream type.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_stream_type(skv_handle* handle, uint32_t stream_id,
                                                        skv_stream_type* stream_type, skv_error* error);

/**
 * \brief Gets the number of frames in the stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id A stream ID.
 * \param[out] frame_count The total number of frames in the stream.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_stream_frame_count(skv_handle* handle, uint32_t stream_id,
                                                               uint32_t* frame_count, skv_error* error);

/**
 * \brief Gets the number of bytes in the current frame.
 *
 * \param[in] handle An SKV handle.
 * \param[out] byte_count The total number of bytes in the current frame.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \sa skv_get_current_frame_data
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_current_frame_byte_count(skv_handle* handle, size_t* byte_count, skv_error* error);

/**
 * \brief Copies the current frame data into an array provided by the user.
 *
 * \param[in] handle An SKV handle.
 * \param[out] data A pointer pre-allocated array of fixed size.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \sa skv_get_current_frame_byte_count
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_current_frame_data(skv_handle* handle, void* data, skv_error* error);

/**
 * \brief Gets the current frame index.
 *
 * \param[in] handle An SKV handle.
 * \param[out] index The current frame index.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_seek_not_initiated</tt>: The seek functions were not called prior to this function.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_current_frame_index(skv_handle* handle, uint32_t* index, skv_error* error);

/**
 * \brief Gets the stream ID for the current frame.
 *
 * \param[in] handle An SKV handle.
 * \param[out] stream_id The stream ID.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_seek_not_initiated</tt>: The seek functions were not called prior to this function.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_current_frame_stream_id(skv_handle* handle, uint32_t* stream_id, skv_error* error);

/**
 * \brief Gets the timestamp for the current frame.
 *
 * \param[in] handle An SKV handle.
 * \param[out] timestamp The timestamp (microseconds) of the current frame.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_seek_not_initiated</tt>: The seek functions were not called prior to this function.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_current_frame_timestamp(skv_handle* handle, uint64_t* timestamp, skv_error* error);

/**
* \brief Gets the number of bytes in the frame specified by stream_id, frame_index.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The ID of the stream we are looking in.
* \param[in] frame_index The index of the frame that we want to retrieve.
* \param[out] byte_count The total number of bytes in the specified frame.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
* \sa skv_get_frame_data
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_frame_byte_count(skv_handle* handle, uint32_t stream_id, uint32_t frame_index, size_t* byte_count, skv_error* error);

/**
* \brief Gets the number of bytes in the frame specified by stream_id and timestamp.
*
* Unlike get_frame_byte_count([...], index, [...]), it is not an error for the timestamp parameter p
* to be larger than the maximum timestamp of the stream.
* Instead, it will retrieve the frame with the largest timestamp.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The ID of the stream we are looking in.
* \param[in] time_stamp The timestamp of the frame that we want to retrieve.
* \param[out] byte_count The total number of bytes in the specified frame.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
* - <tt>skv_error_code_timestamp_out_of_range</tt>: The timestamp is out of range.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
* \sa skv_get_frame_data_by_timestamp
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_frame_byte_count_by_timestamp(skv_handle* handle, uint32_t stream_id, uint64_t time_stamp, size_t* byte_count, skv_error* error);


/**
* \brief Copies the data from the frame specified by stream_id, frame_index into an array provided by the user.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The ID of the stream we are looking in.
* \param[in] frame_index The index of the frame that we want to retrieve.
* \param[out] data A pointer pre-allocated array of fixed size.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
* \sa skv_get_frame_byte_count
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_frame_data(skv_handle* handle, uint32_t stream_id, uint32_t frame_index, void* data, skv_error* error);

/**
* \brief Copies the data from the frame specified by stream_id and timestamp into an array provided by the user.
*
* Unlike get_frame_data([...], index, [...]), it is not an error for the timestamp parameter
* to be larger than the maximum timestamp of the stream.
* Instead, it will retrieve the frame with the largest timestamp.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The ID of the stream we are looking in.
* \param[in] time_stamp The timestamp of the frame that we want to retrieve.
* \param[out] data A pointer pre-allocated array of fixed size.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
* - <tt>skv_error_code_timestamp_out_of_range</tt>: The timestamp is out of range.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
* \sa skv_get_frame_byte_count_by_timestamp
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_frame_data_by_timestamp(skv_handle* handle, uint32_t stream_id, uint64_t time_stamp, void* data, skv_error* error);

/**
* \brief Gets the index of the frame specified by stream_id and timestamp.
*
* Unlike get_frame_byte_count([...], index, [...]), it is not an error for the timestamp parameter p
* to be larger than the maximum timestamp of the stream.
* Instead, it will retrieve the frame with the largest timestamp.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The ID of the stream we are looking in.
* \param[in] time_stamp The timestamp of the frame that we want to retrieve.
* \param[out] frame_index The index of the frame that we want to retrieve.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
* - <tt>skv_error_code_timestamp_out_of_range</tt>: The timestamp is out of range.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
* \sa skv_get_frame_data_by_timestamp
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_frame_index(skv_handle* handle, uint32_t stream_id, uint64_t time_stamp, uint32_t* frame_index, skv_error* error);

/**
* \brief Gets the timestamp of the frame specified by stream_id, frame_index.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The ID of the stream we are looking in.
* \param[in] frame_index The index of the frame that we want to retrieve.
* \param[out] time_stamp The timestamp of the frame that we want to retrieve.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
* \sa skv_get_frame_data
*/
SKV_API skv_error_code SKV_SDK_DECL skv_get_frame_timestamp(skv_handle* handle, uint32_t stream_id, uint32_t frame_index, uint64_t* time_stamp, skv_error* error);

/**
 * \brief Seeks to the frame with the given index.
 *
 * A typical usage of the seek functions might be:
 *
 * \code
 * // seek frame
 * uint32_t index = ...;
 * skv_error_code ec = skv_seek_frame_by_index(handle, stream_id, frame_index, &error);
 *
 * // allocate a buffer and read the data of the frame at this index
 * size_t byte_count = 0;
 * ec = skv_get_current_frame_byte_count(handle, &byte_count);
 * auto frame_data = new char[byte_count];
 * ec = skv_get_current_frame_data(handle, frame_data);
 * // ...
 * \endcode
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[in] index The frame index.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \sa seek_frame_by_timestamp
 */
SKV_API skv_error_code SKV_SDK_DECL skv_seek_frame_by_index(skv_handle* handle, uint32_t stream_id, uint32_t index, skv_error* error);

/**
 * \brief Seeks to the frame closest to the given timestamp.
 *
 * A typical usage of the seek functions might be:
 *
 * \code
 * // seek frame
 * uint64_t timestamp = ...;
 * skv_seek_frame_by_timestamp(handle, stream_id, timestamp);
 *
 * // allocate a buffer and read the data of the frame at this index
 * size_t byte_count = 0;
 * skv_get_current_frame_byte_count(handle, &byte_count);
 * auto frame_data = new char[byte_count];
 * skv_get_current_frame_data(handle, frame_data);
 * // ...
 * \endcode
 *
 * Unlike skv_seek_frame_by_index, it is not an error for the timestamp parameter to be larger than the maximum timestamp
 * of the stream. Instead, it will seek to the frame with the largest timestamp.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[in] timestamp The frame timestamp (microseconds).
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
 * - <tt>skv_error_code_timestamp_out_of_range</tt>: The timestamp is out of range.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 * \sa seek_frame_by_index
 */
SKV_API skv_error_code SKV_SDK_DECL skv_seek_frame_by_timestamp(skv_handle* handle, uint32_t stream_id,
                                                                uint64_t timestamp, skv_error* error);

/**
 * \brief Seeks to the next frame in the SKV file.
 *
 * \param[in] handle The SKV handle.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist in this stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_seek_next_frame(skv_handle* handle, skv_error* error);

/**
 * \brief Adds a frame to a stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[in] timestamp A timestamp value (microseconds).
 * \param[in] data A pointer to the frame data.
 * \param[in] byte_count The number of bytes to write.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
 * - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_byte_count_does_not_match</tt>: Can't add a frame whose byte_count does not match with the expected size of the image stream. 
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_add_frame(skv_handle* handle, uint32_t stream_id, uint64_t timestamp,
                                                  const void* data, size_t byte_count, skv_error* error);

/**
 * \brief Adds multiple consecutive frames to a stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[in] timestamps An array of timestamps for the frames.
 * \param[in] data An array of pointers to the frames' data.
 * \param[in] frame_count The number of frames to write.
 * \param[in] byte_count The number of bytes per frame to write.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
 * - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_byte_count_does_not_match</tt>: Can't add a frame whose byte_count does not match with the expected size of the image stream. 
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_add_frames(skv_handle* handle, uint32_t stream_id, const uint64_t* timestamps,
                                                   const void* const* data, size_t frame_count, size_t byte_count, skv_error* error);

/**
* \brief Modifies the internal data of a frame.
*
 * \warning The size of the data buffer must be the same as what is in the file. Providing data with a smaller or
 * larger buffer size will result in unexpected behavior.
 *
* \param[in] handle An SKV handle.
* \param[in] stream_id The stream ID.
* \param[in] index The frame index.
* \param[in] data The new frame data.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
*/
SKV_API skv_error_code SKV_SDK_DECL skv_modify_frame_data(skv_handle* handle, uint32_t stream_id, uint32_t index,
	const void* data, skv_error* error);

/**
* \brief Modifies the timestamp of a series of successive frames.
*
* \warning The number of frame indices between <tt>start_index</tt> and <tt>end_index</tt> (inclusive)
* determines the size of the array of timestamps. If the number of timestamps in the array does not
* equal <tt>end_index - start_index</tt> the behavior is undefined.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The stream ID.
* \param[in] start_index The frame index of the first frame to modify (inclusive)
* \param[in] end_index The frame index of the last frame to modify (inclusive)
* \param[in] timestamps An array of new timestamps that will replace the original timestamps.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded.
* - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
*/
SKV_API skv_error_code SKV_SDK_DECL skv_modify_frame_timestamps(skv_handle* handle, uint32_t stream_id,
	uint32_t start_index, uint32_t end_index, uint64_t* timestamps, skv_error* error);

/**
* \brief Removes a set of frames from a stream.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The stream ID.
* \param[in] index_begin The frame index of the first frame to be removed.
* \param[in] index_end The frame index of the last frame to be removed.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded
* - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
*/
SKV_API skv_error_code SKV_SDK_DECL skv_remove_frames(skv_handle* handle, uint32_t stream_id, uint32_t index_begin, uint32_t index_end, skv_error* error);

/**
* \brief Removes a set of frames from a stream based on timestamp.
*
* \param[in] handle An SKV handle.
* \param[in] stream_id The stream ID.
* \param[in] timestamp_begin The timestamp of the first frame to be removed.
* \param[in] timestamp_end The timestamp of the last frame to be removed.
* \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
* \returns
* - <tt>skv_error_code_success</tt>: The operation succeeded
* - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
* - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
* - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
* - <tt>skv_error_code_timestamp_out_of_range</tt>: The timestamp is out of range.
* - <tt>skv_error_code_frame_does_not_exist</tt>: The frame does not exist.
* - <tt>skv_error_code_internal_error</tt>: There was an internal error.
*/
SKV_API skv_error_code SKV_SDK_DECL skv_remove_frames_by_timestamp(skv_handle* handle, uint32_t stream_id, uint64_t timestamp_begin, uint64_t timestamp_end, skv_error* error);

/**
 * \brief Tests if this SKV stream has a pinhole model.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[out] has_pinhole_model \c true if the stream has a pinhole model, \c false otherwise.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_has_pinhole_model(skv_handle* handle, uint32_t stream_id,
                                                          bool* has_pinhole_model, skv_error* error);

/**
 * \brief Gets the pinhole model for an SKV stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[out] model The pinhole model, if it exists.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_pinhole_model(skv_handle* handle, uint32_t stream_id,
                                                          skv_pinhole_model* model, skv_error* error);

/**
 * \brief Sets or overwrites the pinhole model for an SKV stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[in] model The pinhole model.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_set_pinhole_model(skv_handle* handle, uint32_t stream_id,
                                                          const skv_pinhole_model* model, skv_error* error);

/**
 * \brief Tests if this SKV stream has a distortion model.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[out] has_distortion_model \c true if the stream has a distortion model, \c false otherwise.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_has_distortion_model(skv_handle* handle, uint32_t stream_id,
                                                             bool* has_distortion_model, skv_error* error);

/**
 * \brief Gets the distortion model for an SKV stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[out] model The distortion model, if it exists.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_distortion_model(skv_handle* handle, uint32_t stream_id,
                                                             skv_distortion_model* model, skv_error* error);

/**
 * \brief Sets or overwrites the distortion model for an SKV stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[in] model The distortion model.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_set_distortion_model(skv_handle* handle, uint32_t stream_id,
                                                             const skv_distortion_model* model, skv_error* error);

/**
 * \brief Tests if this SKV stream has a stereo transform.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[out] has_stereo_transform \c true if the stream has a stereo transform, \c false otherwise.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_has_stereo_transform(skv_handle* handle, uint32_t stream_id,
                                                             bool* has_stereo_transform, skv_error* error);

/**
 * \brief Gets the stereo transform for an SKV stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[out] transform The stereo transform, if it exists.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_stereo_transform(skv_handle* handle, uint32_t stream_id,
                                                             skv_stereo_transform* transform, skv_error* error);

/**
 * \brief Sets or overwrites the stereo transform for an SKV stream.
 *
 * \param[in] handle An SKV handle.
 * \param[in] stream_id The stream ID.
 * \param[in] transform The stereo transform.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_stream_does_not_exist</tt>: The stream does not exist.
 * - <tt>skv_error_code_custom_stream_does_not_support_field</tt>: This stream is a custom stream.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_set_stereo_transform(skv_handle* handle, uint32_t stream_id,
                                                             const skv_stereo_transform* transform, skv_error* error);

/**
 * \brief Gets the number of custom buffers in the SKV stream.
 *
 * \param[in] handle An SKV handle.
 * \param[out] buffer_count The number of custom buffers.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an intebuffer_byte_count(handle, 
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_custom_buffer_count(skv_handle* handle, uint32_t* buffer_count, skv_error* error);

/**
 * \brief Tests if the SKV file contains a custom buffer with the given name.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The custom buffer name.
 * \param[out] has_custom_buffer \c true if the custom buffer exists, \c false otherwise.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_has_custom_buffer(skv_handle* handle, const char* name,
                                                          bool* has_custom_buffer, skv_error* error);

/**
 * \brief Gets the name of the buffer with the given ID.
 *
 * \param[in] handle An SKV handle.
 * \param[in] buffer_id The buffer ID.
 * \param[out] name The name of the custom buffer.
 * \param[out] name_size The number of bytes in the name.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_custom_buffer_does_not_exist</tt>: The buffer does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_custom_buffer_name(skv_handle* handle, uint32_t buffer_id, char* name,
                                                               uint32_t name_size, skv_error* error);

/**
 * \brief Gets the number of bytes in the custom buffer.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The name of the custom buffer.
 * \param[out] byte_count The number of bytes in the custom buffer.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_custom_buffer_does_not_exist</tt>: The buffer does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_custom_buffer_byte_count(skv_handle* handle, const char* name,
                                                                     size_t* byte_count, skv_error* error);

/**
 * \brief Gets the compression used in the custom buffer.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The name of the custom buffer.
 * \param[out] compression The compression used in the custom buffer.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_custom_buffer_does_not_exist</tt>: The buffer does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_custom_buffer_compression(skv_handle* handle, const char* name,
                                                                      skv_compression* compression, skv_error* error);

/**
 * \brief Gets the custom buffer data.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The name of the custom buffer.
 * \param[out] data An address to a pre-allocated block of data.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_custom_buffer_does_not_exist</tt>: The buffer does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_get_custom_buffer_data(skv_handle* handle, const char* name, void* data, skv_error* error);

/**
 * \brief Adds a custom buffer to the file.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The name of the new custom buffer.
 * \param[in] data An address to the data to write.
 * \param[in] byte_count The number of bytes to write.
 * \param[in] compression The compression to use for the custom buffer.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_custom_buffer_already_exists</tt>: A buffer with this name already exists.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_add_custom_buffer(skv_handle* handle, const char* name, const void* data,
                                                          size_t byte_count, skv_compression compression, skv_error* error);

/**
 * \brief Removes a custom buffer from the file.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The name of the custom buffer.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_custom_buffer_does_not_exist</tt>: The custom buffer does not exist.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_remove_custom_buffer(skv_handle* handle, const char* name, skv_error* error);

/**
 * \brief Renames a custom buffer.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The name of the existing custom buffer to rename.
 * \param[in] new_name The new name of the custom buffer.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
* - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_rename_custom_buffer(skv_handle* handle, const char* name,
                                                             const char* new_name, skv_error* error);

/**
 * \brief Modifies an existing custom buffer's data.
 *
 * \warning The byte count must be the same as what is in the file. Providing data with a smaller or larger buffer
 * size will result in unexpected behavior.
 *
 * \param[in] handle An SKV handle.
 * \param[in] name The name of the custom buffer.
 * \param[in] data The new data to write to the custom buffer.
 * \param[in] byte_count The number of bytes to write.
 * \param[out] error Data structure containing additional error information. Pass a null pointer to ignore.
 * \returns
 * - <tt>skv_error_code_success</tt>: The operation succeeded.
 * - <tt>skv_error_code_cannot_modify_readonly_file</tt>: Can't modify a file that was opened in read-only mode.
 * - <tt>skv_error_code_cannot_modify_legacy_format</tt>: Can't modify a file recorded using the legacy file format.
 * - <tt>skv_error_code_file_is_closed</tt>: The file is not open, and can't be accessed.
 * - <tt>skv_error_code_internal_error</tt>: There was an internal error.
 */
SKV_API skv_error_code SKV_SDK_DECL skv_modify_custom_buffer(skv_handle* handle, const char* name, const void* data,
                                                             size_t byte_count, skv_error* error);
#ifdef __cplusplus
}
#endif

#endif  // SOFTKINETIC_SKV_CORE_INC
