/***************************************************************************************/
//  SoftKinetic SKV library
//  Project Name      : SKV
//  Module Name	      : SKV API
//  Description       : C API for reading/writing skv files
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

/*!
  \mainpage SKV API Manual
 */

#pragma once
#ifndef SOFTKINETIC_SKV_API_INCLUDED
#define SOFTKINETIC_SKV_API_INCLUDED

#include <SK/SKV/platform.h>
#include <SK/SKV/types.h>
#include <SK/SKV/version.h>

#define _unused(x) ((void)(x))

#ifdef __cplusplus
extern "C" {
#endif

/// A handle to an SKV movie file.
typedef void* SKV_FileHandle;

/// Type of stream
typedef enum SKV_StreamType
{
	SKV_STREAM_UNKNOWN = 0,
	/// Image stream, further specified with \ref SKV_ImageType.
	SKV_STREAM_IMAGE,
	/// Custom stream, contains blob data defined in client app.
	SKV_STREAM_CUSTOM,
} SKV_StreamType;

/// Subtype for image streams
typedef enum SKV_ImageType
{
	SKV_IMAGE_UNKNOWN = 0,
	SKV_IMAGE_INT8 = 1,
	SKV_IMAGE_UINT8 = 2,
	SKV_IMAGE_INT16 = 3,
	SKV_IMAGE_UINT16 = 4,
	SKV_IMAGE_INT32 = 5,
	SKV_IMAGE_UINT32 = 6,
	SKV_IMAGE_BGR24 = 7,
	SKV_IMAGE_YUV16 = 8,
	SKV_IMAGE_FLOAT = 9,
	SKV_IMAGE_RGB24 = 10,
	SKV_IMAGE_BGRA32 = 11,
	SKV_IMAGE_RGBA32 = 12,
	SKV_IMAGE_DOUBLE = 13
} SKV_ImageType;

/// Compression mode for data
typedef enum SKV_CompressionType
{
	/// No compression.
	SKV_COMPRESSION_NONE = 0,
	/// Lossless compression using Snappy.
	SKV_COMPRESSION_SNAPPY = 1,
	/// Lossless compression using ZLib.
	SKV_COMPRESSION_ZLIB = 3,
	/// Lossless compression using LZ4.
	SKV_COMPRESSION_LZ4 = 5
} SKV_CompressionType;

typedef enum SKV_FileMode
{
	/// Open a file with read and write capabilities.
	SKV_READ_WRITE = 0,
	/// Open a file with read only capabilities.
	SKV_READ_ONLY = 1,
} SKV_FileMode;

/// Description structure of an image stream
typedef struct SKV_ImageStreamInfo
{
	/// Name of the stream.
	char name[256];
	/// Image type, as specified in \ref SKV_ImageType.
	SKV_ImageType type;
	/// Compression type, as specified in \ref SKV_CompressionType.
	SKV_CompressionType compression;
	/// Width (in pixels) of the image data in the stream.
	uint32_t width;
	/// Height (in pixels) of the image data in the stream.
	uint32_t height;
} SKV_ImageStreamInfo;

/// Description structure of a custom stream
typedef struct SKV_CustomStreamInfo
{
	/// Name of the stream.
	char name[256];
	/// Compression type, as specified in \ref SKV_CompressionType.
	SKV_CompressionType compression;
} SKV_CustomStreamInfo;

/// Description structure of a custom stream
typedef struct SKV_CustomBufferInfo
{
	/// Name of the buffer.
	char name[256];
	/// Size of the buffer.
	uint32_t size;
	/// Compression type, as specified in \ref SKV_CompressionType.
	SKV_CompressionType compression;
} SKV_CustomBufferInfo;

/// \brief Pinhole camera model parameters
/// \remark The position at (0,0) in the image is at top left.
typedef struct SKV_PinholeModel
{
	/// The field of view along the x axis expressed in radians units.
	float fx;
	/// The field of view along the y axis expressed in radians units.
	float fy;
	/// The central point along the x axis expressed as a ratio of the image width.
	float cx;
	/// The central point along the y axis expressed as a ratio of the image height.
	float cy;
} SKV_PinholeModel;

/// Brown image distortion model parameters
typedef struct SKV_DistortionModel
{
	/// the focal length along the x axis, expressed in pixel units
	float fx;
	/// the focal length along the y axis, expressed in pixel units
	float fy;
	/// the first radial distortion coefficient
	float k1;
	/// the second radial distortion coefficient
	float k2;
	/// the third radial distortion coefficient
	float k3;
	/// the fourth radial distortion coefficient
	float k4;
	/// the first tangential distortion coefficient
	float p1;
	/// the second tangential distortion coefficient
	float p2;
} SKV_DistortionModel;

/// Stream to stream transform
typedef struct SKV_StereoTransform
{
	/**
	 * \name 3x3 Rotation matrix
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
	 * @{
	 */
	float t1;
	float t2;
	float t3;
	/**
	 * @}
	 */
} SKV_StereoTransform;

/// Structure for information about the device used for recording
typedef struct SKV_DeviceInfo
{
	/// name of the vendor of the device
	char vendorName[256];
	/// model name of the device
	char cameraModel[256];
} SKV_DeviceInfo;




/**
 * \name Predefined stream names for color and depth streams
 * @{
 */
#define SKV_ACCELEROMETER_STREAM "Accelerometer"
#define SKV_ATTITUDEMATRIX_STREAM "Attitude matrix"
#define SKV_STATUSBITMASK_STREAM "Status bit mask"

#define SKV_COLOR_STREAM "Color stream"
#define SKV_COLOR_STREAM_LEFT "Color stream left"
#define SKV_COLOR_STREAM_RIGHT "Color stream right"

#define SKV_DEPTH_STREAM "Depth stream"
#define SKV_DEPTH_STREAM_LEFT "Depth stream left"
#define SKV_DEPTH_STREAM_RIGHT "Depth stream right"
#define SKV_DEPTH_STREAM_FLOAT_PLANES_XYZ "Depth stream float XYZ"

#define SKV_CONFIDENCE_STREAM "Confidence stream"
#define SKV_CONFIDENCE_STREAM_LEFT "Confidence stream left"
#define SKV_CONFIDENCE_STREAM_RIGHT "Confidence stream right"

#define SKV_YUV_STREAM "YUV stream" 
#define SKV_YUV_STREAM_LEFT "YUV stream left" 
/**
 * @}
 */

/**
 * \brief Get the version of the DLL (major version, minor version, patch version, and release stage)
 *
 * \param[out] version Pointer to a string that will contain the version
 *
 * \pre `version` may not be NULL
 * \pre `version` must be allocated with at least 256
 *
 * \post `version` will contain the version numbers and release stage of the DLL
 */
SKV_API void SKV_SDK_DECL SKV_getDLLVersion( char* version );

/**
 * \brief Open a movie
 *
 * \param[in] fileName The absolute or relative path of the file to open
 * \param[out] fileHandle A handle to the file
 * \param[in] mode The mode in which the file will be open (r/w or read only).
 *
 * \return `0` if no error \n
 *         `1` if file doesn't open (doesn't exist, is locked, ...) \n
 *         `2` if file is corrupted or not a proper skv \n
 *         `3` if filehandle is invalid
 *
 * \pre `fileHandle` may not be NULL
 *
 * \post `fileHandle` is updated to the opened file
 */
SKV_API uint32_t SKV_SDK_DECL SKV_openMovie( const char* fileName, SKV_FileHandle* fileHandle, SKV_FileMode mode);

/**
 * \brief Close a movie
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return `true` if successfully closed \n
 *         `false` if file didn't close properly
 *
 * \pre `fileHandle` may not be NULL
 *
 * \post Movie is opened in read mode
 */
SKV_API bool_t SKV_SDK_DECL SKV_closeMovie( SKV_FileHandle fileHandle );

/**
 * \brief Get the major version of the file format
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return The major version part of the file format version
 *
 * \pre `fileHandle` may not be NULL
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getMajorFormatVersion( SKV_FileHandle fileHandle );

/**
 * \brief Get the minor version of file the format
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return The minor version part of the file format version
 *
 * \pre `fileHandle` may not be NULL
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getMinorFormatVersion( SKV_FileHandle fileHandle );

/**
 * \brief Check if a stream has information about the device used for recording
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return `true` if the stream has information about the device used during recording \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The file must be in read mode
 */
SKV_API bool_t SKV_SDK_DECL SKV_isDeviceInfoSupported( SKV_FileHandle fileHandle );

/**
 * \brief Check if a stream has pinhole model information
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream for which to check pinhole model information
 *
 * \return `true` if the stream has pinhole model information \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The file must be in read mode
 */
SKV_API bool_t SKV_SDK_DECL SKV_isPinholeModelSupported( SKV_FileHandle fileHandle, uint32_t streamID );

/**
 * \brief Check if a stream has distortion model information
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream for which to check distortion model information
 *
 * \return `true` if the stream has distortion model information \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The file must be in read mode
 */
SKV_API bool_t SKV_SDK_DECL SKV_isDistortionModelSupported( SKV_FileHandle fileHandle, uint32_t streamID );

/**
 * \brief Check if stereo transform information exists for a stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the source stream for which to check stereo transform information
 *
 * \return `true` if there exists a stereo transform for the stream \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The file must be in read mode
 */
SKV_API bool_t SKV_SDK_DECL SKV_isStereoTransformSupported( SKV_FileHandle fileHandle, uint32_t streamID );

/**
 * \brief Get information about the recording device
 *
 * \param[in] fileHandle A handle to the file
 * \param[out] deviceInfo Pointer to a struct that will contain the information
 *
 * \pre `fileHandle` may not be NULL
 * \pre `deviceInfo` may not be NULL
 * \pre The file must be in read mode
 *
 * \post The deviceInfo struct will contain the requested recording device information
 */
SKV_API void SKV_SDK_DECL SKV_getDeviceInfo( SKV_FileHandle fileHandle, SKV_DeviceInfo* deviceInfo );

/**
 * \brief Get pinhole model information for a stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream from which to get the information
 * \param[out] pinholeModel Pointer to a struct that will contain the information
 *
 * \pre `fileHandle` may not be NULL
 * \pre `pinholeModel` may not be NULL
 * \pre The file must be in read mode
 * \pre The stream with given ID must contain pinhole information (to be checked with \ref SKV_isPinholeModelSupported)
 *
 * \post The pinholeModel struct will contain the requested pinhole model information
 */
SKV_API void SKV_SDK_DECL SKV_getPinholeModel( SKV_FileHandle fileHandle, uint32_t streamID, SKV_PinholeModel* pinholeModel );

/**
 * \brief Get distortion model information for a stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream from which to get the information
 * \param[out] distortionModel Pointer to a struct that will contain the information
 *
 * \pre `fileHandle` may not be NULL
 * \pre `distortionModel` may not be NULL
 * \pre The file must be in read mode
 * \pre The stream with given ID must contain distortion information (to be checked with \ref SKV_isDistortionModelSupported)
 *
 * \post The distortionModel struct will contain the requested distortion model information
 */
SKV_API void SKV_SDK_DECL SKV_getDistortionModel( SKV_FileHandle fileHandle, uint32_t streamID, SKV_DistortionModel* distortionModel );

/**
 * \brief Get stereo transform information for a stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream for the requested transform
 * \param[out] transform Pointer to a struct that will contain the information
 *
 * \pre `fileHandle` may not be NULL
 * \pre `transform` may not be NULL
 * \pre The file must be in read mode
 * \pre A transform from `streamID` must exist (to be checked with \ref SKV_isStereoTransformSupported)
 *
 * \post The transform struct will contain the requested stereo transform information from source to target stream
 */
SKV_API void SKV_SDK_DECL SKV_getStereoTransform( SKV_FileHandle fileHandle, uint32_t streamID, SKV_StereoTransform* transform );

/**
 * \brief Get the number of streams in the movie
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return The number of streams in the movie
 *
 * \pre `fileHandle` may not be NULL
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getStreamCount( SKV_FileHandle fileHandle );

/**
 * \brief Get the type of the stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream for which to get the type
 *
 * \return Type of the stream (Image, ...)
 *
 * \pre `fileHandle` may not be NULL
 * \pre The stream with given ID must exist in the movie
 */
SKV_API SKV_StreamType SKV_SDK_DECL SKV_getStreamType( SKV_FileHandle fileHandle, uint32_t streamID );

/**
 * \brief Get image information about an image stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the image stream for which to get the image information
 * \param[out] info Pointer to a struct that will contain the image information
 *
 * \pre `fileHandle` may not be NULL
 * \pre `info` may not be NULL
 * \pre The stream with given ID must exist in the movie
 * \pre The stream with given ID must be an image stream (to be checked with \ref SKV_getStreamType)
 *
 * \post The info struct will contain the requested image stream information
 */
SKV_API void SKV_SDK_DECL SKV_getImageStreamInfo( SKV_FileHandle fileHandle, uint32_t streamID, SKV_ImageStreamInfo* info );

/**
 * \brief Get information about a custom stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the custom stream for which to get the information
 * \param[out] info Pointer to a struct that will contain the information
 *
 * \pre `fileHandle` may not be NULL
 * \pre `info` may not be NULL
 * \pre The stream with given ID must exist in the movie
 * \pre The stream with given ID must be a custom stream (to be checked with \ref SKV_getStreamType)
 *
 * \post The info struct will contain the requested custom stream information
 */
SKV_API void SKV_SDK_DECL SKV_getCustomStreamInfo( SKV_FileHandle fileHandle, uint32_t streamID, SKV_CustomStreamInfo* info );

/**
 * \brief Get the pixel size in bytes for a given image type
 *
 * \param[in] imageType The type of image (\ref SKV_ImageType) to get the pixel size for
 *
 * \return The size of a pixel in bytes
 *
 * \pre imageType must be valid type
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getPixelSize( SKV_ImageType imageType );

/**
 * \brief Get the number of frames of the given stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the image stream for which to get the number of frames
 *
 * \return The number of frames in the given stream
 *
 * \pre `fileHandle` may not be NULL
 * \pre The stream with given ID must exist in the movie
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getStreamFrameCount( SKV_FileHandle fileHandle, uint32_t streamID );

/**
 * \brief Get a stream's average fps
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the image stream for which to get the number of frames
 *
 * \return The average number of frames per second of the given stream. If the stream has 2 frames or less, then this function returns 0.0. If the total duration of the movie is 0, then this function returns 0.0
 *
 * \pre `fileHandle` may not be NULL
 * \pre The stream with given ID must exist in the movie
 */
SKV_API float SKV_SDK_DECL SKV_getStreamAverageFPS( SKV_FileHandle fileHandle, uint32_t streamID );

/**
 * \brief Seek the next frame across all streams (independent of type)
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return `false` if there is no next frame \n
 *         `true` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read
 *
 * \post The current frame marker is advanced to the next frame in time (which can come from any stream) if it exists.\n
 *       If it doesn't, the current frame is not changed.
 */
SKV_API bool_t SKV_SDK_DECL SKV_seekNextFrame( SKV_FileHandle fileHandle );

/**
 * \brief Seek the frame specified by the index and the stream ID
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the image stream in which to seek
 * \param[in] index The frame index of the chosen stream to seek
 *
 * \return `false` if there is no frame \n
 *         `true` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read mode
 * \pre The stream with given ID must exist in the movie
 *
 * \post The current frame marker is advanced to the frame with the given index in the given stream if it exists.\
 *       If it doesn't, the current frame is not changed.
 */
SKV_API bool_t SKV_SDK_DECL SKV_seekFrameByIndex( SKV_FileHandle fileHandle, uint32_t streamID, uint32_t index );

/**
 * \brief Seek the frame specified by the timestamp and the stream ID
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the image stream in which to seek
 * \param[in] timeStamp The timestamp in the chosen stream to seek
 *
 * \return `false` if the timestamp is not within the total timeframe for this stream \n
 *         `true` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read mode
 * \pre The stream with given ID must exist in the movie
 *
 * \post The current frame marker is advanced to the frame that is active at the given timestamp, if within time limits.\
 *       If the timestamp is beyond the total movie time, the current frame is not changed.
 */
SKV_API bool_t SKV_SDK_DECL SKV_seekFrameByTimeStamp( SKV_FileHandle fileHandle, uint32_t streamID, uint64_t timeStamp );

/**
 * \brief Get the stream type of the current frame
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return The ID of the stream that contains the current frame. \n
 *         `UINT_MAX` if the current frame is not set.
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read mode
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getCurrentFrameStreamID( SKV_FileHandle fileHandle );

/**
 * \brief Get the index of the current frame inside its stream
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return The frame index of the current frame.
 *         `UINT_MAX` if the current frame is not set.
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read mode
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getCurrentFrameIndex( SKV_FileHandle fileHandle );

/**
 * \brief Get the timeStamp of the current frame
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return The time stamp of the current frame within the movie. \n
 *         If the frame is not set, returns minimal 64-bit integer value (std::numeric_limits<uint64_t>::min()).
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read mode
 */
SKV_API uint64_t SKV_SDK_DECL SKV_getCurrentFrameTimeStamp( SKV_FileHandle fileHandle );

/**
 * \brief Get the size of the uncompressed frame data of the current frame
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return The size (in bytes) of the current frame data buffer. \n
 *         `0` if the current frame is not set.
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read mode
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getCurrentFrameDataSize( SKV_FileHandle fileHandle );

/**
 * \brief Get the uncompressed frame data of the current frame
 *
 * \param[in] fileHandle A handle to the file
 * \param[out] frameData A buffer for the uncompressed frame data
 *
 * \return `true` if the current frame is set. \n
 *         `false` otherwise.
 *
 * \pre `fileHandle` may not be NULL
 * \pre File must be in read mode
 *
 * \post If the current frame is set, `frameData` contains the uncompressed frame data of the current frame. \n
 *       Otherwise, `frameData` is unchanged.
 */
SKV_API bool_t SKV_SDK_DECL SKV_getCurrentFrameData( SKV_FileHandle fileHandle, void* frameData );

/**
* \brief Gets the number of bytes in the frame specified by streamID, index.
*
* \param[in] fileHandle An SKV handle.
* \param[in] streamID The ID of the stream we are looking in.
* \param[in] index The index of the frame that we want to retrieve.
* \return The size (in bytes) of the frame specified by streamID, index. \n
*
* \pre `fileHandle` may not be NULL
* \pre File must be in read mode
*/
SKV_API uint32_t SKV_SDK_DECL SKV_getFrameDataSize(SKV_FileHandle fileHandle, uint32_t streamID, uint32_t index);

/**
* \brief Gets the number of bytes in the frame specified by streamID and timestamp.
*
* Unlike SKV_getFrameDataSize([...], index, [...]), it is not an error for the timestamp parameter p
* to be larger than the maximum timestamp of the stream.
* Instead, it will retrieve the frame with the largest timestamp.
*
* \param[in] fileHandle An SKV handle.
* \param[in] streamID The ID of the stream we are looking in.
* \param[in] timeStamp The timestamp of the frame that we want to retrieve.
* \return The size (in bytes) of the frame specified by streamID and timestamp. \n
*
* \pre `fileHandle` may not be NULL
* \pre File must be in read mode
*/
SKV_API uint32_t SKV_SDK_DECL SKV_getFrameDataSizeByTimestamp(SKV_FileHandle fileHandle, uint32_t streamID, uint64_t timeStamp);

/**
* \brief Copies the data from the frame specified by streamID, index into an array provided by the user.
*
* \param[in] fileHandle An SKV handle.
* \param[in] streamID The ID of the stream we are looking in.
* \param[in] index The index of the frame that we want to retrieve.
* \param[out] frameData A pointer pre-allocated array of fixed size.
*
* \return `true` if the current frame is set. \n
*         `false` otherwise.
*
* \pre `fileHandle` may not be NULL
* \pre File must be in read mode
*
* \post If the current frame is set, `frameData` contains the uncompressed frame data of the current frame. \n
*       Otherwise, `frameData` is unchanged.
* \sa SKV_getFrameDataSize
*/
SKV_API bool_t SKV_SDK_DECL SKV_getFrameData(SKV_FileHandle fileHandle, uint32_t streamID, uint32_t index, void* frameData);

/**
* \brief Copies the data from the frame specified by streamID and timestamp into an array provided by the user.
*
* \param[in] fileHandle An SKV handle.
* \param[in] streamID The ID of the stream we are looking in.
* \param[in] timeStamp The timestamp of the frame that we want to retrieve.
* \param[out] frameData A pointer pre-allocated array of fixed size.
*
* \return `true` if the current frame is set. \n
*         `false` otherwise.
*
* \pre `fileHandle` may not be NULL
* \pre File must be in read mode
*
* \post If the current frame is set, `frameData` contains the uncompressed frame data of the current frame. \n
*       Otherwise, `frameData` is unchanged.
* \sa SKV_getFrameDataSizeByTimestamp
*/
SKV_API bool_t SKV_SDK_DECL SKV_getFrameDataByTimestamp(SKV_FileHandle fileHandle, uint32_t streamID, uint64_t timeStamp, void* frameData);

/**
* \brief Gets the index of the frame specified by streamID and timestamp.
*
* Unlike get_frame_byte_count([...], index, [...]), it is not an error for the timestamp parameter p
* to be larger than the maximum timestamp of the stream.
* Instead, it will retrieve the frame index with the largest timestamp.
*
* \param[in] fileHandle A handle to the file
* \param[in] streamID The ID of the stream we are looking in.
* \param[in] timeStamp The timestamp of the frame that we want to retrieve.
*
* \return The frame index of the current frame.
*         `UINT_MAX` if the current frame is not set.
*
* \pre `fileHandle` may not be NULL
* \pre File must be in read mode
* \sa SKV_getFrameTimestamp
*/
SKV_API uint32_t SKV_SDK_DECL SKV_getFrameIndex(SKV_FileHandle fileHandle, uint32_t streamID, uint64_t timeStamp);

/**
* \brief Gets the timestamp of the frame specified by streamID, index.
*
* \param[in] fileHandle A handle to the file
* \param[in] streamID The ID of the stream we are looking in.
* \param[in] index The index of the frame that we want to retrieve.
* \return The time stamp of the current frame within the movie. \n
*         If the frame is not set, returns minimal 64-bit integer value (std::numeric_limits<uint64_t>::min()).
*
* \pre `fileHandle` may not be NULL
* \pre File must be in read mode
* \sa SKV_getFrameIndex
*/
SKV_API uint64_t SKV_SDK_DECL SKV_getFrameTimestamp(SKV_FileHandle fileHandle, uint32_t streamID, uint32_t index);

/**
 * \brief Get the number of custom buffers present in the file
 *
 * \param[in] fileHandle A handle to the file
 *
 * \return the number of custom buffers present in the file
 *
 * \pre `fileHandle` may not be NULL
 * \pre The file must be in read mode
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getCustomBufferCount( SKV_FileHandle fileHandle);

/**
 * \brief Get the name of the custom buffer
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] customBufferID The ID of the custom buffer for which to get the info
 * \param[out] info The info of the custom buffer
 *
 * \pre `fileHandle` may not be NULL
 * \pre The file must be in read mode
 * \pre `info` may not be NULL
 * \pre The custom buffer with given ID must exist in the movie
 *
 * \post The info struct will contain the requested custom buffer information
 */
SKV_API void SKV_SDK_DECL SKV_getCustomBufferInfo( SKV_FileHandle fileHandle, uint32_t customBufferID, SKV_CustomBufferInfo* info);

/**
 * \brief Check if a custom buffer exists in the movie
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] name The name of the buffer to look for
 *
 * \return `true` if the named buffer exists in the movie. \n
 *         `false` otherwise.
 *
 * \pre `fileHandle` may not be NULL
 * \pre `name` may not be NULL
 * \pre File must be in read mode
 */
SKV_API bool_t SKV_SDK_DECL SKV_isCustomBufferSupported( SKV_FileHandle fileHandle, const char* name );

/**
 * \brief Get the data size of a custom buffer
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] name The name of the buffer to look for
 *
 * \return The size (in bytes) of the named custom data
 *
 * \pre `fileHandle` may not be NULL
 * \pre `name` may not be NULL
 * \pre File must be in read mode
 */
SKV_API uint32_t SKV_SDK_DECL SKV_getCustomBufferSize( SKV_FileHandle fileHandle, const char* name );

/**
 * \brief Get the compression type of a custom buffer
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] name The name of the buffer to look for
 *
 * \return The type of compression used to compress the custom data (see \ref SKV_CompressionType).
 *
 * \pre `fileHandle` may not be NULL
 * \pre `name` may not be NULL
 * \pre File must be in read mode
 */
SKV_API SKV_CompressionType SKV_SDK_DECL SKV_getCustomBufferCompressionType( SKV_FileHandle fileHandle, const char* name );

/**
 * \brief Get the data of a custom buffer
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] name The name of the buffer to look for
 * \param[out] outputData The buffer that will contain the custom data
 *
 * \pre `fileHandle` may not be NULL
 * \pre `name` may not be NULL
 * \pre `outputData` may not be NULL
 * \pre File must be in read mode
 */
SKV_API void SKV_SDK_DECL SKV_getCustomBufferData( SKV_FileHandle fileHandle, const char* name, void* outputData );

/**
 * \brief Create a new movie
 *
 * \return `0` if no error \n
 *         `1` if file can't be created (is locked, ...) \n
 *         `2` if filehandle is invalid
 *
 * \pre `fileHandle` may not be NULL
 *
 * \post fileHandle points to the new movie
 * \post Movie file is opened in write mode
 */
SKV_API uint32_t SKV_SDK_DECL SKV_createMovie( const char* fileName, SKV_FileHandle *fileHandle );

/**
 * \brief Add an image stream in the file
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] info A struct containing information on the image stream
 *
 * \return The id of the new stream
 *
 * \pre `fileHandle` may not be NULL
 * \pre `info` may not be NULL
 * \pre The name in info must be a zero-terminated string
 * \pre The image type in info must be a valid type
 * \pre The compression type in `info` must be a valid type
 * \pre The image width and height in info must be greater than 0
 * \pre The file must be in write mode
 *
 * \post A new image stream with the returned ID is created based on the given stream info
*/
SKV_API uint32_t SKV_SDK_DECL SKV_addImageStream( SKV_FileHandle fileHandle, const SKV_ImageStreamInfo* info );

/**
 * \brief Add a custom stream in the file
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] info A struct containing information on the custom stream
 *
 * \return The id of the new stream
 *
 * \pre `fileHandle` may not be NULL
 * \pre `info` may not be NULL
 * \pre The name in info must be a zero-terminated string
 * \pre The compression type in info must be a valid type
 * \pre The file must be in write mode
 *
 * \post A new custom stream with the returned ID is created based on the given stream info
*/
SKV_API uint32_t SKV_SDK_DECL SKV_addCustomStream( SKV_FileHandle fileHandle, const SKV_CustomStreamInfo* info );

/**
 * \brief Add information about the device used for recording
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] deviceInfo A struct containing the info about the recording device
 *
 * \return `true` if the device info is successfully added \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre `deviceInfo` may not be NULL
 * \pre The vendorName in deviceInfo must be a zero-terminated string
 * \pre The cameraModel in deviceInfo must be a zero-terminated string
 * \pre The file must be in write mode
 *
 * \post The information about the recording device is added
 */
SKV_API bool_t SKV_SDK_DECL SKV_addDeviceInfo( SKV_FileHandle fileHandle, const SKV_DeviceInfo* deviceInfo );

/**
 * \brief Add pinhole model information to a stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream to which the pinhole model should be added
 * \param[in] pinholeModel A struct containing the info about the pinhole model
 *
 * \return `true` if the pinhole model info is successfully added \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The stream with given ID must exist in the file
 * \pre The file must be in write mode
 *
 * \post The pinhole model information is added for the given stream
 */
SKV_API bool_t SKV_SDK_DECL SKV_addPinholeModel( SKV_FileHandle fileHandle, uint32_t streamID, const SKV_PinholeModel* pinholeModel );

/**
 * \brief Add distortion model information to a stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream to which the distortion model should be added
 * \param[in] distortionModel A struct containing the info about the distortion model
 *
 * \return `true` if the distortion model info is successfully added \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The stream with given ID must exist in the file
 * \pre The file must be in write mode
 *
 * \post The distortion model information is added for the given stream
 */
SKV_API bool_t SKV_SDK_DECL SKV_addDistortionModel( SKV_FileHandle fileHandle, uint32_t streamID, const SKV_DistortionModel* distortionModel );

/**
 * \brief Add stereo transform information for a stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The ID of the stream to which the stereo transform should be added
 * \param[in] transform A struct containing the stereo transform information
 *
 * \return `true` if the stereo transform info is successfully added \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The stream with given ID must exist in the file
 * \pre The file must be in write mode
 *
 * \post The stereo transform is added for the two streams
 */
SKV_API bool_t SKV_SDK_DECL SKV_addStereoTransform( SKV_FileHandle fileHandle, uint32_t streamID, const SKV_StereoTransform* transform );

/**
 * \brief Add a frame to the given image stream
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] streamID The image stream in which to add the frame
 * \param[in] timeStamp The time stamp of the frame
 * \param[in] frameData Raw frame data buffer
 * \param[in] frameDataSize Size of the frameData buffer
 *
 * \return `true` if the frame is added to the stream \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre The stream with given ID must exist in the file
 * \pre The stream must be an image stream
 * \pre The file must be in write mode
 * \pre frameDataSize must match the size of a frame, determined by width, height, and pixelsize (found in the image stream info)
 *
 * \post A new frame is added to the given stream on the timestamp
 */
SKV_API bool_t SKV_SDK_DECL SKV_addFrame( SKV_FileHandle fileHandle, uint32_t streamID, uint64_t timeStamp, const void* frameData, uint32_t frameDataSize );

/**
 * \brief Add a custom data buffer to the movie
 *
 * \param[in] fileHandle A handle to the file
 * \param[in] name The name of the buffer
 * \param[in] bufferData Buffer to be written to the file
 * \param[in] bufferDataSize Size of the buffer
 * \param[in] compressionType The compression mode to use on the data
 *
 * \return `true` if the custom buffer is added tp the file \n
 *         `false` otherwise
 *
 * \pre `fileHandle` may not be NULL
 * \pre `name` may not be NULL
 * \pre `bufferData` may not be NULL
 * \pre The file must be in write mode
 *
 * \post A new data buffer is added
 */
SKV_API bool_t SKV_SDK_DECL SKV_addCustomBuffer( SKV_FileHandle fileHandle, const char* name, const void* bufferData, uint32_t bufferDataSize , SKV_CompressionType compressionType );

#ifdef __cplusplus
}
#endif

#endif //SOFTKINETIC_SKV_API_INCLUDED
