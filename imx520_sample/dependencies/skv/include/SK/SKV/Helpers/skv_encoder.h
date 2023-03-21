/***************************************************************************************/
//  SoftKinetic SKV library
//  Project Name      : SKV
//  Module Name		  : SKV helpers
//  Description       : C++ API for writing skv files
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

#pragma once

#ifndef SOFTKINETIC_SKV_ENCODER_INCLUDED
#define SOFTKINETIC_SKV_ENCODER_INCLUDED

#ifndef strcpy_no_warning
#ifdef _MSC_VER
#define strcpy_no_warning(A,B)\
	__pragma(warning(push))\
	__pragma(warning(disable: 4996))\
	strcpy(A,B)\
	__pragma(warning(pop))
#else
#define strcpy_no_warning strcpy
#endif
#endif

#include <SK/SKV/skv_api.h>
#include "skv_helper_types.h"

#include <stdexcept>
#include <cassert>
#include <cstring>
#include <string>
#include <chrono>

#include <sstream>


namespace SKV
{

	inline void copy4ChannelsTo3Channels(unsigned char* out_3channels, const unsigned  char* in_4channels, size_t width, size_t height);
	inline void copyColorFrameToSKVBuffer(uint8_t* outBuffer, const SKV::Image32& image, SKV_ImageType skvImageType);


	/**
	 * \brief SKV recorder used to write a movie.
	 */
	class Encoder
	{

	public:

		/**
		 * \brief Constructor which creates an SKV movie.
		 * \param[in] movie path of the skv to encode.
		 * \param[in] stream to write on the file.
		 * \param[in] Camera parameters of the encoded frames.
		 * \exception exception thrown if the movie path is invalid.
		 */
		Encoder(const std::string& moviePath, Stream streams, const CameraParameters &cameraParameters) :
			m_cameraParameters(cameraParameters),m_fileHandle(nullptr),m_depthStreamID(std::numeric_limits<uint32_t>::max()),
			m_depthfloatPlanesXYZStreamID(std::numeric_limits<uint32_t>::max()),
			m_colorStreamID(std::numeric_limits<uint32_t>::max()),m_colorLeftStreamID(std::numeric_limits<uint32_t>::max()),
			m_YUVStreamID(std::numeric_limits<uint32_t>::max()),m_YUVLeftStreamID(std::numeric_limits<uint32_t>::max()),
			m_confidenceStreamID(std::numeric_limits<uint32_t>::max()),
			m_accelerometerStreamID(std::numeric_limits<uint32_t>::max()),
			m_attitudematrixStreamID(std::numeric_limits<uint32_t>::max()),
			m_statusbitmaskStreamID(std::numeric_limits<uint32_t>::max()),
			m_colorRGB(nullptr),m_colorLeftRGB(nullptr),
			m_streams(streams)
		{
			uint32_t status = SKV_createMovie(moviePath.c_str(), &m_fileHandle);
			if (status != 0) { throw std::runtime_error("Cannot create movie"); }

			registerStreams();
			registerPinholeModel();
			registerDeviceInfo();

			m_startTimer = std::chrono::system_clock::now();
		}

		/**
		 * \brief Return the internal SKV structure.
		 * \return internal skv structure.
		 */
		SKV_FileHandle getSKVFileHandle() {return m_fileHandle;}

		/**
		 * \brief Destructor.
		 */
		~Encoder()
		{
			if(m_fileHandle != nullptr) { SKV_closeMovie(m_fileHandle); }

			if(m_colorRGB != nullptr) { free(m_colorRGB); }
			if(m_colorLeftRGB != nullptr) { free(m_colorLeftRGB); }
		}

		/**
		 * \brief Writes a frame.
		 * \param[in] Frame to encode.
		 */
		void writeFrame(const SKV::Frame &frame)
		{
			int depthSize = frame.depth.width*frame.depth.height*sizeof(*frame.depth.ptr);

			bool_t status=true;

			if (status && (m_streams & SKV::DEPTH)!=0)
			{
				// write depth frame in SKV
				status = SKV_addFrame(m_fileHandle, m_depthStreamID, frame.timestamp, frame.depth.ptr, depthSize);
				if (!status) { throw std::runtime_error("failed to write the depth image in SKV file"); }
			}

			if (status && (m_streams & SKV::DEPTH_FLOAT_PLANES_XYZ)!=0)
			{
				// write depth frame in SKV
				status = SKV_addFrame(m_fileHandle, m_depthfloatPlanesXYZStreamID, frame.timestamp, frame.depthFloatPlanesXYZ.ptr,  frame.depthFloatPlanesXYZ.width*frame.depthFloatPlanesXYZ.height*sizeof(*frame.depthFloatPlanesXYZ.ptr));
				if (!status) { throw std::runtime_error("failed to write the depth float planes XYZ image in SKV file"); }

			}

			if(status && m_cameraParameters.isYUVAvailable && (m_streams & SKV::YUV)!=0)
			{
				// write confidence frame in SKV
				status = SKV_addFrame(m_fileHandle, m_YUVStreamID, frame.timestamp,frame.YUV.ptr, frame.YUV.width * frame.YUV.height * sizeof(*frame.YUV.ptr));
				if (!status) { throw std::runtime_error("failed to write the YUV image in SKV file"); }
			}

			if(status && m_cameraParameters.isYUVLeftAvailable && (m_streams & SKV::YUV_LEFT)!=0)
			{
				// write confidence frame in SKV
				status = SKV_addFrame(m_fileHandle, m_YUVLeftStreamID, frame.timestamp,frame.YUVLeft.ptr, frame.YUVLeft.width * frame.YUV.height * sizeof(*frame.YUVLeft.ptr));
				if (!status) { throw std::runtime_error("failed to write the YUV left image in SKV file"); }
			}

			if(status && m_cameraParameters.isConfidenceAvailable)
			{
				int confidenceSize = frame.depth.width*frame.depth.height*sizeof(*frame.depth.ptr);
				// write confidence frame in SKV
				status = SKV_addFrame(m_fileHandle, m_confidenceStreamID, frame.timestamp,frame.confidence.ptr, confidenceSize);
				if (!status) { throw std::runtime_error("failed to write the confidence image in SKV file"); }
			}

			if(status && m_cameraParameters.isColorAvailable && (m_streams & SKV::COLOR)!=0 )
			{
				uint32_t width = frame.color.width;
				uint32_t height = frame.color.height;

				// checks if color stream exists
				assert(m_colorStreamID != std::numeric_limits<uint32_t>::max());
				copyColorFrameToSKVBuffer(m_colorRGB, frame.color, m_cameraParameters.colorImageType);

				// write color frame in SKV
				status = SKV_addFrame(m_fileHandle, m_colorStreamID, frame.timestamp, m_colorRGB, width * height * SKV_getPixelSize(m_cameraParameters.colorImageType));
				if (!status) { throw std::runtime_error("failed to write the color image in SKV file"); }
			}

			if(status && m_cameraParameters.isColorLeftAvailable && (m_streams & SKV::COLOR_LEFT)!=0)
			{
				uint32_t width = frame.colorLeft.width;
				uint32_t height = frame.colorLeft.height;

				// checks if left color stream exists
				assert(m_colorLeftStreamID != std::numeric_limits<uint32_t>::max());

				copyColorFrameToSKVBuffer(m_colorLeftRGB, frame.colorLeft, m_cameraParameters.colorImageType);


				// write color frame in SKV
				status = SKV_addFrame(m_fileHandle, m_colorLeftStreamID, frame.timestamp, m_colorLeftRGB, width * height * SKV_getPixelSize(m_cameraParameters.colorImageType));
				if (!status) { throw std::runtime_error("failed to write the left image frame in SKV file"); }
			}

			//write accelerometer data
			if(m_cameraParameters.isAccelerometerAvailable)
			{
				status = SKV_addFrame(m_fileHandle, m_accelerometerStreamID, frame.timestamp, &frame.accelerometer, sizeof(Vector3));
				if (!status) { throw std::runtime_error("failed to write the accelerometer in SKV file"); }
			}

			//write attitude data
			if(m_cameraParameters.isAttitudeMatrixAvailable)
			{
				status = SKV_addFrame(m_fileHandle, m_attitudematrixStreamID, frame.timestamp, &frame.attitudeMatrix, sizeof(float) * 16);
				if (!status) { throw std::runtime_error("failed to write the attitude matrix in SKV file"); }
			}

			//write attitude data
			if (true == m_cameraParameters.isStatusBitMaskAvailable)
			{
				status = SKV_addFrame(m_fileHandle, m_statusbitmaskStreamID, frame.timestamp, &frame.statusBitMask, sizeof(frame.statusBitMask));
				if (!status) { throw std::runtime_error("failed to write the status bit mask in SKV file"); }
			}
		}

		/**
		 * \brief Constructor which creates an SKV movie.
		 * \tparam Calibration structure
		 * \param[in] Calibration to encode.
		 */
 		template<typename Calibration>
		void registerCalibration(const Calibration &calibration)
		{
			if (m_cameraParameters.isCalibrationAvailable)
			{
				SKV_addCustomBuffer(m_fileHandle, "Calibration", &calibration, sizeof(Calibration), SKV_COMPRESSION_NONE);
			}
		}

	protected:



		/**
		 * \brief Encodes pinhole model.
		 */
		void registerPinholeModel()
		{
			SKV_PinholeModel depthPinholeModel;
			depthPinholeModel = m_cameraParameters.depthPinholeParams;

			SKV_PinholeModel colorPinholeModel;
			colorPinholeModel = m_cameraParameters.colorPinholeParams;


			if(m_cameraParameters.isDepthAvailable && (m_streams & SKV::DEPTH)!=0 )
			{
				SKV_addPinholeModel(m_fileHandle, m_depthStreamID, &depthPinholeModel);
				SKV_addDistortionModel(m_fileHandle, m_depthStreamID, &m_cameraParameters.depthDistortionParams);
			}

			if(m_cameraParameters.isDepthFloatPlanesXYZAvailable && (m_streams & SKV::DEPTH_FLOAT_PLANES_XYZ)!=0 )
			{
				SKV_addPinholeModel(m_fileHandle, m_depthfloatPlanesXYZStreamID, &depthPinholeModel);
				SKV_addDistortionModel(m_fileHandle, m_depthfloatPlanesXYZStreamID, &m_cameraParameters.depthDistortionParams);
			}

			if (m_cameraParameters.isConfidenceAvailable && (m_streams & SKV::CONFIDENCE)!=0)
			{
				// there is one sensor for both depth and confidence so the parameters should be the same
				SKV_addPinholeModel(m_fileHandle, m_confidenceStreamID, &depthPinholeModel);
				SKV_addDistortionModel(m_fileHandle, m_confidenceStreamID, &m_cameraParameters.depthDistortionParams);
			}

			if ((m_cameraParameters.isColorAvailable || m_cameraParameters.isColorLeftAvailable) && ((m_streams & SKV::COLOR)!=0 || (m_streams & SKV::COLOR_LEFT)!=0))
			{

				SKV_addDistortionModel(m_fileHandle, m_colorStreamID, &m_cameraParameters.colorDistortionParams);

				if(m_cameraParameters.isUvMappingAvailable)
				{

					SKV_addStereoTransform(m_fileHandle, m_colorStreamID, &m_cameraParameters.cameraToCameraXForm);
				}

				uint32_t colorStreamID = ((m_streams & SKV::COLOR)!=0)?m_colorStreamID:m_colorLeftStreamID;
				SKV_addPinholeModel(m_fileHandle, colorStreamID, &colorPinholeModel);

			}
			if ((m_cameraParameters.isYUVAvailable || m_cameraParameters.isYUVLeftAvailable) && ((m_streams & SKV::YUV)!=0 || (m_streams & SKV::YUV_LEFT)!=0))
			{
				SKV_addDistortionModel(m_fileHandle, m_YUVStreamID, &m_cameraParameters.colorDistortionParams);

				if(m_cameraParameters.isUvMappingAvailable)
				{
					SKV_addStereoTransform(m_fileHandle, m_YUVStreamID, &m_cameraParameters.cameraToCameraXForm);
				}

				uint32_t YUVStreamID = ((m_streams & SKV::YUV)!=0)?m_YUVStreamID:m_YUVLeftStreamID;
				SKV_addPinholeModel(m_fileHandle, YUVStreamID, &colorPinholeModel);

			}
		}

		/**
		 * \brief Encode Camera info.
		 */
		void registerDeviceInfo()
		{
			SKV_DeviceInfo deviceInfo;
			memset(&deviceInfo, 0, sizeof(deviceInfo));
			strcpy_no_warning(deviceInfo.cameraModel,m_cameraParameters.cameraModel.c_str());
			strcpy_no_warning(deviceInfo.vendorName, m_cameraParameters.vendorName.c_str());
			SKV_addDeviceInfo(m_fileHandle, &deviceInfo);
		}

		/**
		 * \brief Encode Streams info.
		 */
		void registerStreams()
		{

			if(m_cameraParameters.isDepthAvailable && (m_streams & SKV::DEPTH)!=0 )
			{
				SKV_ImageStreamInfo streamInfoDepth;
				strcpy_no_warning(streamInfoDepth.name, SKV_DEPTH_STREAM);
				streamInfoDepth.compression = SKV_COMPRESSION_NONE;
				streamInfoDepth.type = SKV_IMAGE_INT16;
				streamInfoDepth.width = m_cameraParameters.depthWidth;
				streamInfoDepth.height =  m_cameraParameters.depthHeight;
				m_depthStreamID = SKV_addImageStream(m_fileHandle, &streamInfoDepth);
			}

			if(m_cameraParameters.isDepthFloatPlanesXYZAvailable && (m_streams & SKV::DEPTH_FLOAT_PLANES_XYZ)!=0 )
			{
				SKV_ImageStreamInfo streamInfoDepthFloatPlanesXYZ;
				strcpy_no_warning(streamInfoDepthFloatPlanesXYZ.name, SKV_DEPTH_STREAM_FLOAT_PLANES_XYZ);
				streamInfoDepthFloatPlanesXYZ.compression = SKV_COMPRESSION_NONE;
				streamInfoDepthFloatPlanesXYZ.type = SKV_IMAGE_FLOAT;
				streamInfoDepthFloatPlanesXYZ.width = m_cameraParameters.depthWidth;
				streamInfoDepthFloatPlanesXYZ.height =  m_cameraParameters.depthHeight * 3 /* x y(depth) z */;
				m_depthfloatPlanesXYZStreamID = SKV_addImageStream(m_fileHandle, &streamInfoDepthFloatPlanesXYZ);
			}

			if(m_cameraParameters.isConfidenceAvailable && (m_streams & SKV::CONFIDENCE)!=0 )
			{
				SKV_ImageStreamInfo streamInfoConfidence;
				strcpy_no_warning(streamInfoConfidence.name, SKV_CONFIDENCE_STREAM);
				streamInfoConfidence.compression = SKV_COMPRESSION_NONE;
				streamInfoConfidence.type = SKV_IMAGE_INT16;
				streamInfoConfidence.width = m_cameraParameters.depthWidth;
				streamInfoConfidence.height = m_cameraParameters.depthHeight;
				assert(m_confidenceStreamID == std::numeric_limits<uint32_t>::max());
				m_confidenceStreamID = SKV_addImageStream(m_fileHandle, &streamInfoConfidence);
			}

			if(m_cameraParameters.isColorAvailable && (m_streams & SKV::COLOR)!=0)
			{
				SKV_ImageStreamInfo streamInfoColor;
				strcpy_no_warning(streamInfoColor.name, SKV_COLOR_STREAM);
				streamInfoColor.compression = m_cameraParameters.colorCompressionType;
				streamInfoColor.type = m_cameraParameters.colorImageType;
				streamInfoColor.width = m_cameraParameters.colorWidth;
				streamInfoColor.height = m_cameraParameters.colorHeight;
				m_colorStreamID = SKV_addImageStream(m_fileHandle, &streamInfoColor);
				m_colorRGB = (uint8_t*) malloc(streamInfoColor.width * streamInfoColor.height * SKV_getPixelSize(streamInfoColor.type));
			}

			if(m_cameraParameters.isColorLeftAvailable && (m_streams & SKV::COLOR_LEFT)!=0)
			{
				SKV_ImageStreamInfo streamInfoColorLeft;
				strcpy_no_warning(streamInfoColorLeft.name, SKV_COLOR_STREAM_LEFT);
				streamInfoColorLeft.compression = m_cameraParameters.colorCompressionType;
				streamInfoColorLeft.type = m_cameraParameters.colorImageType;
				streamInfoColorLeft.width = m_cameraParameters.colorWidth;
				streamInfoColorLeft.height = m_cameraParameters.colorHeight;
				m_colorLeftStreamID = SKV_addImageStream(m_fileHandle, &streamInfoColorLeft);
				m_colorLeftRGB = (uint8_t*)malloc(streamInfoColorLeft.width * streamInfoColorLeft.height * SKV_getPixelSize(streamInfoColorLeft.type));
			}
			if(m_cameraParameters.isYUVAvailable && (m_streams & SKV::YUV)!=0)
			{
				SKV_ImageStreamInfo streamInfoYUV;
				strcpy_no_warning(streamInfoYUV.name, SKV_YUV_STREAM);
				streamInfoYUV.compression = SKV_COMPRESSION_NONE;
				streamInfoYUV.type = SKV_IMAGE_YUV16;
				streamInfoYUV.width = m_cameraParameters.colorWidth;
				streamInfoYUV.height = m_cameraParameters.colorHeight;
				m_YUVStreamID = SKV_addImageStream(m_fileHandle, &streamInfoYUV);

			}

			if(m_cameraParameters.isYUVLeftAvailable && (m_streams & SKV::YUV_LEFT)!=0)
			{
				SKV_ImageStreamInfo streamInfoYUVLeft;
				strcpy_no_warning(streamInfoYUVLeft.name, SKV_YUV_STREAM_LEFT);
				streamInfoYUVLeft.compression = SKV_COMPRESSION_NONE;
				streamInfoYUVLeft.type = SKV_IMAGE_YUV16;
				streamInfoYUVLeft.width = m_cameraParameters.colorWidth;
				streamInfoYUVLeft.height = m_cameraParameters.colorHeight;
				m_YUVLeftStreamID = SKV_addImageStream(m_fileHandle, &streamInfoYUVLeft);

			}
			if(m_cameraParameters.isAccelerometerAvailable)
			{
				SKV_CustomStreamInfo accelerometerStreamInfo;
				strcpy_no_warning(accelerometerStreamInfo.name, SKV_ACCELEROMETER_STREAM);
				accelerometerStreamInfo.compression = SKV_COMPRESSION_NONE;
				m_accelerometerStreamID = SKV_addCustomStream(m_fileHandle, &accelerometerStreamInfo);
			}
			if(m_cameraParameters.isAttitudeMatrixAvailable)
			{
				SKV_CustomStreamInfo atttitudeStreamInfo;
				strcpy_no_warning(atttitudeStreamInfo.name, SKV_ATTITUDEMATRIX_STREAM);
				atttitudeStreamInfo.compression = SKV_COMPRESSION_NONE;
				m_attitudematrixStreamID = SKV_addCustomStream(m_fileHandle, &atttitudeStreamInfo);
			}
			if (true == m_cameraParameters.isStatusBitMaskAvailable)
			{
				SKV_CustomStreamInfo statusbitmaskStreamInfo;
				strcpy_no_warning(statusbitmaskStreamInfo.name, SKV_STATUSBITMASK_STREAM);
				statusbitmaskStreamInfo.compression = SKV_COMPRESSION_NONE;
				m_statusbitmaskStreamID = SKV_addCustomStream(m_fileHandle, &statusbitmaskStreamInfo);
			}
		}

		CameraParameters m_cameraParameters;
		SKV_FileHandle m_fileHandle;
		uint32_t m_depthStreamID, m_depthfloatPlanesXYZStreamID, m_colorStreamID, m_colorLeftStreamID, m_YUVStreamID, m_YUVLeftStreamID, m_confidenceStreamID, m_accelerometerStreamID, m_attitudematrixStreamID, m_statusbitmaskStreamID;
		uint8_t *m_colorRGB;
		uint8_t *m_colorLeftRGB;
		std::chrono::time_point<std::chrono::system_clock> m_startTimer;
		SKV::Stream m_streams;
	};


	void copy4ChannelsTo3Channels(unsigned char* out_3channels, const unsigned char* in_4channels, size_t width, size_t height)
	{
		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t w = 0; w < width; w++)
			{
				out_3channels[3 * (width*y + w) + 0] = in_4channels[4 * (width*y + w)];
				out_3channels[3 * (width*y + w) + 1] = in_4channels[4 * (width*y + w) + 1];
				out_3channels[3 * (width*y + w) + 2] = in_4channels[4 * (width*y + w) + 2];
			}
		}
	}


	void copyColorFrameToSKVBuffer(uint8_t* outBuffer, const SKV::Image32& image, SKV_ImageType skvImageType)
	{
		uint8_t *colorRaw = reinterpret_cast<uint8_t *>(image.ptr);
		size_t width = image.width, height = image.height;

		if (skvImageType == SKV_IMAGE_RGBA32 || skvImageType == SKV_IMAGE_BGRA32)
		{
			memcpy(outBuffer, colorRaw, width*height * 4);
		}
		else if (skvImageType == SKV_IMAGE_RGB24 || skvImageType == SKV_IMAGE_BGR24)
		{
			copy4ChannelsTo3Channels(outBuffer, colorRaw, width, height);
		}
		else
		{
			throw std::runtime_error("Unsupported image type for color image");
		}
	}
}


#endif
