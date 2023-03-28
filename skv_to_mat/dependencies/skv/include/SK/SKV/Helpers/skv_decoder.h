/***************************************************************************************/
//  SoftKinetic SKV library
//  Project Name      : SKV
//  Module Name		  : SKV helpers
//  Description       : C++ API for reading skv files
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

#include <SK/SKV/skv_api.h>
#include "skv_helper_types.h"
#include <cassert>
#include <thread>
#include <string>
#include <sstream>


#ifndef SOFTKINETIC_SKV_DECODER_INCLUDED
#define SOFTKINETIC_SKV_DECODER_INCLUDED

namespace SKV
{
	// allocate an aligned bloc of memory
	inline void* skv_aligned_malloc(size_t size, size_t alignment)
	{
		const size_t pointerSize = sizeof(void*);
		const size_t requestedSize = size + alignment - 1 + pointerSize;
		void* raw = malloc(requestedSize);
		void* start = (char*)raw + pointerSize;
		void* aligned = (void*)(((size_t)((char*)start+alignment-1)) & ~(alignment-1));
		*(void**)((char*)aligned-pointerSize) = raw;
		return aligned;
	}

	// free an aligned bloc of memory
	inline void skv_aligned_free(void* aligned)
	{
		if (aligned) {
			void* raw = *(void**)((char*)aligned-sizeof(void*));
			free(raw);
		}
	}

	inline void copySKVBufferToColorImage(SKV::Image32& image, const uint8_t* inSKVBuffer, SKV_ImageType skvImageType);


	/**
	 * \brief SKV decoder used to read a movie.
	 */
	class Decoder
	{
	public:

		/**
		 * \brief Constructor which loads a SKV movie.
		 * \param[in] movie path.
		 * \param[in] stream to retrieve from the movie.
		 * \param[in] Indicates whether to loop the movie.
		 * \param[in] Allocator to use(optional).
		 * \exception exception thrown if the movie path is invalid.
		 */
		inline Decoder(const std::string& moviePath, Stream streams, bool infiniteLoop, SKV_Allocator* allocator=nullptr);

		/**
		 * \brief Destructor
		 */
		inline ~Decoder();

		/**
		 * \brief Indicates whether the we are at the end of the movie.
		 * \return status.
		 */
		 bool isRunning() const { return m_currentFrameId < m_totalFramecount;}

		/**
		 * \brief Returns the camera parameters.
		 * \return Camera parameters..
		 */
		 const CameraParameters& getCameraParameters() const { return m_cameraParameters; }

		/**
		 * \brief Reads the next frame.
		 */
		inline void readNextFrame();

		/**
		 * \brief Returns the color image coming from the current frame.
		 *        If this movie has been recorded with a stereo camera,this image
		 *		  is coming from the right lens.
		 * \return Color image in BGRA format.
		 */
		inline const Image32& getColorImage() const;

		/**
		 * \brief Returns the color image coming from the current frame.
		 *        If this movie has been recorded with a stereo camera,this image
		 *		  is coming from the left lens.
		 * \return Color image in BGRA format.
		 */
		inline const Image32& getColorLeftImage() const;

		/**
		 * \brief Returns the color image coming from the current frame.
		 *        If this movie has been recorded with a stereo camera,this image
		 *		  is coming from the right lens.
		 * \return Color image in YUV format.
		 */
		inline const Image16& getYUVImage() const;

		/**
		 * \brief Returns the color image coming from the current frame.
		 *        If this movie has been recorded with a stereo camera,this image
		 *		  is coming from the left lens.
		 * \return Color image in YUV format.
		 */
		inline const Image16& getYUVLeftImage() const;

		/**
		 * \brief Returns the depth image coming from the current frame.
		 *        Invalid pixels have the values 32001.
		 * \return Depth image where each value is expressed in mm.
		 */
		inline const Image16& getDepthImage() const;

		/**
		 * \brief Returns the X Y(depth) Z images coming from the current frame.
		 *        Invalid pixels have the values -1.0 and saturated pixels have the value -2.0 on the Y(depth) image.
		 *		  Width = depth image width   Height = 3 x depth image height
		 * \return The X Y Z image pasted below each other where each value is expressed in mm.
		 */
		inline const Image32Float& getDepthFloatPlanesXYZImage() const;

		/**
		 * \brief Returns the confidence image coming from the current frame.
		 *        This image provides the confidence of each value of the depth image.
		 * \return Confidence image.
		 */
		inline const Image16& getConfidenceImage() const;

		/**
		 * \brief Returns the accelerometer value coming from the current frame.
		 * \return 3-axis acceleration expressed in G force unit.
		 */
		inline const Vector3& getAccelerometerValue() const;

		/**
		 * \brief Returns the attitude matrix coming from the current frame.
		 * \return 4 by 4 matrix attitude matrix. It represent the global orientation of the camera. A translation can be included.
		 */
		inline const Matrix4x4& getAttitudeMatrix() const;

		/**
		* \brief Returns the status bit mask coming from the current frame.
		* \returns a 32-bit integer. Application defined bit mask status can be stored in here.
		*/
		inline const uint32_t& getStatusBitMask() const;

		/**
		 * \brief Returns the timestamp value coming from the current frame.
		 * \return timestamp value.
		 */
		uint64_t getTimeStamp() const { return m_lastFrameTimeStamp; }

		/**
		 * \brief Returns the frame id value coming from the current frame.
		 * \return frame id value. Starts at 1.
		 */
		uint32_t getFrameId() const { return m_currentFrameId; }

		/**
		 * \brief Returns the total number of frames of the movie.
		 * \return count frames.
		 */
		size_t getCountFrames() const { return m_totalFramecount; }

		/**
		* \brief Returns the average framerate for the streams of the main stream (i.e. the stream with highest framerate)
		* \return average framerate, in image/s.
		*/
		float getAverageFrameRate() const { return m_frameRate; }

	  protected:

		/**
		 * \brief Retrieves Camera info related to a specifc stream (depth, color,...)
		 * \param[in] stream id.
		 * \param[out] pinhole model.
		 * \param[out] lens distortion model.
		 * \param[out] stereo transform.
		 * \param[out] average framerate (frames / s).
		 */
		inline void updateStreamInfo( unsigned int streamID, SKV_PinholeModel* pinholeParams, SKV_DistortionModel* distortionParams, SKV_StereoTransform* stereoParams);

		/**
		 * \brief Retrieves image data of a specific stream.
		 * \param[in] stream id.
		 * \param[in] timestamp related to a specif frame.
		 * \param[out] image buffer to fill.
		 */
		inline void readImageStream(uint32_t streamID, uint64_t timeStamp, void* imagePtr);

		/**
		 * \brief Internal data storing camera infos.
		 */
		struct SKVData{
			SKVData():
				depthStreamID(0),depthfloatPlanesXYZStreamID(0),confidenceStreamID(0),colorStreamID(0),colorLeftStreamID(0),YUVStreamID(0),YUVLeftStreamID(0),accelerometerStreamID(0),attitudematrixStreamID(0), statusbitmaskStreamID(0),
				mainStreamID(-1),hasDepth(false),hasDepthFloatPlanesXYZ(false),hasColor(false),hasColorLeft(false),hasConfidence(false),hasAccelerometer(false),hasAttitudeMatrix(false),hasStatusBitMask(false),hasYUV(false),hasYUVLeft(false),
				colorImagePtr(nullptr),colorLeftImagePtr(nullptr)
			{}

			SKV_FileHandle fileHandle;
			SKV_ImageStreamInfo depthStreamInfo, depthfloatPlanesXYZStreamInfo, confidenceStreamInfo, colorStreamInfo, colorLeftStreamInfo, YUVStreamInfo, YUVLeftStreamInfo;
			SKV_CustomStreamInfo accelerometerStreamInfo, attitudematrixStreamInfo, statusbitmaskStreamInfo;
			uint32_t depthStreamID; // the stream on which the replay is synchronized
			uint32_t depthfloatPlanesXYZStreamID; // the stream on which the replay is synchronized
			uint32_t confidenceStreamID; // the stream on which the replay is synchronized
			uint32_t colorStreamID;
			uint32_t colorLeftStreamID;
			uint32_t YUVStreamID;
			uint32_t YUVLeftStreamID;
			uint32_t accelerometerStreamID;
			uint32_t attitudematrixStreamID;
			uint32_t statusbitmaskStreamID;
			int32_t mainStreamID;
			SKV_PinholeModel depthPinholeParams, colorPinholeParams, YUVPinholeParams;
			SKV_DistortionModel depthDistortionParams, colorDistortionParams;
			SKV_StereoTransform cameraToCameraXForm;
			bool hasDepth;
			bool hasDepthFloatPlanesXYZ;
			bool hasColor;
			bool hasColorLeft;
			bool hasYUV;
			bool hasYUVLeft;
			bool hasConfidence;
			bool hasAccelerometer;
			bool hasAttitudeMatrix;
			bool hasStatusBitMask;

			// these can be 3 or 4-channels images, depending on what was recorded in the SKV file.
			// use colorStreamInfo and colorLeftStreamInfo to know.
			uint8_t* colorImagePtr;
			uint8_t* colorLeftImagePtr;

		};

		CameraParameters m_cameraParameters;
		bool m_infiniteLoop;
		SKVData m_skvData;

		uint32_t m_currentFrameId;
		uint64_t m_lastFrameTimeStamp;
		uint64_t m_timeSinceLastFrame;
		uint32_t m_totalFramecount;

		double m_timeBetweenFrames;
		float m_frameRate;

		bool m_readColor;
		bool m_readColorLeft;
		bool m_readConfidence;
		bool m_readDepth;
		bool m_readDepthFloatPlanesXYZ;

		Image32 m_colorImage;
		Image32 m_colorLeftImage;
		Image16 m_YUVImage;
		Image16 m_YUVLeftImage;
		Image16 m_depthImage;
		Image32Float m_depthFloatPlanesXYZImage;
		Image16 m_confidenceImage;

		Vector3 m_accelerometerValue;
		Matrix4x4 m_attitudeMatrix;
		uint32_t m_statusBitMask;
		bool m_readYUV;
		bool m_readYUVLeft;


		SKV_Allocator m_allocator;
	};

	//////////////////////////////////////////////////////////////////////////
	Decoder::Decoder(const std::string& moviePath, Stream streams, bool infiniteLoop, SKV_Allocator* allocator):
		m_infiniteLoop(infiniteLoop),
		m_currentFrameId(0),
		m_lastFrameTimeStamp(0),
		m_statusBitMask(0),
		m_frameRate(-std::numeric_limits<float>::max()),
		m_totalFramecount(-std::numeric_limits<int>::max())
	{
		m_readColor = (streams & COLOR) != 0;
		m_readColorLeft = (streams & COLOR_LEFT) != 0;
		m_readConfidence = (streams & CONFIDENCE) != 0;
		m_readDepth = (streams & DEPTH) != 0;
		m_readDepthFloatPlanesXYZ = (streams & DEPTH_FLOAT_PLANES_XYZ) != 0;
		m_readYUV = (streams & YUV) != 0;
		m_readYUVLeft = (streams & YUV_LEFT) != 0;

		m_skvData.colorImagePtr = nullptr;
		uint32_t success = SKV_openMovie(moviePath.c_str(), &m_skvData.fileHandle, SKV_READ_ONLY);
		if(success!=0)
		{
			throw std::runtime_error(std::string("Could not open movie file: ")+moviePath);
		}
		if (allocator==nullptr)
		{
			m_allocator.allocate = &skv_aligned_malloc;
			m_allocator.deallocate = &skv_aligned_free;
		}
		else
		{
			m_allocator =  *allocator;
		}

		SKV_DeviceInfo device_info;
		SKV_getDeviceInfo(m_skvData.fileHandle,&device_info);
		m_cameraParameters.vendorName = device_info.vendorName;
		m_cameraParameters.cameraModel = device_info.cameraModel;

		uint32_t streamCount;
		streamCount = SKV_getStreamCount(m_skvData.fileHandle);

		m_cameraParameters.isDepthAvailable = false;

		for (unsigned int i=0; i<streamCount; i++)
		{
			SKV_StreamType streamType = SKV_getStreamType(m_skvData.fileHandle, i);
			if(streamType == SKV_STREAM_IMAGE)
			{
				SKV_ImageStreamInfo tmp_info;
				SKV_getImageStreamInfo(m_skvData.fileHandle, i, &tmp_info);
				if(std::string(tmp_info.name)==SKV_DEPTH_STREAM && m_readDepth)
				{
					m_skvData.depthStreamInfo = tmp_info;
					m_skvData.depthStreamID = i;
					m_skvData.hasDepth = m_cameraParameters.isDepthAvailable = true;
					m_cameraParameters.depthWidth =  m_skvData.depthStreamInfo.width;
					m_cameraParameters.depthHeight = m_skvData.depthStreamInfo.height;
					updateStreamInfo(i, &m_skvData.depthPinholeParams, &m_skvData.depthDistortionParams, &m_skvData.cameraToCameraXForm);
					m_cameraParameters.fovx = m_cameraParameters.depthPinholeParams.fx = m_skvData.depthPinholeParams.fx;
					m_cameraParameters.fovy = m_cameraParameters.depthPinholeParams.fy = m_skvData.depthPinholeParams.fy;
					m_cameraParameters.depthPinholeParams = m_skvData.depthPinholeParams;

					m_depthImage = SKV::Image16(
						m_skvData.depthStreamInfo.width,
						m_skvData.depthStreamInfo.height,
						(int16_t*)m_allocator.allocate(sizeof(int16_t)*m_skvData.depthStreamInfo.width*m_skvData.depthStreamInfo.height, 16)
					);
				}
				else if(std::string(tmp_info.name)==SKV_DEPTH_STREAM_FLOAT_PLANES_XYZ && m_readDepthFloatPlanesXYZ)
				{
					m_skvData.depthfloatPlanesXYZStreamInfo = tmp_info;
					m_skvData.depthfloatPlanesXYZStreamID = i;
					m_skvData.hasDepthFloatPlanesXYZ = m_cameraParameters.isDepthFloatPlanesXYZAvailable = true;
					m_cameraParameters.depthWidth =  m_skvData.depthfloatPlanesXYZStreamInfo.width;
					m_cameraParameters.depthHeight = m_skvData.depthfloatPlanesXYZStreamInfo.height / 3;
					updateStreamInfo(i, &m_skvData.depthPinholeParams, &m_skvData.depthDistortionParams, &m_skvData.cameraToCameraXForm);
					m_cameraParameters.fovx = m_cameraParameters.depthPinholeParams.fx = m_skvData.depthPinholeParams.fx;
					m_cameraParameters.fovy = m_cameraParameters.depthPinholeParams.fy = m_skvData.depthPinholeParams.fy;
					m_cameraParameters.depthPinholeParams = m_skvData.depthPinholeParams;

					m_depthFloatPlanesXYZImage = SKV::Image32Float(
						m_skvData.depthfloatPlanesXYZStreamInfo.width,
						m_skvData.depthfloatPlanesXYZStreamInfo.height,
						1,
						(float*)m_allocator.allocate(sizeof(float)*m_skvData.depthfloatPlanesXYZStreamInfo.width*m_skvData.depthfloatPlanesXYZStreamInfo.height, 16)
					);

				}
				else if(std::string(tmp_info.name)==SKV_CONFIDENCE_STREAM && m_readConfidence)
				{
					m_skvData.confidenceStreamInfo = tmp_info;
					m_skvData.confidenceStreamID = i;
					m_skvData.hasConfidence = m_cameraParameters.isConfidenceAvailable = true;
					m_confidenceImage.ptr = (int16_t*) m_allocator.allocate(sizeof(int16_t)*m_skvData.confidenceStreamInfo.width*m_skvData.confidenceStreamInfo.height, 16);
					m_confidenceImage.width = m_skvData.confidenceStreamInfo.width;
					m_confidenceImage.height = m_skvData.confidenceStreamInfo.height;
				}
				else if(std::string(tmp_info.name)==SKV_COLOR_STREAM && m_readColor)
				{
					m_skvData.colorStreamInfo = tmp_info;
					m_skvData.colorStreamID = i;
					m_skvData.hasColor = m_cameraParameters.isColorAvailable = true;
					m_cameraParameters.colorImageType = tmp_info.type;
					m_cameraParameters.colorWidth =  m_skvData.colorStreamInfo.width;
					m_cameraParameters.colorHeight = m_skvData.colorStreamInfo.height;
					m_cameraParameters.colorChannelCount = SKV_getPixelSize(m_skvData.colorStreamInfo.type);
					updateStreamInfo(i, &m_skvData.colorPinholeParams, &m_skvData.colorDistortionParams, &m_skvData.cameraToCameraXForm);

					//allocate buffer
					m_colorImage.ptr = (int32_t*) m_allocator.allocate(sizeof(int32_t)*m_skvData.colorStreamInfo.width*m_skvData.colorStreamInfo.height, 16);
					m_colorImage.width = m_skvData.colorStreamInfo.width;
					m_colorImage.height = m_skvData.colorStreamInfo.height;
					m_colorImage.countChannel = 4;

					m_skvData.colorImagePtr = (uint8_t*) m_allocator.allocate(m_cameraParameters.colorChannelCount * m_skvData.colorStreamInfo.width*m_skvData.colorStreamInfo.height, 16);
				}
				else if(std::string(tmp_info.name)==SKV_COLOR_STREAM_LEFT && m_readColorLeft)
				{
					m_skvData.colorLeftStreamInfo = tmp_info;
					m_skvData.colorLeftStreamID = i;
					m_skvData.hasColorLeft = m_cameraParameters.isColorLeftAvailable = true;
					m_cameraParameters.colorImageType = tmp_info.type;
					m_cameraParameters.colorWidth =  m_skvData.colorStreamInfo.width;
					m_cameraParameters.colorHeight = m_skvData.colorStreamInfo.height;
					m_cameraParameters.colorChannelCount = SKV_getPixelSize(m_skvData.colorLeftStreamInfo.type);;
					m_colorLeftImage.ptr = (int32_t*) m_allocator.allocate(sizeof(int32_t)*m_skvData.colorLeftStreamInfo.width*m_skvData.colorLeftStreamInfo.height, 16);
					m_colorLeftImage.width = m_skvData.colorLeftStreamInfo.width;
					m_colorLeftImage.height = m_skvData.colorLeftStreamInfo.height;
					m_colorLeftImage.countChannel = 4;

					m_skvData.colorLeftImagePtr = (uint8_t*)m_allocator.allocate(m_cameraParameters.colorChannelCount * m_skvData.colorLeftStreamInfo.width*m_skvData.colorLeftStreamInfo.height, 16);
				}
				else if(std::string(tmp_info.name)==SKV_YUV_STREAM && m_readYUV)
				{
					m_skvData.YUVStreamInfo = tmp_info;
					m_skvData.YUVStreamID = i;
					m_skvData.hasYUV = true;
					m_cameraParameters.isYUVAvailable = m_cameraParameters.isYUVAvailable = true;
					m_cameraParameters.colorWidth =  m_skvData.YUVStreamInfo.width;
					m_cameraParameters.colorHeight = m_skvData.YUVStreamInfo.height;
					m_cameraParameters.colorChannelCount = 1;
					updateStreamInfo(i, &m_skvData.YUVPinholeParams, nullptr, nullptr);

					//allocate buffer
					m_YUVImage.ptr = (int16_t*) m_allocator.allocate(sizeof(int16_t)*m_skvData.YUVStreamInfo.width*m_skvData.YUVStreamInfo.height, 256);
					m_YUVImage.width = m_skvData.YUVStreamInfo.width;
					m_YUVImage.height = m_skvData.YUVStreamInfo.height;
				}
				else if(std::string(tmp_info.name)==SKV_YUV_STREAM_LEFT && m_readYUVLeft)
				{
					m_skvData.YUVLeftStreamInfo = tmp_info;
					m_skvData.YUVLeftStreamID = i;
					m_skvData.hasYUVLeft = m_cameraParameters.isYUVLeftAvailable = true;
					m_cameraParameters.colorWidth =  m_skvData.YUVLeftStreamInfo.width;
					m_cameraParameters.colorHeight = m_skvData.YUVLeftStreamInfo.height;
					m_cameraParameters.colorChannelCount = 1;
					m_YUVLeftImage.ptr = (int16_t*) m_allocator.allocate(sizeof(int16_t)*m_skvData.YUVLeftStreamInfo.width*m_skvData.YUVLeftStreamInfo.height, 256);
					m_YUVLeftImage.width = m_skvData.YUVLeftStreamInfo.width;
					m_YUVLeftImage.height = m_skvData.YUVLeftStreamInfo.height;
				}

			}
			else if(streamType == SKV_STREAM_CUSTOM)
			{
				SKV_CustomStreamInfo tmp_info;
				SKV_getCustomStreamInfo(m_skvData.fileHandle, i, &tmp_info);
				if(std::string(tmp_info.name)==SKV_ACCELEROMETER_STREAM)
				{
					m_skvData.accelerometerStreamInfo = tmp_info;
					m_skvData.accelerometerStreamID = i;
					m_skvData.hasAccelerometer = m_cameraParameters.isAccelerometerAvailable = true;
				}
				else if(std::string(tmp_info.name)==SKV_ATTITUDEMATRIX_STREAM)
				{
					m_skvData.attitudematrixStreamInfo = tmp_info;
					m_skvData.attitudematrixStreamID = i;
					m_skvData.hasAttitudeMatrix = m_cameraParameters.isAttitudeMatrixAvailable = true;
				}
				else if (std::string(tmp_info.name) == SKV_STATUSBITMASK_STREAM)
				{
					m_skvData.statusbitmaskStreamInfo = tmp_info;
					m_skvData.statusbitmaskStreamID = i;
					m_skvData.hasStatusBitMask = m_cameraParameters.isStatusBitMaskAvailable = true;
				}
			}
		}

		m_cameraParameters.depthDistortionParams = m_skvData.depthDistortionParams;
		m_cameraParameters.depthPinholeParams = m_skvData.depthPinholeParams;
		m_cameraParameters.colorDistortionParams = m_skvData.colorDistortionParams;
		m_cameraParameters.colorPinholeParams = m_skvData.colorPinholeParams;
		m_cameraParameters.cameraToCameraXForm = m_skvData.cameraToCameraXForm;

		if (m_skvData.mainStreamID == -1)
		{
			throw std::runtime_error("No image stream. Aborting");
		}

		m_timeSinceLastFrame = 0;


	}

	//////////////////////////////////////////////////////////////////////////
	Decoder::~Decoder()
	{
		if (m_depthImage.ptr!=nullptr)
		{
			m_allocator.deallocate(m_depthImage.ptr);
		}
		if (m_depthFloatPlanesXYZImage.ptr!=nullptr)
		{
			m_allocator.deallocate(m_depthFloatPlanesXYZImage.ptr);
		}
		if (m_colorImage.ptr!=nullptr)
		{
			m_allocator.deallocate(m_colorImage.ptr);
		}
		if (m_colorLeftImage.ptr!=nullptr)
		{
			m_allocator.deallocate(m_colorLeftImage.ptr);
		}
		if (m_confidenceImage.ptr!=nullptr)
		{
			m_allocator.deallocate(m_confidenceImage.ptr);
		}
		if (m_YUVImage.ptr!=nullptr)
		{
			m_allocator.deallocate(m_YUVImage.ptr);
		}
		if (m_YUVLeftImage.ptr!=nullptr)
		{
			m_allocator.deallocate(m_YUVLeftImage.ptr);
		}
		if (m_skvData.colorImagePtr)
		{
			m_allocator.deallocate(m_skvData.colorImagePtr);
		}
		if (m_skvData.colorLeftImagePtr)
		{
			m_allocator.deallocate(m_skvData.colorLeftImagePtr);
		}

		if (!SKV_closeMovie(m_skvData.fileHandle))
		{
			assert(!"Could not close movie.");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Decoder::readNextFrame()
	{
		if (m_currentFrameId==m_totalFramecount) throw std::runtime_error("Reached the end of the movie");

		//process main stream first to get timestamp
		if(!SKV_seekFrameByIndex(m_skvData.fileHandle, m_skvData.mainStreamID, m_currentFrameId))
		{
			throw std::runtime_error("Could not seek to frame " +std::to_string(m_currentFrameId));
		}
		uint64_t timeStamp = SKV_getCurrentFrameTimeStamp(m_skvData.fileHandle);
		m_timeSinceLastFrame = timeStamp - m_lastFrameTimeStamp;
		m_lastFrameTimeStamp = timeStamp;


		if (m_readDepth && m_skvData.hasDepth)
		{
			readImageStream(m_skvData.depthStreamID, timeStamp, m_depthImage.ptr);
		}

		if (m_readDepthFloatPlanesXYZ && m_skvData.hasDepthFloatPlanesXYZ)
		{
			readImageStream(m_skvData.depthfloatPlanesXYZStreamID, timeStamp, m_depthFloatPlanesXYZImage.ptr);
		}

		if (m_readConfidence && m_skvData.hasConfidence)
		{
			readImageStream(m_skvData.confidenceStreamID, timeStamp, m_confidenceImage.ptr);
		}

		if(m_readColor && m_skvData.hasColor)
		{
			readImageStream(m_skvData.colorStreamID, timeStamp, m_skvData.colorImagePtr);
			copySKVBufferToColorImage(m_colorImage, m_skvData.colorImagePtr, m_cameraParameters.colorImageType);
		}

		// Color right image (stereo)
		if(m_readColorLeft && m_skvData.hasColorLeft)
		{
			readImageStream(m_skvData.colorLeftStreamID, timeStamp, m_skvData.colorLeftImagePtr);
			copySKVBufferToColorImage(m_colorLeftImage, m_skvData.colorLeftImagePtr, m_cameraParameters.colorImageType);
		}

		if (m_readYUV && m_skvData.hasYUV)
		{
			readImageStream(m_skvData.YUVStreamID, timeStamp, m_YUVImage.ptr);
		}

		if (m_readYUVLeft && m_skvData.hasYUVLeft)
		{
			readImageStream(m_skvData.YUVLeftStreamID, timeStamp, m_YUVLeftImage.ptr);
		}


		//accelerometer data
		if(m_skvData.hasAccelerometer)
		{
			readImageStream(m_skvData.accelerometerStreamID, timeStamp, &(m_accelerometerValue));
		}

		// attitude data
		if(m_skvData.hasAttitudeMatrix)
		{
			readImageStream(m_skvData.attitudematrixStreamID, timeStamp, &(m_attitudeMatrix));
		}

		// status bit mask
		if (m_skvData.hasStatusBitMask)
		{
			readImageStream(m_skvData.statusbitmaskStreamID, timeStamp, &(m_statusBitMask));
		}

		m_currentFrameId++;

		if(m_infiniteLoop && m_currentFrameId==m_totalFramecount)
		{
			m_currentFrameId=0;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Decoder::updateStreamInfo( unsigned int streamID, SKV_PinholeModel* pinholeParams, SKV_DistortionModel* distortionParams, SKV_StereoTransform* stereoParams)
	{
		if(SKV_isPinholeModelSupported(m_skvData.fileHandle, streamID))
		{
			SKV_getPinholeModel(m_skvData.fileHandle, streamID, pinholeParams);
		}

		if(distortionParams!= nullptr && SKV_isDistortionModelSupported(m_skvData.fileHandle, streamID))
		{
			SKV_getDistortionModel(m_skvData.fileHandle, streamID, distortionParams);
		}

		if(stereoParams!=nullptr && SKV_isStereoTransformSupported(m_skvData.fileHandle, streamID))
		{
			SKV_getStereoTransform(m_skvData.fileHandle, streamID, stereoParams);
		}

		float frameRate = SKV_getStreamAverageFPS(m_skvData.fileHandle, streamID);
		if (frameRate > m_frameRate)
		{
			m_frameRate = frameRate;
			m_totalFramecount = SKV_getStreamFrameCount(m_skvData.fileHandle, streamID);
			m_timeBetweenFrames = 1000.0 / m_frameRate;
			m_skvData.mainStreamID = streamID;
		}

	}

	//////////////////////////////////////////////////////////////////////////
	void Decoder::readImageStream(uint32_t streamID, uint64_t timeStamp, void* imagePtr)
	{
		if(!SKV_seekFrameByTimeStamp(m_skvData.fileHandle, streamID, timeStamp))
		{
			throw std::runtime_error("Could not seek to timestamp" + std::to_string(timeStamp)+ " for stream "+ std::to_string(streamID));
		}

		if(!SKV_getCurrentFrameData(m_skvData.fileHandle, imagePtr))
		{
			throw std::runtime_error("Could not get image for timestamp " +std::to_string(timeStamp));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Image32& Decoder::getColorImage() const
	{
		if (m_readColor && m_skvData.hasColor)
		{
			return m_colorImage;
		}
		else
		{
			throw std::runtime_error("Color image doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Image32& Decoder::getColorLeftImage() const
	{
		if (m_readColorLeft && m_skvData.hasColorLeft)
		{
			return m_colorLeftImage;
		}
		else
		{
			throw std::runtime_error("Color left image doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Image16& Decoder::getYUVImage() const
	{
		if (m_readYUV && m_skvData.hasYUV)
		{
			return m_YUVImage;
		}
		else
		{
			throw std::runtime_error("YUV image doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Image16& Decoder::getYUVLeftImage() const
	{
		if (m_readYUVLeft && m_skvData.hasYUVLeft)
			return m_YUVLeftImage;
		else
		{
			throw std::runtime_error("YUV left image doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Image16& Decoder::getDepthImage() const
	{
		if (m_readDepth && m_skvData.hasDepth)
		{
			return m_depthImage;
		}
		else
		{
			throw std::runtime_error("Depth image doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Image32Float& Decoder::getDepthFloatPlanesXYZImage() const
	{
		if (m_readDepthFloatPlanesXYZ && m_skvData.hasDepthFloatPlanesXYZ)
		{
			return m_depthFloatPlanesXYZImage;
		}
		else
		{
			throw std::runtime_error("Depth float planes xyz image doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Image16& Decoder::getConfidenceImage() const
	{
		if (m_readConfidence && m_skvData.hasConfidence)
		{
			return m_confidenceImage;
		}
		else
		{
			throw std::runtime_error("Confidence image doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Vector3& Decoder::getAccelerometerValue() const
	{
		if (m_skvData.hasAccelerometer)
		{
			return m_accelerometerValue;
		}
		else
		{
			throw std::runtime_error("Accelerometer value doesn't exist");
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Matrix4x4& Decoder::getAttitudeMatrix() const
	{
		if (m_skvData.hasAttitudeMatrix)
		{
			return m_attitudeMatrix;
		}
		else
		{
			throw std::runtime_error("Attitude matrix doesn't exist");
		}
	}
	//////////////////////////////////////////////////////////////////////////
	const uint32_t& Decoder::getStatusBitMask() const
	{
		if (m_skvData.hasStatusBitMask)
		{
			return m_statusBitMask;
		}
		else
		{
			throw std::runtime_error("status bit mask doesn't exist");
		}
	}

	void copySKVBufferToColorImage(SKV::Image32& image, const uint8_t* inSKVBuffer, SKV_ImageType skvImageType)
	{
		const size_t width = image.width, height = image.height;

		if (skvImageType == SKV_IMAGE_RGBA32 || skvImageType == SKV_IMAGE_BGRA32)
		{
			memcpy(image.ptr, inSKVBuffer, width*height * 4);
		}
		else if (skvImageType == SKV_IMAGE_RGB24 || skvImageType == SKV_IMAGE_BGR24)
		{
			// convert from 3 channels to 4 channels
			for (uint32_t y = 0; y < height; y++)
			{
				for (uint32_t w = 0; w < width; w++)
				{
					((uint8_t*)image.ptr)[4 * (width*y + w)] = ((uint8_t*)inSKVBuffer)[3 * (width*y + w)];
					((uint8_t*)image.ptr)[4 * (width*y + w) + 1] = ((uint8_t*)inSKVBuffer)[3 * (width*y + w) + 1];
					((uint8_t*)image.ptr)[4 * (width*y + w) + 2] = ((uint8_t*)inSKVBuffer)[3 * (width*y + w) + 2];
					((uint8_t*)image.ptr)[4 * (width*y + w) + 3] = 255;
				}
			}
		}
		else
		{
			throw std::runtime_error("Unsupported pixel type for color image");
		}
	}

}


#endif
