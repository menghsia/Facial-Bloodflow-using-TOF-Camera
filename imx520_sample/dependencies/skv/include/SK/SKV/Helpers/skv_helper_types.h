/***************************************************************************************/
//  SoftKinetic SKV library
//  Project Name      : SKV
//  Module Name		  : SKV helpers
//  Description       : Types for the SKV C++ helpers
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

#ifndef SOFTKINETIC_SKV_HELPERS_TYPES_INCLUDED
#define SOFTKINETIC_SKV_HELPERS_TYPES_INCLUDED

#include <tuple>
#include <string>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <iomanip>


/// \brief compare content of SKV_DistortionModel structs
inline bool operator==(const SKV_DistortionModel& lhs, const SKV_DistortionModel& rhs)
{
	return memcmp(&lhs, &rhs, sizeof(SKV_DistortionModel)) == 0;
}


/// \brief compare content of SKV_DistortionModel structs
inline bool operator==(const SKV_PinholeModel& lhs, const SKV_PinholeModel& rhs)
{
	return memcmp(&lhs, &rhs, sizeof(SKV_PinholeModel)) == 0;
}


/// \brief compare content of SKV_StereoTransform structs
inline bool operator==(const SKV_StereoTransform& lhs, const SKV_StereoTransform& rhs)
{
	return memcmp(&lhs, &rhs, sizeof(SKV_StereoTransform)) == 0;
}


/// \brief pretty printing for SKV_DistortionModel
inline std::ostream & operator<<(std::ostream &os, const SKV_DistortionModel& distortion)
{
	return os
		<< std::setw(10) << std::setprecision(8) << distortion.fx << " "
		<< std::setw(10) << std::setprecision(8) << distortion.fy << std::endl
		<< std::setw(10) << std::setprecision(8) << distortion.k1 << " "
		<< std::setw(10) << std::setprecision(8) << distortion.k2 << " "
		<< std::setw(10) << std::setprecision(8) << distortion.k3 << " "
		<< std::setw(10) << std::setprecision(8) << distortion.k4 << " "
		<< std::setw(10) << std::setprecision(8) << distortion.p1 << " "
		<< std::setw(10) << std::setprecision(8) << distortion.p2 << std::endl;
}


/// \brief pretty printing for SKV_PinholeModel
inline std::ostream & operator<<(std::ostream &os, const SKV_PinholeModel& pinhole)
{
	return os
		<< std::setw(10) << std::setprecision(8) << pinhole.fx << " "
		<< std::setw(10) << std::setprecision(8) << pinhole.fy << " "
		<< std::setw(10) << std::setprecision(8) << pinhole.cx << " "
		<< std::setw(10) << std::setprecision(8) << pinhole.cy;
}

/// \brief pretty printing for SKV_StereoTransform
inline std::ostream & operator<<(std::ostream &os, const SKV_StereoTransform& transform)
{
	return os
		<< std::setw(10) << std::setprecision(8) << transform.r11 << " "
		<< std::setw(10) << std::setprecision(8) << transform.r12 << " "
		<< std::setw(10) << std::setprecision(8) << transform.r13 << std::endl
		<< std::setw(10) << std::setprecision(8) << transform.r21 << " "
		<< std::setw(10) << std::setprecision(8) << transform.r22 << " "
		<< std::setw(10) << std::setprecision(8) << transform.r23 << std::endl
		<< std::setw(10) << std::setprecision(8) << transform.r31 << " "
		<< std::setw(10) << std::setprecision(8) << transform.r32 << " "
		<< std::setw(10) << std::setprecision(8) << transform.r33 << std::endl << std::endl
		<< std::setw(10) << std::setprecision(8) << transform.t1 << " "
		<< std::setw(10) << std::setprecision(8) << transform.t2 << " "
		<< std::setw(10) << std::setprecision(8) << transform.t3 << std::endl;
}


#if SKV_COMPILER == SKV_COMPILER_GCC && SKV_COMPILER_VERSION <= 480
#include <sstream>
namespace std{
	// std::to_string is not available on older gcc versions
	template <typename T>
	string to_string(T i)
	{
		stringstream s;
		s << i;
		return s.str();
	}
}
#else
// fallback on the default impl, assumes a c++11-compliant compiler and stdlib
#include <string>
#endif



namespace SKV
{

	enum Stream
	{
		COLOR = 1,
		DEPTH = 2,
		CONFIDENCE = 4,
		COLOR_LEFT = 8,
		YUV = 16,
		YUV_LEFT = 32,
		DEPTH_FLOAT_PLANES_XYZ = 64

	};

	/**
	 * \brief 8 bits image.
	 */
	struct Image8
	{
		Image8():
		 width(0),height(0),ptr(nullptr)
		{
		}
		int32_t width;
		int32_t height;
		int8_t *ptr;
	};

	/**
	 * \brief 16 bits image.
	 */
	struct Image16
	{
		Image16():
		 width(0),height(0),ptr(nullptr)
		{
		}

		Image16(int32_t w,int32_t h,int16_t *p):
		 width(w),height(h),ptr(p)
		{
		}

		int32_t width;
		int32_t height;
		int16_t *ptr;
	};

	/**
	 * \brief Color image.
	 */
	struct Image32
	{
		Image32():
		 width(0),height(0),countChannel(0),ptr(nullptr)
		{
		}

		Image32(int32_t w,int32_t h,int32_t cntChannel ,int32_t *p):
		 width(w),height(h),countChannel(cntChannel),ptr(p)
		{
		}

		int32_t width;
		int32_t height;
		int32_t countChannel;
		int32_t *ptr;
	};

	/**
	 * \brief 32 bits float image.
	 */
	struct Image32Float
	{
		Image32Float():
		 width(0),height(0),countChannel(0),ptr(nullptr)
		{
		}

		Image32Float(int32_t w,int32_t h,int32_t cntChannel ,float *p):
		 width(w),height(h),countChannel(cntChannel),ptr(p)
		{
		}

		int32_t width;
		int32_t height;
		int32_t countChannel;
		float	*ptr;
	};



	/**
	 * \brief  Camera settings.
	 */
	struct CameraParameters
	{
		CameraParameters():
			isDepthAvailable(true),
			isDepthFloatPlanesXYZAvailable(false),
			isColorAvailable(false),
			isColorLeftAvailable(false),
			isYUVAvailable(false),
			isYUVLeftAvailable(false),
			isConfidenceAvailable(false),
			depthWidth(0),depthHeight(0),
			colorWidth(0),colorHeight(0),colorChannelCount(0),
			fovx(0),fovy(0),
			isAccelerometerAvailable(false),
			isAttitudeMatrixAvailable(false),
			isStatusBitMaskAvailable(false),
			isUvMappingAvailable(false),
			isCalibrationAvailable(false),
			colorCompressionType(SKV_COMPRESSION_NONE),
			colorImageType(SKV_IMAGE_RGBA32)
		{
		}

		bool isDepthAvailable;
		bool isDepthFloatPlanesXYZAvailable;
		bool isColorAvailable;
		bool isColorLeftAvailable;
		bool isYUVAvailable;
		bool isYUVLeftAvailable;
		bool isConfidenceAvailable;

		/// depth image width
		int32_t depthWidth;
		/// depth image height
		int32_t depthHeight;

		/// depth image width
		int32_t colorWidth;
		/// depth image height
		int32_t colorHeight;


		/// number of channels of color image
		int32_t colorChannelCount;

		/// The field of view along the x axis expressed in degree units, for the depth camera.
		float fovx;
		/// The field of view along the y axis expressed in degree units, for the depth camera.
		float fovy;
		/// indicates whether the device includes an accelerometer or not.
		bool isAccelerometerAvailable;
		/// indicates whether the device includes an attitude matrix or not.
		bool isAttitudeMatrixAvailable;
		/// indicates whether the device includes an status bit mask or not.
		bool isStatusBitMaskAvailable;
		/// camera model
		std::string cameraModel;
		/// camera vendor name
		std::string vendorName;

		bool isUvMappingAvailable;
		SKV_DistortionModel depthDistortionParams, colorDistortionParams;
		SKV_PinholeModel depthPinholeParams, colorPinholeParams;
		SKV_StereoTransform cameraToCameraXForm;

		bool isCalibrationAvailable;

		// assume same settings for all color streams
		SKV_CompressionType colorCompressionType;
		SKV_ImageType colorImageType;

	};


	struct Vector3
	{
		Vector3():
			x(0),y(0),z(0)
		{
		}

		float x,y,z;
	};

	struct Matrix4x4
	{
		Matrix4x4()
		{
			memset(&data, 0, sizeof(float) * 16);
			data[0 + 4 * 0] = 1.0f;
			data[1 + 4 * 1] = 1.0f;
			data[2 + 4 * 2] = 1.0f;
			data[3 + 4 * 3] = 1.0f;
		}

		float data[16];
	};

	struct Frame
	{
		Image16			depth;
		Image32Float	depthFloatPlanesXYZ;
		Image16			confidence;
		Image32			color;
		Image32			colorLeft;
		Image16			YUV;
		Image16			YUVLeft;
		Vector3			accelerometer;
		Matrix4x4		attitudeMatrix;
		uint32_t		statusBitMask;
		uint64_t		timestamp;
	};


/**
 * \brief Callback function signature used to allocate memory.
 *
 */
typedef void *(*SKV_Allocate)(size_t size,size_t align);

/**
 * \brief Callback function signature used to deallocate memory.
 *
 */
typedef void (*SKV_Deallocate)(void *memory);

/**
 * \brief Allocator containing allocation/deallocation callback functions.
 */
struct SKV_Allocator
{
	SKV_Allocate allocate;
	SKV_Deallocate deallocate;
};




} //SKV

#endif
