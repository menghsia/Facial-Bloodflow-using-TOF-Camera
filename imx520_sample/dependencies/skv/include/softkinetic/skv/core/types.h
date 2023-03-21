/***************************************************************************************/
//  SoftKinetic SKV library
//  Project Name      : SKV
//  Module Name	      : SKV API v2
//  Description       : Types for the C API v2
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

#if !defined(SOFTKINETIC_SKV_CORE_TYPES_INC)
#define SOFTKINETIC_SKV_CORE_TYPES_INC

#pragma once

#ifndef SKVLIB_OVERRIDE_TYPES

#ifdef _MSC_VER
	#include <yvals.h>
	// Types
	typedef unsigned char         uint8_t;
	typedef unsigned short        uint16_t;
	typedef short                 int16_t;
	typedef __int32               int32_t;
	typedef unsigned __int32      uint32_t;
	typedef unsigned __int64      uint64_t;
	typedef __int64               int64_t;

	#ifdef __cplusplus
		typedef bool                  bool_t;
	#else
		typedef char                  bool_t;
	#endif

	#else
	#include <stdint.h>
	#include <stdbool.h>
	#include <stddef.h>

	#ifdef __cplusplus
		typedef bool              bool_t;
	#else
		#if defined( __ORBIS__ )
			typedef std::_Bool bool_t;
		#elif defined( __APPLE__ ) || defined(__clang__)
			typedef bool bool_t;
		#elif defined (__ghs__)
			typedef char          bool_t;		
		#else
			typedef _Bool bool_t;
		#endif
	#endif

#endif // _MSC_VER

#endif // SKVLIB_OVERRIDE_TYPES 

#endif // SOFTKINETIC_SKV_CORE_TYPES_INC
