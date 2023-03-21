/****************************************************************************************/
// COPYRIGHT AND CONFIDENTIALITY NOTICE
// SONY DEPTHSENSING SOLUTIONS CONFIDENTIAL INFORMATION
//
// All rights reserved to Sony Depthsensing Solutions SA/NV, a
// company incorporated and existing under the laws of Belgium, with
// its principal place of business at Boulevard de la Plainelaan 11,
// 1050 Brussels (Belgium), registered with the Crossroads bank for
// enterprises under company number 0811 784 189
//
// This file is part of cloudify Library, which is proprietary
// and confidential information of Sony Depthsensing Solutions SA/NV.


// Copyright (c) 2020 Sony Depthsensing Solutions SA/NV
/****************************************************************************************/

/** \file cloudify_types.h
*  Public types for the cloudify API
*/

#pragma once

#ifndef cloudify_TYPES_INCLUDED
#define cloudify_TYPES_INCLUDED

#ifdef CUSTOM_STDINT
#include CUSTOM_STDINT
#else
#if (defined(_MSC_VER) && _MSC_VER > 1600) || !defined(_MSC_VER)
#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#include <cstring>
#else
#include <cstdlib.h>
#include <cstring.h>
#include <stdint.h>
#endif
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

	/************************************************************************
	*                               General                                 *
	************************************************************************/

	/**
	* \brief Handle to a cloudify instance
	*/
	typedef struct cloudify_handle cloudify_handle;

	struct cloudify_position_3d
	{
		float x;
		float y;
		float z;
	};

	struct cloudify_intrinsic_parameters
	{
		int width;
		int height;

		float cx;
		float cy;

		float fx;
		float fy;

		float k1;
		float k2;
		float k3;
		float k4;
	};
	
	/**
	* \brief Return codes for cloudify functions
	*/
	enum cloudify_error_code
	{
		/// Everything went fine
		cloudify_success = 0,
		/// Generic code for unforeseen errors
		cloudify_other_failure,
		/// The function received a null handle
		cloudify_null_handle,
		/// The function received a non-null handle that does not correspond to an allocated instance
		cloudify_invalid_handle,
		/// Failed to initialize
		cloudify_failed_to_initialize,
		/// Failed to shutdown
		cloudify_failed_to_shutdown,
		/// Failed to compute
		cloudify_failed_to_compute,

		cloudify_null_data_pointer,

		cloudify_error_code_count
	};

	/**
	* \brief Describes an API call error.
	*/
	struct cloudify_error_details
	{
		/// The error code. See cloudify_errorcodes.h for the complete list
		cloudify_error_code code;
		/// Human-readable message
		const char* message;
	};

	static const char* cloudify_error_descriptions[cloudify_error_code_count] = {
		/* cloudify_success=0                   */ "success",
		/* cloudify_other_failure               */ "undocumented error",
		/* cloudify_null_handle                 */ "the given handle parameter is a null pointer",
		/* cloudify_invalid_handle              */ "the given handle parameter is not a pointer that has been created by cloudify_create()",
		/* cloudify_failed_to_initialize        */ "initialization failed",
		/* cloudify_failed_to_shutdown          */ "shutdown failed",
		/* cloudify_failed_to_compute           */ "Failed to compute cloudify",
		/* cloudify_null_data_pointer           */ "Failed - null_data_pointer",
	};


#define cloudify_IN
#define cloudify_OUT
#define cloudify_IN_OUT

#ifdef __cplusplus
}
#endif

#endif
