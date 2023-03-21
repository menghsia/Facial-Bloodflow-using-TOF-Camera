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

/**
 * \file cloudify_api.h
 *  Main file for the cloudify API
 */

#pragma once

#ifndef cloudify_API_INCLUDED
#define cloudify_API_INCLUDED

#include "cloudify_types.h"
#include "cloudify_platform.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

	/**
	 * \brief  Return the version number of cloudify.
	 *
	 * \param[out]  version             A pointer to a char array containing the version.
	 *                                  This will contain the version after this call.
	 * \param[out]  error_details       Pointer to an allocated cloudify_error_details struct that will
	 * contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns     true if the call succeeded
	 */
	cloudify_API bool cloudify_DECL cloudify_get_version(
		cloudify_OUT const char** version,
		cloudify_OUT cloudify_error_details* error_details);


	/**
	 * \brief  Create an instance of cloudify.
	 *
	 * \param[in]	camera_parameters	Array of allocated cloudify_camera_parameters structs,
	 *									contains the camera parameters to use. The size must be equal to
	 *									camera_count.
	 * \param[in]	config_path			Relative cloudify configuration path, by default if field empty read "./cloudify.config.xml path"
	 * \param[out]	handle				Pointer to a handle pointer.
	 * \param[out]	error_details		Pointer to an allocated cloudify_error_details struct that will
	 *contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns		true if the call succeeded
	 *
	 * \post
	 *	- if successful, handle is a valid cloudify handle which can be used by the other API functions
	 *	- if not handle is a NULL
	 */
	cloudify_API bool cloudify_DECL cloudify_init(
		cloudify_IN const cloudify_intrinsic_parameters* const intrinsic,
		cloudify_OUT cloudify_handle** handle,
		cloudify_OUT cloudify_error_details* error_details);


	/**
	 * \brief  Release an instance of cloudify.
	 *
	 * \param[in,out]	handle			Pointer to a valid handle pointer.
	 * \param[out]		error_details	Pointer to an allocated cloudify_error_details struct that will
	 *contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns		true if the call succeeded
	 *
	 * \post
	 *	- handle set to NULL if the destruction was successful
	 */
	cloudify_API bool cloudify_DECL cloudify_release(
		cloudify_IN_OUT cloudify_handle** handle,
		cloudify_OUT cloudify_error_details* error_details);

	/**
	 * \brief Process frames
	 *
	 * \param[in,out]	handle			handle created by cloudify_create().
	 * \param[in]		frames			Array of allocated cloudify_frame structs, contains the
	 *									newest frames to feed to the pipeline.
	 * \param[out]		error_details	Pointer to an allocated cloudify_error_details struct that will
	 *contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns		true if the call succeeded
	 *
	 */
	cloudify_API bool cloudify_DECL cloudify_compute_radial_to_cartesian_depth(
		cloudify_IN_OUT cloudify_handle* handle,
		cloudify_IN const size_t x_index,
		cloudify_IN const size_t y_index,
		cloudify_IN const float radial_depth,
		cloudify_OUT float* cartesian_depth,
		cloudify_OUT cloudify_error_details* error_details);

	cloudify_API bool cloudify_DECL cloudify_compute_3d_point(
		cloudify_IN_OUT cloudify_handle* handle,
		cloudify_IN const size_t x_index,
		cloudify_IN const size_t y_index,
		cloudify_IN const float depth,
		cloudify_OUT cloudify_position_3d* position_3d,
		cloudify_OUT cloudify_error_details* error_details);



#ifdef __cplusplus
}
#endif

#endif
