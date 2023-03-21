// COPYRIGHT AND CONFIDENTIALITY NOTICE
// SONY DEPTHSENSING SOLUTIONS CONFIDENTIAL INFORMATION
//
// All rights reserved to Sony Depthsensing Solutions SA/NV, a
// company incorporated and existing under the laws of Belgium, with
// its principal place of business at Boulevard de la Plainelaan 11,
// 1050 Brussels (Belgium), registered with the Crossroads bank for
// enterprises under company number 0811 784 189
//
// This file is part of the iu456_library, which is proprietary
// and confidential information of Sony Depthsensing Solutions SA/NV.
//
// Copyright (c) 2017 Sony Depthsensing Solutions SA/NV

/**
 * \file types.h    Declaration of the Frame Grabbers API types
 */

#pragma once

#ifndef SOFTKINETIC_IU456_LIBRARY_API_TYPES_H_INCLUDED_
#define SOFTKINETIC_IU456_LIBRARY_API_TYPES_H_INCLUDED_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief IU456 Library API error codes.
 */
typedef enum
{
    /// Everything went fine except the weather
    iu456_error_success = 0,
    /// Generic error - we're afraid of the unknown
    iu456_error_generic = -1,
    /// Invalid handle object
    iu456_error_invalid_object_handle = -2,
    /// NULL pointer argument
    iu456_error_null_pointer_argument = -3,
    /// timeout
    iu456_error_timeout = -4,
    /// device is overheating
    iu456_error_overheating = -5,
    /// out of range error
    iu456_error_out_of_range = -6,
    /// Device is unresponsive
    iu456_error_hardware_unresponsive = -7,
    /// Overrun of the mode fps
    iu456_error_mode_fps_overrun = -8,
    /// Invalid raw data
    iu456_error_invalid_rawdata = -9,
} iu456_error_t;

/**
 * \brief The state of a property.
 */
typedef enum
{
    /// Disabled
    iu456_property_state_disabled = 0,
    /// Enabled
    iu456_property_state_enabled = 1
} iu456_property_state_t;


/**
 * \brief Error details object to provide additional information regarding an error.
 *
 * \note This type of object is client allocated and passed along to the API functions
 * to obtain error details if desired.
 */
typedef struct iu456_error_details_t
{
    /// Error code
    iu456_error_t code;
    /// Error specific message
    const char* message;
} iu456_error_details_t;

/**
 * \brief Library version object to provide version information.
 */
typedef struct iu456_version_t
{
    /// The major
    int major;
    /// The minor
    int minor;
    /// The patch
    int patch;
    /// The stage
    const char* stage;
} iu456_version_t;

/**
 * \brief An object reprensenting a physical device
 *
 */
typedef struct iu456_device_t
{
    /// The vendor id of the SSP-500
    int vid;
    /// The product id of the SSP-500
    int pid;
    /// The revision of the SSP-500
    int revision;
    /// The PRV number of the module
    int prv;
    /// The serial number of the module
    const char* serial_number;
    /// The path to the control interface on the host
    const char* control_path;
    /// The path to the stream interface on the host
    const char* stream_path;
} iu456_device_t;


/**
 * \brief An object reprensenting the lens model of the module
 *
 */
typedef struct iu456_intrinsics_t
{
    /// The focal length along the horizontal axis in pixel unit
    float fx;
    /// The focal length along the vertical axis in pixel unit
    float fy;
    /// The position of the central point along the horizontal axis
    float cx;
    /// The position of the central point along the vertical axis
    float cy;
    /// The first radial distortion coefficient of the Brown-Conrady model
    float k1;
    /// The second radial distortion coefficient of the Brown-Conrady model
    float k2;
    /// The third radial distortion coefficient of the Brown-Conrady model
    float k3;
    /// The first tangential distortion coefficient of the Brown-Conrady model
    float p1;
    /// The second tangential distortion coefficient of the Brown-Conrady model
    float p2;
} iu456_intrinsics_t;

/**
 * \brief An object providing additional information of a mode
 *
 */
typedef struct iu456_configuration_info_t
{
    /// Whether calibrated data is available in the device or not
    bool is_calibrated;
    /// The unique id of the configuration
    int uid;
    /// The width of the output maps
    int width;
    /// The height of the output maps
    int height;
    /// The expected rate at which frames are streamed
    float frame_rate;
    /// The unambiguous range
    float unambiguous_range;
    /// The pinhole and Brown-Conrady model of the lens
    const struct iu456_intrinsics_t* intrinsics;
    /// The name of the mode
    const char* name;
} iu456_configuration_info_t;

/**
 * \brief An object providing the data for a input raw data
 *
 * This structure holds the pixel data and embedded data of the raw data separately.
 * In the case that you have the data conforms to this structure,
 * you can reduce the copying process for reordering by using this structure.
 */
typedef struct iu456_raw_data_t
{
    /// pixel data pointer
    uint8_t* pixel_data;
    /// pixel data length
    size_t pixel_data_length;
    /// embedded data pointer
    uint8_t* embedded;
    /// embedded data length
    size_t embedded_length;
} iu456_raw_data_t;

/**
 * \brief An object providing the data for a received frame
 *
 * This structure holds the post processed depth and confidence as well
 * as other meta data linked to the frame itself.
 *
 * The confidence map reflects the quality of the data at a given pixel.
 * The greater this value is, the lesser noise the data will have.
 *
 * The depth map is the cartesian depth expressed in millimeters and uses 
 * values greater than 32000 to flag invalid values.
 * 
 * The x and y pointers provides access to the first 2 cartesian coordinates.
 * This means that when using (x, y, depth), one gets a list of 3d vertices 
 * in camera space. Note that x and y maps are optional 
 * (see <em>iu456_set_x_and_y_property_state()</em>). When disabled, these 2
 * pointers will be set to nullptr.
 */
typedef struct iu456_frame_t
{
    /// A counter provided by the module
    int32_t frame_id;
    /// The time in microseconds since an unspecified point in time at which the sample was received
    int64_t timestamp;
    /// A pointer to additional information about the mode that was used to capture this frame
    const struct iu456_configuration_info_t* configuration_info;
    /// An opaque pointer
    const uint8_t* raw;
    /// The length of the raw array
    size_t raw_length;
    /// A pointer to the confidence map
    const int16_t* confidence;
    /// The number of elements in the confidence map array
    size_t confidence_length;
    /// A pointer to the depth map in millimeters. 
    /// The values greater than 32000 is used to flag invalid values as follows:
    /// - 32001: Invalid pixels due to de-aliasing failure
    /// - 32002: Saturated pixels
    /// - others: reserved
    const int16_t* depth;
    /// The number of elements in the depth map array
    size_t depth_length;
    /// A pointer to an array holding the cartesian position in millimeters of the pixel along the x axis
    const int16_t* x;
    /// The number of elements in the x array
    size_t x_length;
    /// A pointer to an array holding the cartesian position in millimeters of the pixel along the y axis
    const int16_t* y;
    /// The number of elements in the y array
    size_t y_length;
    /// A pointer to the depth map in millimeters (float type). 
    /// The values greater than 32000 is used to flag invalid values as follows:
    /// - 32001: Invalid pixels due to de-aliasing failure
    /// - 32002: Saturated pixels
    /// - others: reserved
    const float* depth_float;
    /// The number of elements in the depth map array
    size_t depth_float_length;
    /// A pointer to an array holding the cartesian position in millimeters of the pixel along the x axis(float type)
    const float* x_float;
    /// The number of elements in the x_float array
    size_t x_float_length;
    /// A pointer to an array holding the cartesian position in millimeters of the pixel along the y axis(float type)
    const float* y_float;
    /// The number of elements in the y_float array
    size_t y_float_length;    
    /// The laser temperature in celsius, acquire the temperature from the top component of the frame
    float laser_temperature;
    /// The exposure time in microseconds
    int exposure;
    /// Whether the data were acquired with an out of spec temperature
    bool low_accuracy_data;
    /// The sensor temperature in celsius, the temperature is average of all components of the frame
    float sensor_temperature;
    /// The datatype of error information field
    /// - 0x00: none
    /// - 0x01: ErrorMonitor for CXA4016/S6674
    /// - 0x02: ErrorMonitor1 for CXA4026
    /// - 0x03: ErrorMonitor2 for CXA4026
    /// - 0x10: APC2_CHECK_DATA[9:8], APC1_CHECK_DATA[9:8], PD_H2[9:8]
    /// - 0x11: APC2_CHECK_DATA[7:0]
    uint32_t error_information_type;
    /// The error information, it's format depend on the error_information_type
    uint32_t error_information;
} iu456_frame_t;

/**
* \brief The trigger mode of the sensor.
*/
typedef enum
{
    /// the master timing generator is free running
    iu456_trigger_mode_free_running = 0,
    /// The master timing generator is triggered by a pin of the sensor chip.
    iu456_trigger_mode_externally_triggered = 1,
} iu456_trigger_mode_t;

/**
* \brief The type of sampling used when sensor is externally triggered.
*/
typedef enum
{
    /// Downsample the trigger frequency
    iu456_trigger_sampling_downsample = 0,
    /// Upsample the trigger frequency
    iu456_trigger_sampling_upsample = 1,
} iu456_trigger_sampling_t;

/*!
 * \typedef iu456_on_device_removal_callback_t
 *
 * \brief Callable used to notify whenever a device has been removed.
 *
 */
typedef void(*iu456_on_device_removal_callback_t)(void* user_data);

/*!
 * \typedef iu456_on_arrival_callback_t
 *
 * \brief Callable used to notify whenever a new device has been added.
 *
 */
typedef void(*iu456_on_arrival_callback_t)(const iu456_device_t* device, void* user_data);

/*!
 * \typedef iu456_on_removal_callback_t
 *
 * \brief Callable used to notify whenever a new device has been removed.
 *
 */
typedef void(*iu456_on_removal_callback_t)(const iu456_device_t* device, void* user_data);

typedef enum
{
    /// No Transformation
    iu456_transformation_mode_default = 0,
    /// Mirror horizontally
    iu456_transformation_mode_mirror_h = 1,
    /// Mirror vertically
    iu456_transformation_mode_mirror_v = 2,
    /// Flip
    iu456_transformation_mode_flip = 3,
} iu456_transformation_mode_t;

#ifdef __cplusplus
}
#endif

#endif
