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

#pragma once

#ifndef SOFTKINETIC_IU456_LIBRARY_API_H
#define SOFTKINETIC_IU456_LIBRARY_API_H

#include "iu456/visibility.h"
#include "iu456/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iu456_handle iu456_handle_t;

/*!
 *  \brief Initializes the library.
 *
 *  This function initializes the library. The user can also supply callbacks to be triggered whenever
 *  a compatible device is plugged in or removed. The user has also the ability to define where
 *  the 'data' folder will be located once installed.
 * 
 *  \note This function should be called before any other call to the library.
 *
 *  \note The 'on_arrival_callback' will be also called for devices already plugged in. 
 *
 *  \param[in]  on_arrival_callback   [in]a callback called whenever a new device is plugged in; could be passed as NULL.
 *  \param[in]  on_removal_callback   [in]a callback called whenever a device is unplugged; could be passed as NULL.
 *  \param[in]  user_data             [in]a pointer that will be passed as second argument to both on_arrival_callback and on_removal_callback; could be passed as NULL.
 *  \param[in]  path_to_bundle        [in]path to the zip or directory containing configuration data. If NULL, the builtin data will be used.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_initialize(iu456_on_arrival_callback_t on_arrival_callback, iu456_on_removal_callback_t on_removal_callback, void* user_data, const char* path_to_bundle, iu456_error_details_t* error_details);

/*!
*  \brief Initializes the library with device property.
*
*  This function initializes the library using device property. The user can also supply callbacks to be triggered whenever
*  a compatible device is plugged in or removed. The user has also the ability to define where
*  the 'data' folder will be located once installed. The user can also supply EEPROM salve address and where
*  the production data and calibration data located on th EEPROM. The user can also select one from multiple devices.
*
*  \note This function should be called before any other call to the library.
*
*  \note The 'on_arrival_callback' will be also called for devices already plugged in.
*
*  \note When using 'device_index', 'on_arrival_callback' after initialize is disabled,
*  'on_removal_callback' is valid only for the selected device.
*
*  \param[in]  on_arrival_callback   [in]a callback called whenever a new device is plugged in; could be passed as NULL.
*  \param[in]  on_removal_callback   [in]a callback called whenever a device is unplugged; could be passed as NULL.
*  \param[in]  user_data             [in]a pointer that will be passed as second argument to both on_arrival_callback and on_removal_callback; could be passed as NULL.
*  \param[in]  path_to_bundle        [in]path to the zip or directory containing configuration data. If NULL, the builtin data will be used.
*  \param[in]  rom_slave_address     [in]a pointer that the rom slave adresses, If NULL, the default value will be used.
*  \param[in]  number_of_address     [in]element count of rom_slave_address, If 0, the default eeprom slave address will be used.
*  \param[in]  production_address    [in]a production data location address on the EEPROM, can be passed as -1 for default location address.
*  \param[in]  calibration_address   [in]a calibration data location address on the EEPROM, can be passed as -1 for default location address.
*  \param[in]  device_index          [in]the index of specified device to initialize; can be passed as -1 for connecting all devices.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
*  \return  true if the operation performed correctly or false otherwise with the eventual
*           error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_initialize_with_device_property(iu456_on_arrival_callback_t on_arrival_callback, iu456_on_removal_callback_t on_removal_callback, void* user_data,
    const char* path_to_bundle, const uint8_t* rom_slave_address, uint32_t number_of_address, int production_address, int calibration_address, int device_index, iu456_error_details_t* error_details);

/*!
 *  \brief Shuts down the library.
 *
 *  \note This function should be called after any other call to the library.
 *
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_shutdown(iu456_error_details_t* error_details);

/*!
 *  \brief Returns the library version.
 *
 * \return  the library version.
 */
IU456_LIBRARY_API iu456_version_t IU456_LIBRARY_DECL iu456_get_version();

/*!
 *  \brief List already connected devices.
 *
 *  \note The allocation of the <em>device_array</em> is done in the library and the ownership is transfered
 *  to the client which should call <em>iu456_release_device()</em> for each element in the array and
 *  <em>iu456_release_device_list()</em> for the array itself.
 *
 *  \param[out] device_array          [out]pointer to an unallocated <em>iu456_device_t **</em>.
 *  \param[out] number_of_elements    [out]pointer to a <em>size_t</em> which will hold the length of <em>device_array</em>.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_device_list(const iu456_device_t*** device_array, size_t* number_of_elements, iu456_error_details_t* error_details);

/*!
 *  \brief Release a device array.
 *
 *  \note This function only deallocate the array itself not the elements in the array. client should
 *  call <em>iu456_release_device()</em> for each element in the array.
 * 
 *  \param[in] device_array           [in]the device list to release.
 */
IU456_LIBRARY_API void IU456_LIBRARY_DECL iu456_release_device_list(const iu456_device_t** device_array);

/*!
 *  \brief Release a device.
 *
 *  \param[in] device                 [in]the device list to release.
 */
IU456_LIBRARY_API void IU456_LIBRARY_DECL iu456_release_device(const iu456_device_t* device);

/*!
 *  \brief Construct a device object handle.
 *
 *  This function instantiates a specific device. The user can supply a callback that will be called whenever
 *  the said device is removed.
 * 
 *  \note The allocation of the object handle is done in the library and the ownership is transferred
 *  to the client which should call <em>iu456_destroy()</em>
 * 
 *  \param[out] handle                [out]pointer to an unallocated object handle.
 *  \param[in]  device                [in]pointer to a <em>iu456_device_t </em> object.
 *  \param[in]  on_removal_callback   [in]a callback called whenever this device is unplugged; could be passed as NULL.
 *  \param[in]  user_data             [in]a pointer that will be passed as argument to the on_removal_callback; could be passed as NULL.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 * \return  true if the operation performed correctly or false otherwise with the eventual 
 *          error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_create(iu456_handle_t** handle, const iu456_device_t* device, iu456_on_device_removal_callback_t on_removal_callback, void* user_data, iu456_error_details_t* error_details);

/*!
*  \brief Construct a device object handle using an external data bundle.
*
*  This function instantiates a specific device. The user can supply a callback that will be called whenever
*  the said device is removed.
*
*  \note The allocation of the object handle is done in the library and the ownership is transferred
*  to the client which should call <em>iu456_destroy()</em>
*
*  \param[out] handle                [out]pointer to an unallocated object handle.
*  \param[in]  device                [in]pointer to a <em>iu456_device_t </em> object.
*  \param[in]  on_removal_callback   [in]a callback called whenever this device is unplugged; could be passed as NULL.
*  \param[in]  user_data             [in]a pointer that will be passed as argument to the on_removal_callback; could be passed as NULL.
*  \param[in]  path_to_bundle        [in]the path to the data bundle; could be passed as NULL. In this case the same data as the one set during iu456_initialize will be used.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
* \return  true if the operation performed correctly or false otherwise with the eventual
*          error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_create_with_data_bundle(iu456_handle_t** handle, const iu456_device_t* device, iu456_on_device_removal_callback_t on_removal_callback, void* user_data, const char* path_to_bundle, iu456_error_details_t* error_details);

/*!
 *  \brief Destruct a device object handle.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_destroy(iu456_handle_t* handle, iu456_error_details_t* error_details);

/*!
 *  \brief Start the streaming.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_start(iu456_handle_t* handle, iu456_error_details_t* error_details);

/*!
 *  \brief Stop the streaming.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_stop(iu456_handle_t* handle, iu456_error_details_t* error_details);

/*!
 *  \brief Get data for the next frame.
 *
 *  This function will wait for the next frame and populate the <em>frame_data</em> pointer with the new data.
 *  The client can pass a timeout in milliseconds after which the function will return if no new data has been received.
 * 
 *  \note The value of <em>frame_data</em> is guaranteed until the next call to <em>iu456_get_last_frame()</em> or
 *        a call to <em>iu456_stop()</em>.
 *
 *  \note Even though an infinite timeout can be passed. The stream may be interrupted after a certain amount of time
 *        if the device is not streaming.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] frame_data            [out]pointer to an unallocated <em>const iu456_frame_t *</em>.
 *  \param[in]  timeout               [in]a timeout in milliseconds; can be passed as -1 for infinite timeout.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_last_frame(iu456_handle_t* handle, const iu456_frame_t** frame_data, int32_t timeout, iu456_error_details_t* error_details);

/*!
 *  \brief Get the serial number of the device.
 *
 *  \note The allocation of the <em>serial_number</em> is done in the library and the ownership is transferred
 *  to the client which should call <em>iu456_release_pointer()</em>
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] serial_number         [out]pointer to an unallocated <em>const char*</em>.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_serial_number(iu456_handle_t* handle, const char** serial_number, iu456_error_details_t* error_details);

/*!
 *  \brief Release memory allocated by the library
 *
 *  \param[in] data                   [in]the memory to release
 *
 *  \return  true if the operation performed correctly or false otherwise
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_release_pointer(const void* data);

/*!
 *  \brief Get the First 32bit of PRV number of the device.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] prv_number            [out]pointer to a <em>uint32_t</em>.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_prv_number(iu456_handle_t* handle, uint32_t* prv_number, iu456_error_details_t* error_details);

/*!
 *  \brief Get the full PRV number of the device.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] prv_number            [out]pointer to a <em>uint64_t</em>. 
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_prv_number64(iu456_handle_t* handle, uint64_t* prv_number, iu456_error_details_t* error_details);

/*!
 *  \brief Get the software_id of the device.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] software_id           [out]pointer to a <em>uint32_t</em>.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_software_id(iu456_handle_t* handle, uint32_t* software_id, iu456_error_details_t* error_details);

/*!
 *  \brief List available configurations.
 *
 *  \note The allocation of the <em>configuration_list</em> is done in the library and the ownership is transfered
 *  to the client which should call <em>iu456_release_pointer()</em>.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] configuration_list    [out]pointer to an unallocated <em>iu456_configuration_info_t **</em>.
 *  \param[out] number_of_entries     [out]pointer to a <em>size_t</em> which will hold the length of <em>configuration_list</em>.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_configuration_list(iu456_handle_t* handle, const iu456_configuration_info_t*** configuration_list, size_t* number_of_entries, iu456_error_details_t* error_details);

/*!
 *  \brief Get information about the currently selected configuration.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[out] configuration         [out]pointer to an unallocated <em>iu456_configuration_info_t *</em>.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_current_configuration(iu456_handle_t* handle, const iu456_configuration_info_t** configuration, iu456_error_details_t* error_details);

/*!
 *  \brief Select the mode uid to be used.
 *
 *  If the device is not in a streaming state, this function will set the configuration to be used whenever <em>iu456_start()</em> is called. 
 *  Otherwise, the mode in which the camera is running will be changed without interrupting the stream.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the mode uid to use.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_configuration_uid(iu456_handle_t* handle, int configuration_uid, iu456_error_details_t* error_details);

/*!
 *  \brief Select whether the depth map should be filtered or not.
 *
 *  This function can only be used when the device is not in a streaming state.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  state                 [in]whether the filtering is enabled or not.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_filtering(iu456_handle_t* handle, iu456_property_state_t state, iu456_error_details_t* error_details);

/*!
*  \brief Select whether the int16_t type depth map should be calculated or not.
*
*  This function can only be used when the device is not in a streaming state.
*
*  \param[in]  handle                [in]pointer to an already allocated object handle.
*  \param[in]  state                 [in]whether the filtering is enabled or not.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
*  \return  true if the operation performed correctly or false otherwise with the eventual
*           error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_int16_depth_calculation(iu456_handle_t* handle, iu456_property_state_t state, iu456_error_details_t* error_details);

/*!
*  \brief Select whether the float type depth map should be calculated or not.
*
*  This function can only be used when the device is not in a streaming state.
*
*  \param[in]  handle                [in]pointer to an already allocated object handle.
*  \param[in]  state                 [in]whether the filtering is enabled or not.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
*  \return  true if the operation performed correctly or false otherwise with the eventual
*           error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_float_depth_calculation(iu456_handle_t* handle, iu456_property_state_t state, iu456_error_details_t* error_details);

/*!
*  \brief Select whether the confidence map should be outputted filtered or not.
*
*  This function can only be used when the device is not in a streaming state.
*
*  \param[in]  handle                [in]pointer to an already allocated object handle.
*  \param[in]  state                 [in]whether the filtering is enabled or not.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
*  \return  true if the operation performed correctly or false otherwise with the eventual
*           error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_confidence_filtering(iu456_handle_t* handle, iu456_property_state_t state, iu456_error_details_t* error_details);

/*!
 *  \brief Adjust the frame rate at which the device runs.
 *
 *  This function only works when the device is in a streaming state.
 *  This value should be in the range returned by <em>iu456_get_frame_rate_range()</em>.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  desired               [in]the desired frame rate. 
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_frame_rate()</em>, <em>iu456_get_frame_rate_range</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_frame_rate(iu456_handle_t* handle, float desired, iu456_error_details_t* error_details);

/*!
 *  \brief Get the minimum and maximum frame rate for a given configuration.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid.
 *  \param[out] minimum               [out]pointer to a float that will be filled with the minimum frame rate.
 *  \param[out] maximum               [out]pointer to a float that will be filled with the maximum frame rate.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_frame_rate()</em>, <em>iu456_set_frame_rate</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_frame_rate_range(iu456_handle_t* handle, int configuration_uid, float* minimum, float* maximum, iu456_error_details_t* error_details);

/*!
 *  \brief Update the default frame rate for a given mode.
 *
 *  This value should be in the range returned by <em>iu456_get_frame_rate_range()</em>.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid for which to update the default.
 *  \param[in]  desired               [in]the default frame rate.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_set_frame_rate()</em>, <em>iu456_get_frame_rate_range</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_update_default_frame_rate(iu456_handle_t* handle, int configuration_uid, float desired, iu456_error_details_t* error_details);

/*!
*  \brief Select whether the auto exposure should be enabled or not for a given configuration.
*
*  When the auto exposure is enabled, the exposure cannot be controlled using iu456_set_exposure_time anymore
*
*  \param[in]  handle                [in]pointer to an already allocated object handle.
*  \param[in]  configuration_uid     [in]the configuration uid for which to enable or disable the auto exposure.
*  \param[in]  state                 [in]whether the auto exposure is enabled or not.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
*  \return  true if the operation performed correctly or false otherwise with the eventual
*           error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_auto_exposure(iu456_handle_t* handle, int configuration_uid, iu456_property_state_t state, iu456_error_details_t* error_details);

/*!
*  \brief Set the region of interest (ROI) that is taken into acount to control the exposure for a given configuration.
*
*  By default, the ROI is set to the full frame.
*
*  \param[in]  handle                [in]pointer to an already allocated object handle.
*  \param[in]  configuration_uid     [in]the configuration uid for which to update the auto exposure ROI.
*  \param[in]  x                     [in]the x coordinate of the ROI's top left corner.
*  \param[in]  y                     [in]the y coordinate of the ROI's top left corner.
*  \param[in]  width                 [in]the ROI's width.
*  \param[in]  height                [in]the ROI's height.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
*  \return  true if the operation performed correctly or false otherwise with the eventual
*           error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_auto_exposure_roi(iu456_handle_t* handle, int configuration_uid, int x, int y, int width, int height, iu456_error_details_t* error_details);

/*!
 *  \brief Change the trigger mode of the sensor.
 *
 *  The sensor is configured in free-running mode by default.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid for which to update the default.
 *  \param[in]  trigger_mode          [in]the trigger mode.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_update_default_trigger_mode(iu456_handle_t* handle, int configuration_uid, iu456_trigger_mode_t trigger_mode, iu456_error_details_t* error_details);

/*!
 *  \brief Specify the trigger upsampling or downsampling factor.
 *
 *  When choosing <em>iu456_trigger_sampling_downsample</em>, <em>number_of_frames</em> specifies the number
 *  of triggers to be ignored. For instance if <em>number_of_frames</em> is equal to 1, the trigger
 *  frequency will be halved.
 *
 *  When choosing <em>iu456_trigger_sampling_upsample</em>, <em>number_of_frames</em> specifies the number
 *  of frames to capture between 2 triggers. For instance if <em>number_of_frames</em> is equal to 2, the trigger
 *  frequency will be doubled.
 *
 *  The default values are 1 frame and iu456_trigger_sampling_upsample.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid for which to update the default.
 *  \param[in]  number_of_frames      [in]the number of frames to capture.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_trigger_delay</em>, <em>iu456_get_trigger_factor_range</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_update_default_trigger_factor(iu456_handle_t* handle, int configuration_uid, int number_of_frames, iu456_trigger_sampling_t sampling_mode, iu456_error_details_t* error_details);

/*!
 *  \brief Specify the delay between the external vsync pad and the internal frame start.
 *
 *  The default value is 0 seconds.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid for which to update the default.
 *  \param[in]  delay                 [in]the delay in seconds.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_trigger_factor()</em>, <em>iu456_get_trigger_delay_range()</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_update_default_trigger_delay(iu456_handle_t* handle, int configuration_uid, float delay, iu456_error_details_t* error_details);

/*!
 *  \brief Get the minimum and maximum number of frames for trigger frequency upsampling and downsampling.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid.
 *  \param[out] minimum               [out]pointer to an int that will be filled with the minimum number of frames.
 *  \param[out] maximum               [out]pointer to an int that will be filled with the maximum number of frames.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_trigger_factor()</em>, <em>iu456_update_default_trigger_delay</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_trigger_factor_range(iu456_handle_t* handle, int configuration_uid, int* minimum, int* maximum, iu456_error_details_t* error_details);

/*!
 *  \brief Get the minimum and maximum delay in second.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid.
 *  \param[out] minimum               [out]pointer to a float that will be filled with the minimum delay in seconds.
 *  \param[out] maximum               [out]pointer to a float that will be filled with the maximum delay in seconds.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_trigger_factor()</em>, <em>iu456_update_default_trigger_delay</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_trigger_delay_range(iu456_handle_t* handle, int configuration_uid, float* minimum, float* maximum, iu456_error_details_t* error_details);

/*!
 *  \brief Adjust the exposure time of the currently running mode.
 *
 *  This function only works when the device is in a streaming state.
 *  This value should be in the range returned by <em>iu456_get_exposure_time_list()</em>.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  exposure_time         [in]the desired exposure time in microseconds. 
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_exposure_time()</em>, <em>iu456_get_exposure_time_list</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_exposure_time(iu456_handle_t* handle, int exposure_time, iu456_error_details_t* error_details);

/*!
 *  \brief Get a list of supported exposure time range for a given configuration.
 *
 *  The list consists of three exposure time, minimum, typical, and maximum.
 *  Exposure times are expressed in microseconds.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid.
 *  \param[out] values                [out]pointer to an unallocated array of int.
 *  \param[out] length                [out]pointer to a size_t holding the length of the array.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_update_default_exposure_time()</em>, <em>iu456_set_frame_rate</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_exposure_time_list(iu456_handle_t* handle, int configuration_uid, const int** values, size_t* length, iu456_error_details_t* error_details);

/*!
 *  \brief Update the default exposure time for a given mode.
 *
 *  This value should be in the range returned by <em>iu456_get_exposure_time_list()</em>.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  configuration_uid     [in]the configuration uid for which to update the default.
 *  \param[in]  desired               [in]the default exposure time.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 *
 *  \see <em>iu456_set_frame_rate()</em>, <em>iu456_get_exposure_time_list</em>
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_update_default_exposure_time(iu456_handle_t* handle, int configuration_uid, int desired, iu456_error_details_t* error_details);

/*!
 *  \brief Get the uid matching the mode defined in the specifications.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  mode_id               [in]the mode id as stated in the datasheet.
 *  \param[in]  fps                   [in]the nominal frame rate of the matching mode.
 *  \param[out] uid_to_use            [out]the configuration uid to use in all other APIs.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_uid_for_mode(iu456_handle_t* handle, int mode_id, int fps, int* uid_to_use, iu456_error_details_t* error_details);

/*!
 *  \brief Whether the x and y maps are computed or not.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  state                 [in]the state of the property.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_x_and_y_property_state(iu456_handle_t* handle,
                                                                           iu456_property_state_t state,
                                                                           iu456_error_details_t* error_details);

/*!
 *  \brief Select whether the update uid in frame or not.
 *
 *  This function can only be used when the device is not in a streaming state.
 *
 *  \param[in]  handle                [in]pointer to an already allocated object handle.
 *  \param[in]  state                 [in]the state of the property.
 *  \param[out] error_details         [out]error details object handle; could be passed as NULL.
 *
 *  \return  true if the operation performed correctly or false otherwise with the eventual 
 *           error details object handle holding an error code and a message explaining the underlying error
 */
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_set_update_uid_property_state(iu456_handle_t* handle,
                                                                              iu456_property_state_t state,
                                                                              iu456_error_details_t* error_details);

/*!
*  \brief Get the raw calibration data of the device.
*
*  \param[in]  handle                [in]pointer to an already allocated object handle.
*  \param[out] data                  [out]pointer to a <em>const uint8_t *</em>.
*  \param[out] length                [out]pointer to a <em>size_t</em>.
*  \param[out] error_details         [out]error details object handle; could be passed as NULL.
*
*  \return  true if the operation performed correctly or false otherwise with the eventual
*           error details object handle holding an error code and a message explaining the underlying error
*/
IU456_LIBRARY_API bool IU456_LIBRARY_DECL iu456_get_raw_calibration_data(iu456_handle_t* handle, const uint8_t** data, size_t* length, iu456_error_details_t* error_details);

#ifdef __cplusplus
}
#endif

#endif

