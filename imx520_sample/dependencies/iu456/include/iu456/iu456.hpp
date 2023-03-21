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

#ifndef SOFTKINETIC_IU456_LIBRARY_API_HPP
#define SOFTKINETIC_IU456_LIBRARY_API_HPP

#include "iu456/iu456.h"
#include "iu456/iu456_private.h"

#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>


namespace softkinetic
{
namespace camera
{
    class iu456
    {
    public:
        static iu456_version_t get_version()
        {
            return iu456_get_version();
        }

        explicit iu456(const iu456_device_t* device = nullptr, iu456_on_device_removal_callback_t on_removal_callback = nullptr, void* user_data = nullptr, const char* path_to_bundle = nullptr)
            : handle(nullptr)
        {
            if (!iu456_create_with_data_bundle(&handle, device, on_removal_callback, user_data, path_to_bundle, nullptr))
            {
                throw std::runtime_error("failed to instantiate device");
            }
        }

        ~iu456()
        {
            if (handle != nullptr)
            {
                iu456_destroy(handle, nullptr);
            }
            handle = nullptr;
        }

        iu456(iu456 const&) = delete;
        iu456& operator=(iu456 const&) = delete;

        iu456 ( iu456 && other)
        {
            handle = other.handle;
            other.handle = nullptr;
        }

        std::pair<bool, uint32_t> get_prv_number(iu456_error_details_t* error_details = nullptr)
        {
            uint32_t prv_number = 0;
            return std::make_pair(iu456_get_prv_number(handle, &prv_number, error_details), prv_number);
        }

        std::pair<bool, std::string> get_serial_number(iu456_error_details_t* error_details = nullptr)
        {
            const char* serial_number = nullptr;
            if (!iu456_get_serial_number(handle, &serial_number, error_details))
            {
                return std::make_pair(false, "invalid");
            }

            std::string serial(serial_number);
            iu456_release_pointer(serial_number);

            return std::make_pair(true, serial);
        }

        bool set_configuration_uid(int configuration_uid, iu456_error_details_t* error_details = nullptr)
        {
            return iu456_set_configuration_uid(handle, configuration_uid, error_details);
        }

        bool set_exposure_time(int exposure_time, iu456_error_details_t* error_details = nullptr)
        {
            return iu456_set_exposure_time(handle, exposure_time, error_details);
        }

        std::pair<bool, std::vector<iu456_configuration_info_t> > get_configuration_list(iu456_error_details_t* error_details = nullptr)
        {
            std::vector<iu456_configuration_info_t> output;
            const iu456_configuration_info_t** configuration_list = nullptr;
            size_t number_of_configurations = 0;
            if (!iu456_get_configuration_list(handle, &configuration_list, &number_of_configurations, error_details))
            {
                return std::make_pair(false, output);
            }

            output.resize(number_of_configurations);
            for (size_t c = 0; c < number_of_configurations; c++)
            {
                output[c] = *configuration_list[c];
            }

            iu456_release_pointer(configuration_list);
            return std::make_pair(true, output);
        }

        std::pair<bool, iu456_configuration_info_t> get_current_configuration(iu456_error_details_t* error_details = nullptr)
        {
            const iu456_configuration_info_t* current_configuration = nullptr;
            if (!iu456_get_current_configuration(handle, &current_configuration, error_details))
            {
                return std::make_pair(false, iu456_configuration_info_t());
            }
            return std::make_pair(true, *current_configuration);
        }

        bool start(iu456_error_details_t* error_details = nullptr)
        {
            return iu456_start(handle, error_details);
        }

        bool stop(iu456_error_details_t* error_details = nullptr)
        {
            return iu456_stop(handle, error_details);
        }

        bool get_last_frame(const iu456_frame_t** data, int32_t timeout, iu456_error_details_t* error_details = nullptr)
        {
            return iu456_get_last_frame(handle, data, timeout, error_details);
        }

        iu456_handle_t* get_handle()
        {
            return handle;
        }

    private:
        iu456_handle_t* handle;
    };

}
}
#endif

