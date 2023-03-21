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

#include <iostream>
#include <cassert>

#include "iu456/iu456.h"

namespace
{
void display_average_of_valid_depth(const iu456_frame_t* frame_data)
{
    // This is an example on how to traverse the depth map
    // and how to tell whether a point is valid or not
    //
    // Note that computing the average over the whole image
    // doesn't make much sense.

    const int16_t* depth = frame_data->depth;
    
    // It is preferable to get the image traits such as the
    // width, height or unambiguous range from the configuration_info
    // pointer provided with the frame data. This is because
    // since one can change the module configuration without
    // interrupting the stream, nothing guarantees that a call
    // to iu456_get_current_configuration will provide the matching
    // configuration.
    int image_width = frame_data->configuration_info->width;
    int image_height = frame_data->configuration_info->height;

    // The number of elements in the array should match the width * height
    assert( static_cast<size_t>(image_width * image_height) == frame_data->depth_length );

    float sum_of_valid_depth = 0.0f;
    int number_of_valid_pixel = 0;

    // values above this threshold are invalid values
    const int16_t invalid_value_threshold = 32000;

    for (int h = 0; h < image_height; ++h)
    {
        for (int w = 0; w < image_width; ++w)
        {
            auto value = depth[h * image_width + w];
            if ( value < invalid_value_threshold )
            {
                sum_of_valid_depth += static_cast<float>(value);
                number_of_valid_pixel++;
            }
        }
    }

    std::cout << "Average distance of valid depth: ";
    std::cout << sum_of_valid_depth / number_of_valid_pixel;
    std::cout << " millimeters" << std::endl;
}
}


int main()
{
    // First initialize the library.
    // This must be called before any other operations on the library
    iu456_error_details_t error_details;
    if (!iu456_initialize(nullptr, nullptr, nullptr, nullptr, &error_details))
    {
        std::cerr << "Failed initialize library: " << error_details.message << std::endl;
        iu456_shutdown(nullptr);
        return -1;
    }

    iu456_handle_t* handle = nullptr;

    // Instantiate the first available camera.
    // This may take some seconds to instantiate.
    if (!iu456_create(&handle, nullptr, nullptr, nullptr, &error_details))
    {
        std::cerr << "Failed to initialize device: " << error_details.message << std::endl;
        iu456_shutdown(nullptr);
        return -1;
    }

    std::cout << "Starting stream" << std::endl;
    // Start the streaming.
    if (!iu456_start(handle, &error_details))
    {
        std::cerr << "Failed to start stream: " << error_details.message << std::endl;
        iu456_destroy(handle, &error_details);
        iu456_shutdown(nullptr);
        return -1;
    }

    std::cout << "Stream started" << std::endl;

    // We will grab up to 25 frames unless an error occurs during the streaming.
    int counter = 0;
    const iu456_frame_t* frame_data = nullptr;
    int last_frame_id = -1;
    int total_missed_frames = 0;

    while (counter < 25)
    {
        // Get the last available frame.
        // We are using an infinite timeout (3rd argument of the function). Note that you can 
        // shorter timeout in the case you need. The timeout is given in milliseconds.
        if (iu456_get_last_frame(handle, &frame_data, -1, &error_details))
        {
            // We successfully received a new frame.
            // The frame_data pointer will hold a number of items.
            // timestamp: the time in milliseconds at which the frame was received 
            // since an unspecified point in time.
            //
            // depth: a pointer to an int16_t array holding the cartesian depth in millimeters.
            // Values exceeding 32000 should be treated as invalid pixels.
            //
            // depth_length: the number of elements (not bytes) in the depth array.
            //
            // confidence: a pointer to an int16_t array holding the confidence.
            //
            // confidence_length: the number of elements (not bytes) in the confidence array.
            //
            // laser_temperature: the laser temperature in celsius degrees.
            //
            // configuration_info: a pointer to a iu456_configuration_info_t object
            // holding some meta data such as the width and height of the frame and the lens model.
            std::cout << "Received frame " << counter;
            std::cout << " with timestamp: " << frame_data->timestamp << " and frame_id: " << frame_data->frame_id << std::endl;
            if (last_frame_id != -1)
            {
                auto missed_frames = frame_data->frame_id - (last_frame_id + 1);
                if (missed_frames > 0)
                {
                    std::cout << "missed frames since last call: " << missed_frames << std::endl;
                    total_missed_frames += missed_frames;
                }
                else if (missed_frames < 0)
                {
                    std::cerr << "There is something wrong with missed frames: " << missed_frames << std::endl;
                }
            }
            last_frame_id = frame_data->frame_id;

            display_average_of_valid_depth(frame_data);
            
            counter++;
        }
        else
        {
            // We failed to get a frame. Exiting the loop
            // Note that in the case the timeout has been reached (which is not the case here
            // since we are using an infinite timeout), the iu456_get_last_frame will return false.
            // To know whether this is a timeout or an error you can check the error_details.code 
            // is equal to iu456_error_timeout.
            std::cerr << error_details.message << std::endl;
            break;
        }
    }

    std::cout << "Total missed frames: " << total_missed_frames << std::endl;
    
    // Stop the streaming.
    if (!iu456_stop(handle, &error_details))
    {
        std::cerr << "Failed to stop stream: " << error_details.message << std::endl;
        iu456_shutdown(nullptr);
        return -1;
    }

    // Destroy the camera
    iu456_destroy(handle, &error_details);

    // Shut down the library
    iu456_shutdown(nullptr);
    return 0;
}
