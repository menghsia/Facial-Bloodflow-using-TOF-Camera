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

#if !defined(SOFTKINETIC_SKV_SKV_EXCEPTION_INC)
#define SOFTKINETIC_SKV_SKV_EXCEPTION_INC

#pragma once

#include <softkinetic/skv/core.h>
#include <stdexcept>
#include <string>

namespace softkinetic
{
namespace skv
{
/**
 * \class skv_exception skv_exception.h softkinetic/skv/skv_exception.h
 * \brief An exception used to report errors from the SKV library.
 */
class skv_exception : public std::runtime_error
{
public:
	/**
	 * \brief Constructor.
	 *
	 * \param ec The SKV error code.
	 * \param message A descriptive error message.
	 */
	explicit skv_exception(skv_error_code ec, const char* message) : std::runtime_error(message), ec(ec) {}

	/**
	 * \brief The error code value.
	 *
	 * \returns The error code.
	 */
	skv_error_code get_error_code() const {
		return ec;
	}

	/**
	 * \brief The error message.
	 *
	 * \returns A string describing the type of error.
	 */
	std::string get_message() const {
		return skv_error_message(ec);
	}

private:

	skv_error_code ec;

};

#if defined(SKV_COMPILER_MSVC) && (_MSC_VER < 1900)
#define SDS_NO_EXCEPT(value)
#else
#define SDS_NO_EXCEPT(value) noexcept(value)
#endif

class throw_on_error
{
public:
	throw_on_error()
	{
	}
    
    ~throw_on_error() SDS_NO_EXCEPT(false)
	{
		if (error.code != skv_error_code_success)
		{
			throw skv_exception(error.code, error.message);
		}
	}

	operator skv_error*() { return &error; }

private:
	skv_error error;
};

}  // namespace skv
}  // namespace softkinetic

#endif  // SOFTKINETIC_SKV_SKV_EXCEPTION_INC
