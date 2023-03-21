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
 *  \cond ENABLE_PLAT_SYMS
 */

/**
    \file visibility.h

    \brief Library symbol visibility handling.
*/

#pragma once

#ifndef IU456_LIBRARY_VISIBILITY_H_INCLUDED_
#define IU456_LIBRARY_VISIBILITY_H_INCLUDED_

#include "iu456/platform.h"

// calling conventions
#if (IU456_LIBRARY_OS == IU456_LIBRARY_PLATFORM_x86_MSVC) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#define IU456_LIBRARY_DECL __stdcall
#define IU456_LIBRARY_CDECL __cdecl
#else
#define IU456_LIBRARY_DECL
#define IU456_LIBRARY_CDECL
#endif

// visibility
#if ((IU456_LIBRARY_PLATFORM == IU456_LIBRARY_PLATFORM_PS4_x86_64_CLANG) || (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_MSVC)) &&                    \
    !defined(IU456_LIBRARY_STATIC_LIB)
#define IU456_LIBRARY_DLL_EXPORT __declspec(dllexport)
#define IU456_LIBRARY_DLL_IMPORT __declspec(dllimport)
#define IU456_LIBRARY_DLL_LOCAL
#define IU456_LIBRARY_DLL_PROTECTED
#elif(IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_GCC || IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_ARMCC ||                                    \
      IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#if (IU456_LIBRARY_COMPILER_VERSION >= ((__GNUC__)*100)) || (IU456_LIBRARY_COMPILER_VERSION >= (500000))
#define IU456_LIBRARY_DLL_EXPORT __attribute__((visibility("default")))
#define IU456_LIBRARY_DLL_IMPORT __attribute__((visibility("default")))
#define IU456_LIBRARY_DLL_LOCAL __attribute__((visibility("hidden")))
#define IU456_LIBRARY_DLL_PROTECTED __attribute__((visibility("protected")))
#else
#define IU456_LIBRARY_DLL_EXPORT
#define IU456_LIBRARY_DLL_IMPORT
#define IU456_LIBRARY_DLL_LOCAL
#define IU456_LIBRARY_DLL_PROTECTED
#endif
#else
#define IU456_LIBRARY_DLL_EXPORT
#define IU456_LIBRARY_DLL_IMPORT
#define IU456_LIBRARY_DLL_LOCAL
#define IU456_LIBRARY_DLL_PROTECTED
#endif

#ifdef IU456_LIBRARY_DYN_LIB
#ifdef IU456_LIBRARY_EXPORTS
#define IU456_LIBRARY_API IU456_LIBRARY_DLL_EXPORT
#else
#define IU456_LIBRARY_API IU456_LIBRARY_DLL_IMPORT
#endif
#else
#define IU456_LIBRARY_API IU456_LIBRARY_DLL_IMPORT
#endif

#endif

/**
 *  \endcond
 */
