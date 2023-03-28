/***************************************************************************************/
//  SoftKinetic SKV library
//  Project Name      : SKV
//  Module Name		  : SKV api platform v2
//  Description       : Platform configuration for SKV v2
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

#if !defined(SOFTKINETIC_SKV_CORE_PLATFORM_INC)
#define SOFTKINETIC_SKV_CORE_PLATFORM_INC

#pragma once

// Supported compiler
#define SKV_COMPILER_MSVC 	1
#define SKV_COMPILER_GCC  	2
#define SKV_COMPILER_INTEL 	3
#define SKV_COMPILER_ARMCC 	4
#define SKV_COMPILER_CLANG 	5
#define SKV_COMPILER_GREENHILLS  6 

// -- unsupported yet
// #define SKV_COMPILER_BORLAND 3
// #define SKV_COMPILER_CWCC    4

// Supported platform
#define SKV_PLATFORM_WIN32       	1
#define SKV_PLATFORM_LINUX_x86   	2
#define SKV_PLATFORM_LINUX_ARMv5 	3
#define SKV_PLATFORM_LINUX_ARMv7	4
#define SKV_PLATFORM_WIN64       	5
#define SKV_PLATFORM_LINUX_x86_64	6
#define SKV_PLATFORM_OSX_x86_64		7
#define SKV_PLATFORM_ANDROID_ARMv7  8
#define SKV_PLATFORM_PS4			9
#define SKV_PLATFORM_GREENHILLS_ARMv7	10
#define SKV_PLATFORM_LINUX_ARMv8a_64	11

// -- unsupported yet
//SKV_PLATFORM_WIN64
//SKV_PLATFORM_PS3
//SKV_PLATFORM_X360
//SKV_PLATFORM_FREEBSD


/* Finds the compiler type and version.
*/
//TODO manage minimal compiler version supported
#if defined( _MSC_VER )
	#define SKV_COMPILER SKV_COMPILER_MSVC
	#define SKV_COMPILER_VERSION _MSC_VER
#elif defined(__arm__) && defined(__ARMCC_VERSION)
	#define SKV_COMPILER SKV_COMPILER_ARMCC
	/* the format is PVbbbb - P is the major version, V is the minor version,
 	 bbbb is the build number*/
	#define SKV_COMPILER_VERSION (__ARMCC_VERSION)
#elif defined( __GNUC__ )

	#define SKV_GEN_VERSION(major, minor, patch) (((major)*100) + \
                     ((minor)*10) + \
                     (patch))

	#if defined(__clang__)
		#define SKV_COMPILER SKV_COMPILER_CLANG
		#define SKV_COMPILER_VERSION SKV_GEN_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
	#else
		#define SKV_COMPILER SKV_COMPILER_GCC
		#define SKV_COMPILER_VERSION SKV_GEN_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)

	#endif
#else
	#error "Compilation error: Unsupported compiler."
#endif

/* Finds the platform - architecture
*/
#if defined( __WIN64__ ) || defined( _WIN64 )
	#define SKV_PLATFORM SKV_PLATFORM_WIN64
#elif defined( __WIN32__ ) || defined( _WIN32 )
	#define SKV_PLATFORM SKV_PLATFORM_WIN32
#elif defined(__linux__) || defined(__LINUX__)
	#define SKV_PLATFORM_LINUX (1)

	#if (__TARGET_ARCH_ARM == 5 || __ARM_ARCH_5TEJ__ == 1 || __ARM_ARCH_5TE__ == 1 || __ARM_ARCH_5T__ == 1)
		#define SKV_PLATFORM SKV_PLATFORM_LINUX_ARMv5
	#elif (__TARGET_ARCH_ARM == 7 || __ARM_ARCH_7A__ == 1)
		#define SKV_PLATFORM SKV_PLATFORM_LINUX_ARMv7
	#elif(__ARM_ARCH == 8 || __ARM_ARCH_8A == 1)
		#if defined(__arm64) || defined(__ARM_ARCH_ISA_A64)
			#define SKV_PLATFORM SKV_PLATFORM_ARMv8a_64
		#endif
	#elif defined(__x86) || defined(__x86__) || defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__)
		#define SKV_PLATFORM SKV_PLATFORM_LINUX_x86
	#elif defined(__amd64__) || defined(__x86_64__)
		#define SKV_PLATFORM SKV_PLATFORM_LINUX_x86_64
	#elif defined(__ghs__)
		#define SKV_COMPILER SKV_COMPILER_GREENHILLS
		#define SKV_COMPILER_VERSION (__GHS_VERSION_NUMBER__)
	#else
		#error "Compilation error: Unsupported version of Linux platform."
	#endif
#elif defined(__APPLE__)
	#define SKV_PLATFORM_OSX (1)

	#if defined(__amd64__) || defined(__x86_64__)
		#define SKV_PLATFORM SKV_PLATFORM_OSX_x86_64
	#else
		#error "Compilation error: Unsupported version of Mac OS X platform."
	#endif
#elif defined(__ORBIS__)
	#define SKV_PLATFORM SKV_PLATFORM_PS4
#elif defined(__ghs__)
	#define SKV_PLATFORM SKV_PLATFORM_GREENHILLS_ARMv7
#else
	#error "Compilation error: Unsupported platform."
#endif

// -------- CALLING CONVENTIONS ------------------------

#if ((SKV_PLATFORM == SKV_PLATFORM_WIN32) || (SKV_PLATFORM == SKV_PLATFORM_WIN64)) && !defined(__MINGW32__) && !defined(__CYGWIN__)
	#define SKV_SDK_DECL  __stdcall
	#define SKV_SDK_CDECL  __cdecl
#else
	#define SKV_SDK_DECL
	#define SKV_SDK_CDECL
#endif

//-------------- DLL Export DEFINITIONS ------------

#ifdef SKV_STATIC_LIB
	#define SKV_DLL_EXPORT
	#define SKV_DLL_IMPORT
#else
	#if (SKV_COMPILER == SKV_COMPILER_MSVC)
		#define SKV_DLL_EXPORT
		#define SKV_DLL_IMPORT __declspec(dllimport)
	#elif (SKV_COMPILER == SKV_COMPILER_GCC || SKV_COMPILER == SKV_COMPILER_ARMCC)
		#if (SKV_COMPILER_VERSION >= ((__GNUC__) * 100)) || (SKV_COMPILER_VERSION >= (500000))
			#define SKV_DLL_EXPORT 		__attribute__ ((visibility ("default")))
			#define SKV_DLL_IMPORT 		__attribute__ ((visibility ("default")))
		#else
			#define SKV_DLL_EXPORT
			#define SKV_DLL_IMPORT
		#endif
	#else
		#define SKV_DLL_EXPORT
		#define SKV_DLL_IMPORT
	#endif
#endif

// import/export macros
#ifdef SKV_EXPORTS
#define SKV_API SKV_DLL_EXPORT
#else
#define SKV_API SKV_DLL_IMPORT
#endif

#endif // SOFTKINETIC_SKV_PLATFORM_INCLUDED
