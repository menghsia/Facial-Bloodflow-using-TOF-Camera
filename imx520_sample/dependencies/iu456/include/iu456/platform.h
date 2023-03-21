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

#ifndef IU456_LIBRARY_PLATFORM_H_INCLUDED_
#define IU456_LIBRARY_PLATFORM_H_INCLUDED_

/**
 *	\cond ENABLE_PLAT_SYMS
 */

/**
 * \file platform.h
 * \brief platform specific definitions
 */

#ifdef _WIN32
#pragma warning(push, 0)
#endif

// Supported compilers
#define IU456_LIBRARY_COMPILER_MSVC 314159u
#define IU456_LIBRARY_COMPILER_GCC 161803u
#define IU456_LIBRARY_COMPILER_INTEL 250290u
#define IU456_LIBRARY_COMPILER_ARMCC 466920u
#define IU456_LIBRARY_COMPILER_ANDROID 271828u
#define IU456_LIBRARY_COMPILER_CLANG 764223u

// Supported architectures
#define IU456_LIBRARY_ARCH_ARMv5 3u
#define IU456_LIBRARY_ARCH_ARMv7 5u
#define IU456_LIBRARY_ARCH_ARM64 9u
#define IU456_LIBRARY_ARCH_x86 17u
#define IU456_LIBRARY_ARCH_x86_64 33u

// Supported OSes
#define IU456_LIBRARY_OS_WINDOWS 1u
#define IU456_LIBRARY_OS_LINUX 2u
#define IU456_LIBRARY_OS_ANDROID 3u
#define IU456_LIBRARY_OS_PS4 5u
#define IU456_LIBRARY_OS_MACOS 8u

// Supported platforms
#define IU456_LIBRARY_PLATFORM_WINDOWS_x86_MSVC 1u
#define IU456_LIBRARY_PLATFORM_WINDOWS_x86_64_MSVC 2u
#define IU456_LIBRARY_PLATFORM_WINDOWS_x86_INTEL 3u
#define IU456_LIBRARY_PLATFORM_LINUX_x86_GCC 4u
#define IU456_LIBRARY_PLATFORM_LINUX_x86_64_GCC 5u
#define IU456_LIBRARY_PLATFORM_LINUX_x86_CLANG 6u
#define IU456_LIBRARY_PLATFORM_LINUX_x86_64_CLANG 7u
#define IU456_LIBRARY_PLATFORM_PS4_x86_64_CLANG 8u
#define IU456_LIBRARY_PLATFORM_ANDROID_ARMv7_GCC 9u
#define IU456_LIBRARY_PLATFORM_ANDROID_ARMv7_CLANG 10u
#define IU456_LIBRARY_PLATFORM_LINUX_ARMv5_GCC 11u
#define IU456_LIBRARY_PLATFORM_LINUX_ARMv7_GCC 12u
#define IU456_LIBRARY_PLATFORM_MACOS_x86_CLANG 13u
#define IU456_LIBRARY_PLATFORM_MACOS_x86_64_CLANG 14u
#define IU456_LIBRARY_PLATFORM_ANDROID_ARMv8a_64_GCC 15u
#define IU456_LIBRARY_PLATFORM_ANDROID_ARMv8a_64_CLANG 16u

/* Find the OS.
*/
#if defined(__WIN64__) || defined(_WIN64) || defined(_WINDOWS) || defined(__WIN32__) || defined(_WIN32)
#define IU456_LIBRARY_OS IU456_LIBRARY_OS_WINDOWS
#elif (defined(__linux__) || defined(__LINUX__) || defined(linux) || defined(__linux)) && !defined(__ANDROID__)
#define IU456_LIBRARY_OS IU456_LIBRARY_OS_LINUX
#elif defined(__ANDROID__)
#define IU456_LIBRARY_OS IU456_LIBRARY_OS_ANDROID
#elif defined(__ORBIS__)
#define IU456_LIBRARY_OS IU456_LIBRARY_OS_PS4
#elif defined(__APPLE__) || defined(__MACH__)
#define IU456_LIBRARY_OS IU456_LIBRARY_OS_MACOS
#else
#error "Compilation error: Unsupported OS."
#endif

/* Find the architecture.
 */
#if defined(__arm__) || defined(__thumb__) || defined(_M_ARM) || defined(__TARGET_ARCH_ARM) || defined(__aarch64__)
#if (__TARGET_ARCH_ARM == 5 || __ARM_ARCH_5TEJ__ == 1 || __ARM_ARCH_5TE__ == 1 || __ARM_ARCH_5T__ == 1)
#define IU456_LIBRARY_ARCH IU456_LIBRARY_ARCH_ARMv5
#elif(__TARGET_ARCH_ARM == 7 || __ARM_ARCH_7A__ == 1)
#define IU456_LIBRARY_ARCH IU456_LIBRARY_ARCH_ARMv7
#elif defined(__aarch64__)
#define IU456_LIBRARY_ARCH IU456_LIBRARY_ARCH_ARM64
#else
#error "Compilation error: Unsupported ARM architecture variant."
#endif
#elif defined(__x86) || defined(__x86__) || defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(_M_IX86)
#define IU456_LIBRARY_ARCH IU456_LIBRARY_ARCH_x86
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(__amd64) || defined(_M_X64)
#define IU456_LIBRARY_ARCH IU456_LIBRARY_ARCH_x86_64
#else
#error "Compilation error: Unsupported architecture."
#endif

/* Find the compiler type and version.
*/
// @TODO Intel compiler check needs to be added
#if defined(_MSC_VER)
#define IU456_LIBRARY_COMPILER IU456_LIBRARY_COMPILER_MSVC
#define IU456_LIBRARY_COMPILER_VERSION _MSC_VER
#elif defined(__arm__) && defined(__ARMCC_VERSION)
#define IU456_LIBRARY_COMPILER IU456_LIBRARY_COMPILER_ARMCC
/* the format is PVbbbb - P is the major version, V is the minor version,
 bbbb is the build number*/
#define IU456_LIBRARY_COMPILER_VERSION (__ARMCC_VERSION)
#elif defined(__GNUC__)
// gcc family of compilers: gcc, clang
#define IU456_LIBRARY_GEN_VERSION(major, minor, patch) (((major)*100) + ((minor)*10) + (patch))

#if defined(__clang__)
#define IU456_LIBRARY_COMPILER IU456_LIBRARY_COMPILER_CLANG
#define IU456_LIBRARY_COMPILER_VERSION IU456_LIBRARY_GEN_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#define IU456_LIBRARY_COMPILER IU456_LIBRARY_COMPILER_GCC
#define IU456_LIBRARY_COMPILER_VERSION IU456_LIBRARY_GEN_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#endif
#else
#error "Compilation error: Unsupported compiler."
#endif

// Define the platform.
#if (IU456_LIBRARY_OS == IU456_LIBRARY_OS_WINDOWS) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86) &&                                                    \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_MSVC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_WINDOWS_x86_MSVC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_WINDOWS) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86_64) &&                                                \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_MSVC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_WINDOWS_x86_64_MSVC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_LINUX) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86) &&                                                     \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_GCC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_LINUX_x86_GCC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_LINUX) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86_64) &&                                                  \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_GCC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_LINUX_x86_64_GCC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_LINUX) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86) &&                                                     \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_LINUX_x86_CLANG
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_LINUX) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86_64) &&                                                  \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_LINUX_x86_64_CLANG
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_LINUX) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_ARMv5) &&                                                   \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_GCC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_LINUX_ARMv5_GCC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_LINUX) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_ARMv7) &&                                                   \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_GCC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_LINUX_ARMv7_GCC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_ANDROID) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_ARMv7) &&                                                 \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_GCC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_ANDROID_ARMv7_GCC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_ANDROID) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_ARMv7) &&                                                 \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_ANDROID_ARMv7_CLANG
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_ANDROID) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_ARM64) &&                                                 \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_GCC)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_ANDROID_ARMv8_64_GCC
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_ANDROID) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_ARM64) &&                                                 \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#define FRAME_GRABBERS_PLATFORM FRAME_GRABBERS_PLATFORM_ANDROID_ARMv8_64_CLANG
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_PS4) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86_64) &&                                                    \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_PS4_x86_64_CLANG
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_MACOS) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86) &&                                                     \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_MACOS_x86_CLANG
#elif(IU456_LIBRARY_OS == IU456_LIBRARY_OS_MACOS) && (IU456_LIBRARY_ARCH == IU456_LIBRARY_ARCH_x86_64) &&                                                  \
    (IU456_LIBRARY_COMPILER == IU456_LIBRARY_COMPILER_CLANG)
#define IU456_LIBRARY_PLATFORM IU456_LIBRARY_PLATFORM_MACOS_x86_64_CLANG
#else
#error "Compilation error: Unsupported platform."
#endif

#ifdef _WIN32
#pragma warning(pop)
#endif

/**
 *	\endcond
 */

#endif
