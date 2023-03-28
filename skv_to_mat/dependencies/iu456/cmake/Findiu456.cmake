# - Try to find iu456 library
# Once done, this will define
#
#  iu456_FOUND                System has found iu456
#  iu456_INCLUDE_DIRS         The iu456 include directories
#  iu456_LIBRARIES            Link these to use iu456
#  iu456_BIN_FILES            Runtime libraries that needs to be installed
#  iu456_WINDOWS_DRIVERS_DIR  A directory holding the required windows drivers
#
# Required variables that must be set before calling this:
#
# iu456_DIR                   The root directory for the dependencies
# iu456_PLATFORM_NAME         The name of the platform you are targeting (e.g. "Windows7_x86_VS2010", "Linux_ARM_CortexA9_linaro", etc)
#
#
# Example usage:
#  set(iu456_DIR ${CMAKE_SOURCE_DIR}/deps/frame_grabbers/dist)
#  set(iu456_PLATFORM_NAME "Windows7_x86_VS2013")
#  find_package(iu456)
#
#  add_executable(my_iu456_based_app ${app_SOURCE_FILES})
#  target_include_directories(my_iu456_based_app PRIVATE ${iu456_INCLUDE_DIRS})
#  target_link_libraries(my_iu456_based_app PRIVATE ${iu456_LIBRARIES})
#  install(FILES ${iu456_BIN_FILES} DESTINATION ${my_bin_path})
#  install(DIRECTORY ${iu456_WINDOWS_DRIVERS_DIR} DESTINATION ${driver_path})

set(iu456_FOUND False)

if(NOT iu456_DIR)
  message("[iu456] iu456_DIR variable must be set to the directory package")
  set(iu456_FOUND False)
endif()

if( NOT EXISTS ${iu456_DIR})
  message("[iu456] Directory ${iu456_DIR} Does not exist.")
  set(iu456_FOUND False)
else()

  if ("${iu456_PLATFORM_NAME}" MATCHES "Windows")
    if ("${iu456_PLATFORM_NAME}" MATCHES "x64")
      set(_PLATFORM_NAME "Windows7_x64_VS2015")
    elseif("${iu456_PLATFORM_NAME}" MATCHES "x86")
      set(_PLATFORM_NAME "Windows7_x86_VS2015")
    else()
      message(FATAL_ERROR "Unsupported flavor: ${iu456_PLATFORM_NAME}")
    endif()
  else()
    set(_PLATFORM_NAME ${iu456_PLATFORM_NAME})
  endif()

  set( iu456_INCLUDE_DIRS ${iu456_DIR}/include/)
  set( iu456_LIB_SEARCH_PATH ${iu456_DIR}/lib/${_PLATFORM_NAME})
  set( iu456_BIN_SEARCH_PATH ${iu456_DIR}/bin/${_PLATFORM_NAME})

  set(iu456_name_shared iu456)

  if ("${_PLATFORM_NAME}" MATCHES "Windows")
    set( iu456_WINDOWS_DRIVERS_DIR ${iu456_DIR}/drivers)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
    find_library(iu456_LIBRARIES NAMES ${iu456_name_shared} HINTS ${iu456_LIB_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)

    find_library(iu456_binary NAMES ${iu456_name_shared} HINTS ${iu456_BIN_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)
    find_library(minicalc_binary NAMES minicalc HINTS ${iu456_BIN_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)

	if (USE_SSP500)
		# uvc SHOULD BE REMOVED when coding without uvc is done
		find_library(sgrabber_openni2_binary NAMES OpenNI2 HINTS ${iu456_BIN_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)
		find_library(sgrabber_usbclnt_binary NAMES usbclnt_api HINTS ${iu456_BIN_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)
		set(iu456_BIN_FILES
		  ${iu456_binary}
		  ${minicalc_binary}
		  ${sgrabber_openni2_binary}
		  ${sgrabber_usbclnt_binary})
	else()
		find_library(uvc_frame_grabber_binary NAMES uvc_frame_grabber HINTS ${iu456_BIN_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)
		set(iu456_BIN_FILES
		  ${iu456_binary}
		  ${minicalc_binary})
	endif()
  elseif("${_PLATFORM_NAME}" MATCHES "Linux_x86_64_gcc")
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")

    find_library(iu456_LIBRARIES NAMES ${iu456_name_shared} HINTS ${iu456_LIB_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)
    set(iu456_BIN_FILES
      ${iu456_LIBRARIES})
  else()
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib")

    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")

    find_library(iu456_LIBRARIES NAMES ${iu456_name_shared} HINTS ${iu456_LIB_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)
    find_library(minicalc_binary NAMES minicalc HINTS ${iu456_LIB_SEARCH_PATH} CMAKE_FIND_ROOT_PATH_BOTH)

    set(iu456_BIN_FILES
      ${iu456_LIBRARIES}
      ${minicalc_binary})
  endif()

  if (${iu456_LIBRARIES} STREQUAL "iu456_LIBRARIES-NOTFOUND")
    message("[iu456] Could not find libraries in ${iu456_DIR}.")
    set(iu456_FOUND False)
  else()
    set(iu456_FOUND True)
  endif()
endif()
