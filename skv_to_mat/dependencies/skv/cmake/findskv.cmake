# - Try to find skv
# Once done, this will define
#
#  skv_FOUND                 System has skv
#  skv_INCLUDE_DIRS          The skv include directories
#  skv_LIBRARIES             Link these to use skv
#  skv_DEBUG_LIBRARIES       Link these to use skv (debug version)
#  skv_CXX_FLAGS_RELEASE     Release link flags used to build skv
#  skv_CXX_FLAGS_DEBUG       Debug link flags used to build skv
#  skv_CXX_FLAGS             Common link flags
#  skv_COMPILE_DEFINITIONS   Common preprocessor defines
#
#
# Required variables that must be set before calling this:
#
# SK_DEPENDENCIES_DIR   The root directory for the dependencies
# skv_PLATFORM_NAME      The name of the platform you are targeting (e.g. "Windows_Win7_x86_VS2010", "Linux_ARM_CortexA9_linaro", etc)
#
#
# Example usage:
#  set(SK_DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)
#  set(SK_DEPENDENCIES_DIR "Windows_Win7_x86_VS2010")
#  find_package(skv)
#
#  include_directories(${skv_INCLUDE_DIRS})
#  link_directories(${skv_LIB_DIRS})
#  add_executable(my_skv_based_app ${skv_LIBRARIES})

unset(skv_FOUND)

if(NOT skv_DIR)
	message(ERROR "Could not find a suitable root directory")
endif()

set(skv_INCLUDE_DIRS ${skv_DIR}/include)

set(skv_LIBRARIES 
	${skv_DIR}/lib/skv.lib
	${skv_DIR}/lib/h5lz4.lib
)

set(skv_BINARIES 
	${skv_DIR}/bin/skv.dll
	${skv_DIR}/bin/h5lz4.dll
)

set(skv_FOUND true)