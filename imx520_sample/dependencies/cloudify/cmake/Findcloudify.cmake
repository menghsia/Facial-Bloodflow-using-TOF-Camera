# - Try to find cloudify_lib
# Once done, this will define
#
#  cloudify_lib_FOUND                 System has cloudify_lib
#  cloudify_lib_INCLUDE_DIRS          The cloudify_lib include directories
#  cloudify_lib_LIBRARIES             Link these to use cloudify_lib
#  cloudify_lib_BINARIES              The cloudify_lib binaries
#
#
# Required variables that must be set before calling this:
#
# SK_DEPENDENCIES_DIR   The root directory for the dependencies
# cloudify_lib_PLATFORM_NAME      The name of the platform you are targeting (e.g. "Windows_Win7_x86_VS2010", "Linux_ARM_CortexA9_linaro", etc)
#
#
# Example usage:
#  set(SK_DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)
#  set(SK_DEPENDENCIES_DIR "Windows_Win7_x86_VS2010")
#  find_package(cloudify_lib)
#
#  include_directories(${cloudify_lib_INCLUDE_DIRS})
#  link_directories(${cloudify_lib_LIB_DIRS})
#  add_executable(my_skv_based_app ${cloudify_lib_LIBRARIES})


unset(cloudify_lib_FOUND)

if(NOT cloudify_lib_DIR)
	message(ERROR "Could not find a suitable root directory")
endif()

set(cloudify_lib_INCLUDE_DIRS ${cloudify_lib_DIR}/include)
set(cloudify_lib_LIBRARIES ${cloudify_lib_DIR}/lib/cloudify.lib)
set(cloudify_lib_BINARIES ${cloudify_lib_DIR}/bin/cloudify.dll)

set(cloudify_lib_FOUND true)