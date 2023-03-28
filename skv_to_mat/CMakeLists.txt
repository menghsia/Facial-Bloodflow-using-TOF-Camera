cmake_minimum_required(VERSION 3.3)

project(imx520_sample)

## ----------------------------------------------------------------------------
## Output directories
## ----------------------------------------------------------------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/.build/output)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

set(iu456_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/iu456)
set(cloudify_lib_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cloudify)
set(iu456_PLATFORM_NAME Windows7_x64_VS2015)
set(skv_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/skv)
set(skv_helper_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/skv_helper)



set(CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH}
	${iu456_DIR}/cmake
	${cloudify_lib_DIR}/cmake
	${skv_DIR}/cmake


)

find_package(iu456 REQUIRED)
find_package(Matlab REQUIRED)
find_package(cloudify REQUIRED)
find_package(skv REQUIRED)

INCLUDE_DIRECTORIES("${Matlab_ROOT_DIR}/extern/include")

# -----------------------------------------------------------------------------
# Build target
# -----------------------------------------------------------------------------

add_executable(imx520_sample
	sources/main.cpp
)

target_include_directories(imx520_sample PRIVATE
	${iu456_INCLUDE_DIRS}
	${cloudify_lib_INCLUDE_DIRS}
	${skv_INCLUDE_DIRS}
	${skv_helper_DIR}
)

target_link_libraries(imx520_sample PRIVATE
	${iu456_LIBRARIES}
	${cloudify_lib_LIBRARIES}
	${skv_LIBRARIES}
	${Matlab_ROOT_DIR}/extern/lib/win64/microsoft/libmex.lib
	${Matlab_ROOT_DIR}/extern/lib/win64/microsoft/libmat.lib
	${Matlab_ROOT_DIR}/extern/lib/win64/microsoft/libmx.lib
	${Matlab_ROOT_DIR}/extern/lib/win64/microsoft/libeng.lib
	
)



target_compile_definitions(imx520_sample PRIVATE
)

# -----------------------------------------------------------------------------
# Copy binaries
# -----------------------------------------------------------------------------

foreach(file ${iu456_BIN_FILES} ${cloudify_lib_BINARIES} ${skv_BINARIES})
	add_custom_command(TARGET imx520_sample POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file} $<TARGET_FILE_DIR:imx520_sample>)
endforeach()
