project(imx520_sample)

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------
set(source_FILES
	main.cpp
)

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
add_executable(imx520_sample
	${source_FILES}
)

target_include_directories(imx520_sample PRIVATE
	${iu456_INCLUDE_DIRS}	
	${Matlab_ROOT_DIR}/extern/include
)

target_link_libraries(imx520_sample PRIVATE
	${iu456_LIBRARIES}
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

foreach(file ${iu456_BIN_FILES})
	add_custom_command(TARGET imx520_sample POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file} $<TARGET_FILE_DIR:imx520_sample>)
endforeach()
