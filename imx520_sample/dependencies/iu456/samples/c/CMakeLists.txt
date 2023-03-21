
cmake_minimum_required (VERSION 2.8)
project(iu456_sample)

if ("${SK_PLATFORM_NAME}" MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/)

set(iu456_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(iu456_PLATFORM_NAME ${SK_PLATFORM_NAME})
find_package(iu456)

if(NOT iu456_FOUND)
    message(FATAL_ERROR "Failed to find iu456 dependency")
endif()

add_executable(iu456_sample main.cpp)

target_include_directories(iu456_sample PRIVATE ${iu456_INCLUDE_DIRS})
target_link_libraries(iu456_sample PRIVATE ${iu456_LIBRARIES})

install(TARGETS iu456_sample RUNTIME DESTINATION bin)
install(FILES ${iu456_BIN_FILES} DESTINATION bin)
