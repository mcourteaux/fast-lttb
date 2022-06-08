project(fast-lttb)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/tracy)
add_library(TracyHeaders INTERFACE)
target_include_directories(TracyHeaders INTERFACE "lib/tracy/")
add_definitions(-DTRACY_ENABLE)

add_library(fast-lttb STATIC "lttb.cpp")
target_compile_options(fast-lttb PRIVATE "-mavx2")
target_include_directories(fast-lttb PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fast-lttb PRIVATE TracyHeaders)
add_subdirectory(lib/gtest)

add_executable(test-lttb "test.cpp")
target_link_libraries(test-lttb PUBLIC
  fast-lttb
  TracyClient
  gtest_main
  )
