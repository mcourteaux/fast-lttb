project(lttb)
cmake_minimum_required(VERSION 2.8)

add_library(lttb STATIC "lttb.cpp")
add_subdirectory(lib/gtest)

