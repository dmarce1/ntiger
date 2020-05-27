# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)

# Libraries linked via full path no longer produce linker search paths.
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)

project(ntiger CXX)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(Boost REQUIRED)

set(common_source_files
    src/gravity.cpp
	src/math.cpp
	src/particle.cpp
	src/rand.cpp
	src/options.cpp
)

set(ewald_source_files
	src/ewald_main.cpp
)

set(ntiger_source_files
	src/initialize.cpp
	src/main.cpp
	src/range.cpp
	src/profiler.cpp
	src/tree/tree_infrastructure.cpp
	src/tree/tree_compute.cpp
	src/tree/tree_gravity.cpp
	src/tree/tree_stats.cpp
)

set(check2silo_source_files
    src/silo.cpp
)

set(common_header_files
	ntiger/dim.hpp
	ntiger/options.hpp
	ntiger/particle.hpp
	ntiger/real.hpp
	ntiger/vect.hpp
	ntiger/gravity.hpp
)


set(ntiger_header_files
	ntiger/initialize.hpp
	ntiger/range.hpp
	ntiger/rand.hpp
	ntiger/tree.hpp
	ntiger/tree_stats.hpp
)

set(check2silo_header_files
)

add_library(
   ntigerlib
    ${common_source_files}
    ${common_header_files}
)  

add_hpx_executable(
  ntiger
  DEPENDENCIES
    Silo::silo Boost::boost ntigerlib
  SOURCES
    ${ntiger_source_files}
  HEADERS
    ${ntiger_header_files}
 )

add_hpx_executable(
  check2silo
  DEPENDENCIES
    Silo::silo Boost::boost ntigerlib
  SOURCES
    ${check2silo_source_files}
  HEADERS
    ${check2silo_header_files}
)

add_hpx_executable(
  ewald
  DEPENDENCIES
    Silo::silo Boost::boost ntigerlib
  SOURCES
    ${ewald_source_files}
  HEADERS
    ${ewald_header_files}
)

target_compile_options(ntigerlib PUBLIC -std=c++1z)
target_include_directories(ntigerlib PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(ntiger PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(check2silo PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(ewald PUBLIC ${PROJECT_SOURCE_DIR})


