cmake_minimum_required(VERSION 3.16)
project(mapf_shortest_paths LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(mapf_shortest_paths MODULE shortest_paths.cpp)
target_compile_features(mapf_shortest_paths PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(mapf_shortest_paths PRIVATE /O2)
else()
    target_compile_options(
        mapf_shortest_paths
        PRIVATE
        -O3
        -DNDEBUG
    )
endif()

set_target_properties(
    mapf_shortest_paths
    PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
