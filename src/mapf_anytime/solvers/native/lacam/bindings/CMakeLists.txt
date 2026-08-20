pybind11_add_module(lacam MODULE NO_EXTRAS pybind.cpp)
target_link_libraries(lacam PRIVATE lacam3)
