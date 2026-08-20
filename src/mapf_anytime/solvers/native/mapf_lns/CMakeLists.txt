cmake_minimum_required(VERSION 3.16)
project(lns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB SOURCES "src/*.cpp" "src/CBS/*.cpp" "src/PIBT/*.cpp")
add_executable(lns ${SOURCES})
target_include_directories(lns PRIVATE "inc" "inc/CBS" "inc/PIBT")

find_package(Boost CONFIG REQUIRED COMPONENTS program_options system filesystem)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(
    lns
    PRIVATE
    Boost::program_options
    Boost::system
    Boost::filesystem
    Eigen3::Eigen
)
