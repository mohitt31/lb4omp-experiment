cmake_minimum_required(VERSION 3.14)
project(lb4omp_experiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(sumfact_harness sumfact_harness.cpp)
target_link_libraries(sumfact_harness PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(sumfact_harness PRIVATE -O3 -march=native -Wall)
