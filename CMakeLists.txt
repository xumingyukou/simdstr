cmake_minimum_required(VERSION 3.1)
project(SIMDTour)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add naivestr librariy
add_library(naivestr SHARED src/naivestr.c)
target_include_directories(naivestr PUBLIC include/)
target_compile_options(naivestr PRIVATE -O3 -Wall -Werror -Wextra -mno-avx2 -mno-avx512f -g)

# add simdstr librariy
add_library(simdstr SHARED src/simdstr.c)
target_link_libraries(simdstr PRIVATE naivestr)
target_include_directories(simdstr PUBLIC include/)
target_compile_options(simdstr PRIVATE -O3 -Wall -Werror -Wextra -march=native -g)

# add google test
set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
add_subdirectory(thirdparty/googletest)
add_subdirectory(tests)

# add google benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(thirdparty/benchmark)
add_subdirectory(bench)
add_subdirectory(examples)
