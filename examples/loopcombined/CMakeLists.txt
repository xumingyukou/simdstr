find_package(OpenMP REQUIRED)

add_executable(bm_loopcombined bm_loopcombined.cpp)
target_compile_options(bm_loopcombined PRIVATE -march=native -O1 -Wall -Wextra -Werror -save-temps)
target_link_libraries(bm_loopcombined PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)