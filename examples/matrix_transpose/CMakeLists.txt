add_executable(bm_matrix bm_matrix.cpp)
target_compile_options(bm_matrix PRIVATE -march=native -O3 -Wall -Wextra -Werror -save-temps)
target_link_libraries(bm_matrix PRIVATE benchmark::benchmark)