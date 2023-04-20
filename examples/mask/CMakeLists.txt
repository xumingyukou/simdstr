add_executable(bm_mask bm_mask.cpp)
target_compile_options(bm_mask PRIVATE -march=native -O3 -Wall -Wextra -Werror -save-temps)
target_link_libraries(bm_mask PRIVATE benchmark::benchmark)