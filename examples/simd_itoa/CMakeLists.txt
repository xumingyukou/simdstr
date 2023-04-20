add_executable(bm_itoa bm_itoa.cpp)
target_compile_options(bm_itoa PRIVATE -march=native -O3 -Wall -Wextra -Werror -save-temps)
target_link_libraries(bm_itoa PRIVATE benchmark::benchmark)