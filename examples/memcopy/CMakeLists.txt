add_executable(bm_memcopy bm_memcopy.cpp)
target_compile_options(bm_memcopy PRIVATE -march=native -O1 -Wall -Wextra -Werror -save-temps -g)
target_link_libraries(bm_memcopy PRIVATE benchmark::benchmark)