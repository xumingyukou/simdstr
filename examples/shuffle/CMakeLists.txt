add_executable(bm_shuffle bm_shuffle.cpp)
target_compile_options(bm_shuffle PRIVATE -march=native -O1 -Wall -Wextra -Werror -save-temps -g)
target_link_libraries(bm_shuffle PRIVATE benchmark::benchmark)