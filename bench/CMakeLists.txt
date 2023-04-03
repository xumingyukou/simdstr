add_executable(bm_str bm_str.cpp)
target_compile_options(bm_str PRIVATE -march=native -O3 -Wall -Wextra -Werror -g)
target_link_libraries(bm_str PRIVATE naivestr simdstr benchmark::benchmark)