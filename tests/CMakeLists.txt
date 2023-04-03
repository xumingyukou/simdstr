
add_executable(test_str test_str.cpp)
target_compile_options(test_str PRIVATE -march=native -O3 -Wall -Wextra -Werror -g)
target_link_libraries(test_str PRIVATE naivestr simdstr gtest_main)

include(GoogleTest)
gtest_discover_tests(test_str)