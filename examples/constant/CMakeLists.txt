add_executable(constant constant.cpp)
target_compile_options(constant PRIVATE -march=native -O1 -Wall -Wextra -Werror -save-temps)