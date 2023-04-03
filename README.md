
# simdstr

Some string kernels accelerated by SIMD.

Requirements:

1. Intel x86_64.
2. CMake 3.1 or above.

Install:

```
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j
```

Test:

```
./build/tests/test_str
```

Bench:

```
./build/bench/bm_str
```

