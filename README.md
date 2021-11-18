# npuemulator
A neural network fast inference library implementing Coral Edge TPU emulator using AVX2.
# Usage
For building npuemulator library type following lines in your terminal:
```
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --config Release
```
CMake builds **npuemul.lib** file in **build** directory. Header files are placed in **include** directory.
CMake builds **tests** and **benchmarks** for testing emulator in the current environment.
