Build instructions
------------------

Requirements:
- CMake >= 2.6
- FLANN

Basic build steps:
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make

If FLANN isn't located on a standard location you must pass the prefix path (where the include and lib folders are)
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=<PATH_TO_FLANN>  -DCMAKE_BUILD_TYPE=Release ..
$ make

Usage:
$ ./indexdesc_benchmark dataset_path query_path (float|byte) dimensions

Contents:
cmake/ ----------> Extra CMake Modules
doc/ ------------> Source code for documentation
results/ --------> Results obtained in the experimentation in txt format
report.pdf ------> Report from the experimentation
CMakeLists.txt --> CMake Script for the project
readme.txt ------> This file

