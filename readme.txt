Build instructions
------------------

Requirements:
- CMake >= 2.6
- FLANN

Basic build steps:
$ mkdir build
$ cd build
$ cmake ..
$ make

If FLANN isn't located on a standard location you must pass the prefix path (where the include and lib folders are)
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=<PATH_TO_FLANN> ..
$ make