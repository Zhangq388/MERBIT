# CUDA SpMV (CMake)

## Requirements
- CUDA Toolkit (e.g. 12.x)
- CMake >= 3.18
- cuSPARSE / cuBLAS
- Ginkgo

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

## Dataset configuration and running

Datasets are configured in a config file.
You run the program by passing a dataset ID:
./main 1

Config files are located in config/ (example: config/config.txt).
Each entry maps an ID to a dataset file path (MTX).

Example (pseudo format):
1 /data3/graph/mtxxia/xxx.mtx
2 /data3/graph/mtxxia/yyy.mtx
...

## Run
So ./main 1 will load the dataset path corresponding to ID 1.
