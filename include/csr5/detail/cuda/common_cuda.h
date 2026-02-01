#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/cub/cub.cuh"
#include "/usr/local/cuda/include/cub/block/block_adjacent_difference.cuh"

#include "../common.h"
#include "../utils.h"

#define ANONYMOUSLIB_CSR5_OMEGA   32
#define ANONYMOUSLIB_THREAD_BUNCH 32
#define ANONYMOUSLIB_THREAD_GROUP 256

#define ANONYMOUSLIB_AUTO_TUNED_SIGMA -1

#endif // COMMON_CUDA_H
