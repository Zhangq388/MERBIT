#ifndef UTILS_CUDA_H
#define UTILS_CUDA_H

#include "./common_cuda.h"
template<typename ValueType> __forceinline__ __device__
ValueType sum_32_shfl(ValueType sum)
{
    
    #pragma unroll
    for(int mask = ANONYMOUSLIB_CSR5_OMEGA / 2 ; mask > 0 ; mask >>= 1)
    {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask);
    }

    return sum;
}


//exclusive scan using a single thread
template<typename T> __inline__ __device__
void scan_single(volatile T *s_scan, const int local_id, const int l)
{
    T old_val, new_val;
    if (!local_id)
    {
        old_val = s_scan[0];
        s_scan[0] = 0;
        for (int i = 1; i < l; i++)
        {
            new_val = s_scan[i];
            s_scan[i] = old_val + s_scan[i-1];
            old_val = new_val;
        }
    }
}


// exclusive scan
template<typename T> __inline__ __device__
void scan_32(volatile T *s_scan, const int local_id)
{
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    T temp;

    if (local_id < 16)  { ai = baseai - 1;      bi = basebi - 1;       s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0)  { s_scan[31] = s_scan[15]; s_scan[15] = 0; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}


// inclusive scan
template<typename T> __forceinline__ __device__
T scan_32_shfl(T x, const int local_id)
{
    T y = __shfl_up_sync(0xFFFFFFFF, x, 1);
    x = local_id >= 1 ? x + y : x;
    y = __shfl_up_sync(0xFFFFFFFF, x, 2);
    x = local_id >= 2 ? x + y : x;
    y = __shfl_up_sync(0xFFFFFFFF, x, 4);
    x = local_id >= 4 ? x + y : x;
    y = __shfl_up_sync(0xFFFFFFFF, x, 8);
    x = local_id >= 8 ? x + y : x;
    y = __shfl_up_sync(0xFFFFFFFF, x, 16);
    x = local_id >= 16 ? x + y : x;

    return x;
}

enum
{
    /// The number of warp scan steps
    STEPS = 5,

    // The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    SHFL_C = ((1 << STEPS) & 31) << 8
};

// inclusive scan for double data type
__forceinline__ __device__
double scan_32_shfl(double x)
{
    const unsigned FULL_MASK = 0xffffffff;  // 全掩码，指示所有线程参与
    #pragma unroll
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
        // 使用内联汇编将 shfl 指令更新为带 sync 的版本
        asm(
            "{"
            "  .reg .s32 lo;"
            "  .reg .s32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            // 使用 shfl.sync.up.b32 替换 shfl.up.b32
            "  shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"
            "  shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.f64 %0, %0, %1;"
            "}"
            : "=d"(x) : "d"(x), "r"(1 << STEP), "r"(SHFL_C), "r"(FULL_MASK));
    }

    return x;
}


// exclusive scan
template<typename T> __inline__ __device__
void scan_32_plus1(volatile  T *s_scan, const int local_id)
{
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    T temp;

    if (local_id < 16)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0)  { s_scan[32] = s_scan[31] + s_scan[15]; s_scan[31] = s_scan[15]; s_scan[15] = 0; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}



template<typename T> __inline__ __device__
T scan_plus1_shfl(volatile T *s_scan, const int local_id, T r_in, const int seg_num)
{
    // 3-stage method. scan-scan-propogate

    // shfl version
    const int lane_id = local_id % ANONYMOUSLIB_THREAD_BUNCH;
    const int seg_id = local_id / ANONYMOUSLIB_THREAD_BUNCH;

    // stage 1. thread bunch scan
    T r_scan = 0;

    //if (seg_id < seg_num)
    //{
        r_scan = scan_32_shfl<T>(r_in, lane_id);

        if (lane_id == ANONYMOUSLIB_THREAD_BUNCH - 1)
            s_scan[seg_id] = r_scan;

        r_scan = __shfl_up_sync(0xFFFFFFFF, r_scan, 1);
        r_scan = lane_id ? r_scan : 0;
    //}

    __syncthreads();

    // stage 2. one thread bunch scan
    r_in = (local_id < seg_num) ? s_scan[local_id] : 0;
    if (!seg_id)
        r_in = scan_32_shfl<T>(r_in, lane_id);

    if (local_id < seg_num)
        s_scan[local_id + 1] = r_in;

    // single thread in-place scan
    //scan_single<T>(s_scan, local_id, seg_num+1);

    __syncthreads();

    // stage 3. propogate (element-wise add) to all
    if (seg_id) // && seg_id < seg_num)
        r_scan += s_scan[seg_id];

    return r_scan;
}

template<typename T> __inline__ __device__
void scan_256_plus1(volatile T *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    T temp;

    if (threadIdx.x < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

template<typename IndexType> __forceinline__ __device__
void fetch_x(cudaTextureObject_t  d_x_tex, const IndexType i,float* x)
{
    *x = tex1Dfetch<float>(d_x_tex, i);
}

template<typename IndexType> __forceinline__ __device__
void fetch_x(cudaTextureObject_t  d_x_tex, const IndexType i, double* x)
{
    int2 x_int2 = tex1Dfetch<int2>(d_x_tex, i);
    *x = __hiloint2double(x_int2.y, x_int2.x);
}
#endif // UTILS_CUDA_H


