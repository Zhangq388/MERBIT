#ifndef __UTILS__
#define __UTILS__
#include "./inc.cuh"
//merge search根据diag寻找coord_x和coord_y
template<unsigned int SIGMA> __host__ __device__
void merge_search(const int* offset, const int nrow, const int nnz, const int diag, int& coord_x, int& coord_y)
{
    //search range
    unsigned int y_min = max(diag - nnz, 0);  //y_min和y_max的设定可以缩小搜索空间
    unsigned int y_max = min(diag, nrow);
    
    //2D binary search
    while(y_min < y_max)
    {
        int mid = (y_min + y_max) >> 1;

        if(offset[mid + 1] <= diag - mid - 1)
        {
            y_min = mid + 1;
        }
        else
        {
            y_max = mid;
        }
    }
    
    coord_x = diag - y_min;
    coord_y = min(y_min, nrow);
}


template<typename ValueType> __forceinline__ __device__
ValueType warp_reduce(ValueType val)
{
    const unsigned int mask = 0xFFFFFFFF; 
    #pragma unroll
    for(unsigned int delta = 16; delta > 0; delta >>= 1)
    {
        val += __shfl_xor_sync(mask, val, delta);
    }
    return val;
}


template<typename ValueType> __forceinline__ __device__
void warp_scan(const int lid, const int len, ValueType& sum)
{
    ValueType val = 0;
    const unsigned int mask = (len == 32)? 0xFFFFFFFF : (0X1 << len) - 1; 
    #pragma unroll
    for(unsigned int i=0x1;i<len;i<<=1)
    {  
        val = __shfl_up_sync(mask, sum, i); //先取再加，稳当些
        sum += (lid >= i)? val : 0;
    }
}

template<typename ValueType> __forceinline__ __device__
void warp_segmented_sum(const int lid, const bool flag, const ValueType sum, const int idx_y, const int len, ValueType* d_y)
{
    ValueType local_sum = sum;
    ValueType tmp_sum = 0;
    bool local_flag = (lid < len)? flag : true;
    bool tmp_flag = local_flag;
    unsigned int i = 0x1;
    const unsigned int mask = (len == 32)? 0xFFFFFFFF : (0X1 << len) - 1; 
    while(__any_sync(mask, !local_flag))
    {
        tmp_sum = __shfl_up_sync(mask, local_sum, i);
        tmp_flag = __shfl_up_sync(mask, local_flag, i);
        local_sum += (!local_flag && lid >= i)? tmp_sum : (ValueType)(0);
        local_flag = local_flag? local_flag : tmp_flag;
        i <<= 1;
    }
    
    int loc = (lid == 0)? len - 1 : lid - 1;
    int idx = __shfl_sync(mask, idx_y, loc);
    tmp_sum = __shfl_sync(mask, local_sum, loc);
    if(flag)
    {
        atomicAdd(&d_y[idx], tmp_sum);
    }
}

template<typename ValueType> __global__ 
void warmup_kernel(ValueType *d_scan)
{
    const int lid = threadIdx.x & 31;

    int sum = 1;
    warp_scan<ValueType>(lid, 32, sum);

    if(lid == 31)
    {
        d_scan[lid] = sum;
    }
}

template<typename ValueType> 
void format_warmup()
{
    ValueType *d_scan;
    cudaMalloc((void **)&d_scan, 32 * sizeof(ValueType));

    int block = 128;
    int grid  = 4000;

    for (int i = 0; i < 50; i++)
    {
        warmup_kernel<<< grid, block >>>(d_scan);
    }

    cudaFree(d_scan);
}

template<typename ValueType>
void Fun_Check(const ValueType* X, const ValueType* Y, const int nrow)
{
    ValueType err(0.0);

    for(int i=0;i<nrow;++i)
    {
        err += (std::abs(X[i]-Y[i]) > std::abs(X[i]) * 0.0001)? std::abs(X[i]-Y[i]) : 0.0;
        if(std::abs(X[i]-Y[i]) > std::abs(X[i]) * 0.0001)
        {
            std::cout << "X[" << i << "] = " << X[i] << ", Y[" << i << "] = " << Y[i] << "\n";
        }
    }

    std::cout << ", total error = " << err << std::endl;
    //std::cout << "&" << err << std::endl;
}
#endif