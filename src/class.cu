#include "../include/class.cuh"
 __global__ void Fun_Values_Kernel1(const int* row_ind, const int nnz, int* cnt)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < nnz)
    {
        atomicAdd(&cnt[row_ind[n]], 1);
    }
}

template<typename ValueType> __global__ 
void Fun_Values_Kernel2(const int* row_ind, const int* cnt, const int nnz, ValueType* values)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < nnz)
    {
        values[n] = 1.0 / cnt[row_ind[n]];
    }
}


template<typename ValueType>
void Fun_Values(const int* row_ind, const int nrow, const int nnz, ValueType* values)
{
    int* cnt;
    cudaMalloc((void**)&cnt, sizeof(int) * nrow);
    cudaMemset(cnt, 0, sizeof(int) * nrow);

    const int block = 512;
    const int grid = (nnz + block - 1) / block;
    Fun_Values_Kernel1<<<grid, block>>>(row_ind, nnz, cnt);
    Fun_Values_Kernel2<ValueType><<<grid, block>>>(row_ind, cnt, nnz, values);
    CUDAFREE(cnt);
}

template<typename ValueType>
ValueType Fun_Check(const ValueType* X, const ValueType* Y, const int nrow)
{
    ValueType err(0.0);

    for(int i=0;i<nrow;++i)
    {
        err += std::abs(X[i]-Y[i]);
    }

    return err;
}

//图
template<typename ValueType> GRAPH<ValueType>::GRAPH()
{
    graph_id = -1;
    nrow = 0;
    nnz = 0;
    row_ind = nullptr; 
    col_ind = nullptr; 
    values = nullptr;
}

template<typename ValueType> GRAPH<ValueType>::GRAPH(const std::string& config, const int gid)
{
    //STEP 1：变量赋值
    std::fstream  file;
    std::string   line = "";
    int           id  = 0;
    int           i   = 0;
    bool          pattern = true;
    bool          zero_base = true;
    std::string   filename = "";
    
    file.open(config, std::fstream::ios_base::in);
    do
    {
        std::getline(file, line);
        std::stringstream(line) >> id >> filename;
    } while (id != gid);
    file.close();
    
    
    //STEP 2：读取数据文件位置
    int Row(0), Col(0), Nnz(0);
    ValueType Val(0.0);

    file.open(filename, std::fstream::ios_base::in);
    do 
    {
        std::getline(file, line);
        if(i == 0)
        {
            pattern = (line.find("symmetric") != std::string::npos)? true : false;
            zero_base = (line.find("basezero") != std::string::npos)? true : false;
        }
        ++i;
    } while (line[0] == '%');
    std::stringstream(line) >> Row >> Col >> Nnz;
 
    graph_id = gid;
    nrow = Row;
    nnz  = pattern? 2 * Nnz : Nnz;
    
    int* h_row_ind = (int*)calloc(nnz, sizeof(int));
    int* h_col_ind = (int*)calloc(nnz, sizeof(int));
    ValueType* h_values = (ValueType*)calloc(nnz, sizeof(ValueType));

    i=0;
    while (std::getline(file, line)) 
    {
        std::stringstream v_str(line);
        v_str >> Row >> Col >> Val;
        
        h_row_ind[i] = (zero_base)? Row : Row - 1;
        h_col_ind[i] = (zero_base)? Col : Col - 1;
        h_values[i]  = Val;

        if(pattern)
        {
            h_row_ind[Nnz + i] = (zero_base)? Col : Col - 1;
            h_col_ind[Nnz + i] = (zero_base)? Row : Row - 1;
            h_values[Nnz + i]  = Val;
        }

        ++i;
    }
    file.close();
    
    //读入显存
    cudaMalloc((void**)&row_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&col_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&values,  sizeof(ValueType) * nnz);
    cudaMemcpy(row_ind, h_row_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, h_col_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(values,  h_values,  sizeof(ValueType) * nnz, cudaMemcpyHostToDevice);
    
    //生成Values
    Fun_Values<ValueType>(row_ind, nrow, nnz, values);

    //STEP 3：清理内存
    FREE(h_row_ind);
    FREE(h_col_ind);
    FREE(h_values);
}

template<typename ValueType>
void GRAPH<ValueType>::sort(const bool dir)
{
    using tupleT = thrust::tuple<int, int, ValueType>;
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(row_ind), thrust::device_pointer_cast(col_ind), thrust::device_pointer_cast(values)));
    auto zip_end = zip_begin + nnz;
    
    if(dir)
    {
        thrust::sort(zip_begin, zip_end, 
                    []__device__(const tupleT& _lhs, const tupleT& _rhs) 
                    {
                        if (thrust::get<0>(_lhs) == thrust::get<0>(_rhs)) 
                        {
                            return thrust::get<1>(_lhs) < thrust::get<1>(_rhs); 
                        } 
                        return thrust::get<0>(_lhs) < thrust::get<0>(_rhs);
                    }
                    );
    }
    else
    {
        thrust::sort(zip_begin, zip_end, 
                    []__device__(const tupleT& _lhs, const tupleT& _rhs) 
                    {
                        if (thrust::get<1>(_lhs) == thrust::get<1>(_rhs)) 
                        {
                            return thrust::get<0>(_lhs) < thrust::get<0>(_rhs); 
                        } 
                        return thrust::get<1>(_lhs) < thrust::get<1>(_rhs);
                    }
                    );
    }
    cudaDeviceSynchronize();
}


//dangling
__global__ void dangling_kernel(const int* offset, const int nrow, int* num_dang)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid<nrow)
    {
        int num = (offset[gid + 1] == offset[gid])? 1 : 0;
        atomicAdd(num_dang, num);
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::dangling(const ENV& env)
{
    int h_num_dang;
    int* num_dang;
    cudaMalloc((void**)&num_dang, sizeof(int));
    cudaMemset(num_dang, 0, sizeof(int));

    //step 1：排序
    sort(false);

    //step 2：生成offset
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //step 3：统计
    int block = 256;
    int grid = (nrow + block - 1) / block;
    dangling_kernel<<<grid, block>>>(offset, nrow, num_dang);
    
    cudaMemcpy(&h_num_dang, num_dang, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "graph = " << graph_id << ", dang = " << h_num_dang << std::endl;

    //step 4：释放空间
    CUDAFREE(offset);
    CUDAFREE(num_dang);
}


//desc
__global__ void desc_kernel1(const int* indices, const int nnz, int* deg)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    if(n < nnz)
    {
        atomicAdd(&(deg[indices[n]]), 1);
    }
}

__global__ void desc_kernel2(const int* d_hist, const int maxdeg, int* reducennz)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    if(n < maxdeg)
    {
        reducennz[n] = n * d_hist[n];
    }
}

__global__ void desc_kernel3(const int* d_reducerow, const int nrow, const int* d_reducennz, const int nnz, const int maxdeg, float* d_ratio1, float* d_ratio2)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    //反过来
    if(n < maxdeg)
    {
        d_ratio1[n] = d_reducerow[n] / (double)nrow;
        d_ratio2[n] = d_reducennz[n] / (double)nnz;
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::desc(const ENV& env)
{
    //step 1：排序
    sort(false);

    //step 2：生成offset
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    thrust::device_ptr<int>  ptr_offset(offset);
    float empty = thrust::count_if(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(nrow), [ptr_offset] __host__ __device__ (int i) { return ptr_offset[i + 1] == ptr_offset[i]; }) / (float)(nrow);

    //step 1：统计顶点入度
    int*     d_deg;
    cudaMalloc((void**)&d_deg, sizeof(int) * nrow);   
    cudaMemset(d_deg, 0, sizeof(int) * nrow);
    thrust::device_ptr<int>  d_ptr_deg(d_deg);
    const int block = 256;
    int grid = (nnz + block - 1) / block;
    desc_kernel1<<<grid, block>>>(row_ind, nnz, d_deg);

    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float) * nrow);
    thrust::device_ptr<float> ptr_out(d_out);

    auto mean = nnz / (float)(nrow);

    thrust::transform(d_ptr_deg, d_ptr_deg + nrow, ptr_out, [mean]__device__(const int& _x){return (_x - mean) * (_x - mean);});
    auto var = thrust::reduce(ptr_out, ptr_out + nrow) / nrow;
    
    thrust::transform(d_ptr_deg, d_ptr_deg + nrow, ptr_out, [mean]__device__(const int& _x){return (_x - mean) * (_x - mean) * (_x - mean);});
    auto skew = thrust::reduce(ptr_out, ptr_out + nrow) / nrow / std::pow(var, 1.5);
    
    thrust::transform(d_ptr_deg, d_ptr_deg + nrow, ptr_out, [mean]__device__(const int& _x){return (_x - mean) * (_x - mean) * (_x - mean) * (_x - mean);});
    auto kurt = thrust::reduce(ptr_out, ptr_out + nrow) / nrow / var / var;

    FSTR << graph_id << "&" 
         << nrow << "&" 
         << nnz << "&" 
         << std::fixed 
         << std::setprecision(2) << mean << "&" 
         << std::setprecision(2) << var  << "&" 
         << std::setprecision(2) << skew << "&" 
         << std::setprecision(2) << kurt << "&"
         << std::setprecision(2) << empty
         << std::endl;

    //std::cout << "graph = " << graph_id  << std::endl;
    /*
    int maxoutdeg = (*thrust::max_element(d_ptr_deg, d_ptr_deg + nrow)) + 1;

    cudaMemset(d_deg, 0, sizeof(int) * nrow);
    desc_kernel1<<<grid, block>>>(col_ind, nnz, d_deg);
    int maxdeg = (*thrust::max_element(d_ptr_deg, d_ptr_deg + nrow)) + 1;
    int cnt1 = thrust::count_if(d_ptr_deg, d_ptr_deg + nrow, thrust::placeholders::_1 > 224);
    int cnt2 = thrust::count_if(d_ptr_deg, d_ptr_deg + nrow, thrust::placeholders::_1 > 448);
    std::cout << "graph_id = " << graph_id << ", nrow = " << nrow << ", nnz = " << nnz << ", deg = " << (float)(nnz) / (float)(nrow) << ", max_outdeg = " << maxoutdeg << ", dangling = " << thrust::count(d_ptr_deg, d_ptr_deg + nrow, 0) / (float)(nrow) << ", r1 = " << (float)(cnt1) / (float)(nrow) << ", r2 = " << (float)(cnt2) / (float)(nrow) << std::endl;
    
    //step 2：生成入度直方图
    int*     d_hist;
    cudaMalloc((void**)&d_hist, sizeof(int) * maxdeg);
    void*    d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_deg, d_hist, maxdeg + 1, 0, maxdeg, nrow);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_deg, d_hist, maxdeg + 1, 0, maxdeg, nrow);
    CUDAFREE(d_temp_storage);
    thrust::device_ptr<int>    d_ptr_hist(d_hist);

    //step 3：统计行数
    int*     d_reducerow;
    cudaMalloc((void**)&d_reducerow, sizeof(int) * maxdeg);
    cudaMemset(d_reducerow, 0, sizeof(int) * maxdeg);
    thrust::device_ptr<int>    d_ptr_reducerow(d_reducerow);
    thrust::inclusive_scan(d_ptr_hist, d_ptr_hist + maxdeg, d_ptr_reducerow);

    //step 4：统计非零元
    int*     d_reducennz;
    cudaMalloc((void**)&d_reducennz, sizeof(int) * maxdeg);
    cudaMemset(d_reducennz, 0, sizeof(int) * maxdeg);
    grid = (maxdeg + block - 1) / block;
    desc_kernel2<<<grid, block>>>(d_hist, maxdeg, d_reducennz);
    thrust::device_ptr<int>    d_ptr_reducennz(d_reducennz);
    thrust::inclusive_scan(d_ptr_reducennz, d_ptr_reducennz + maxdeg, d_ptr_reducennz);

    //step 5：计算比例
    int*   h_hist   = (int*)calloc(maxdeg, sizeof(int));
    float* h_ratio1 = (float*)calloc(maxdeg, sizeof(float));
    float* h_ratio2 = (float*)calloc(maxdeg, sizeof(float));
    float* d_ratio1;
    float* d_ratio2;
    cudaMalloc((void**)&d_ratio1, sizeof(int) * maxdeg);
    cudaMalloc((void**)&d_ratio2, sizeof(int) * maxdeg);
    desc_kernel3<<<grid, block>>>(d_reducerow, nrow, d_reducennz, nnz, maxdeg, d_ratio1, d_ratio2);
    cudaMemcpy(h_hist, d_hist, sizeof(int) * maxdeg, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ratio1, d_ratio1, sizeof(float) * maxdeg, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ratio2, d_ratio2, sizeof(float) * maxdeg, cudaMemcpyDeviceToHost);
    
    //step 6：写文件
    std::fstream    fstr;
    fstr.open(file, std::ios_base::app);
    int num = 0;
    for(int i=0; i<std::min(maxdeg, 512); ++i)
    {
        if(h_hist[i]>0)
        {
            fstr << graph_id << "&" << num << "&" << h_ratio1[i] << "&" << h_ratio2[i] << "\n";
            ++num;
        }
    }
    fstr.close();
    */

    //step 7：回收空间
    CUDAFREE(offset);
    CUDAFREE(d_deg);
    CUDAFREE(d_out);
    //CUDAFREE(d_hist);
    //CUDAFREE(d_reducerow);
    //CUDAFREE(d_reducennz);
    //CUDAFREE(d_ratio1);
    //CUDAFREE(d_ratio2);
    //FREE(h_hist);
    //FREE(h_ratio1);
    //FREE(h_ratio2);
}

//show
template<typename ValueType> 
void GRAPH<ValueType>::show()
{
    //打印统计信息
    std::cout << "GRAPH = " << graph_id << ", nrow = " << nrow << ", nnz = " << nnz << "\n";
    
    //定义图
    thrust::host_vector<ValueType> vec_Graph(nrow * nrow, 0.0);
    thrust::host_vector<int>       vec_row_ind(nnz, 0);
    thrust::host_vector<int>       vec_col_ind(nnz, 0);
    thrust::host_vector<ValueType> vec_values(nnz, 0.0);
    thrust::copy(thrust::device_pointer_cast(row_ind), thrust::device_pointer_cast(row_ind) + nnz, vec_row_ind.begin());
    thrust::copy(thrust::device_pointer_cast(col_ind), thrust::device_pointer_cast(col_ind) + nnz, vec_col_ind.begin());
    thrust::copy(thrust::device_pointer_cast(values),  thrust::device_pointer_cast(values) + nnz,  vec_values.begin());

    for(int i=0;i<nnz;++i)
    {
        vec_Graph[vec_col_ind[i] * nrow + vec_row_ind[i]] = vec_values[i];
    }
    
    //打印数据
    for(int i=0;i<nrow;++i)
    {
        for(int j=0;j<nrow;++j)
        {
            std::cout << " " << std::left << std::setw(9) << vec_Graph[i * nrow + j];
        }
        std::cout << "\n";
    }

    std::cout << std::endl;
}

template<typename ValueType> GRAPH<ValueType>::~GRAPH()
{
    if(owner)
    {
        CUDAFREE(row_ind); 
        CUDAFREE(col_ind); 
        CUDAFREE(values);
    }
}

template struct GRAPH<float>;
template struct GRAPH<double>;
template struct PARAMETER<float>;
template struct PARAMETER<double>;