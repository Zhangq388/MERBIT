#include "../include/class.cuh"
#include <ginkgo/ginkgo.hpp>
#include "../include/csr5/anonymouslib_cuda.h"
#include "../include/hola/hola.cuh"
#include "../include/fun.cuh"
template<typename ValueType>
void Fun_CSRMV(const ENV& env, PARAMETER<ValueType>& para)
{
    //变量定义
    const int nrow = para.graph->nrow;
    const int nnz = para.graph->nnz;
    Timer   timer;
    float   tim1(0), tim2(0);

    //生成offset
    para.graph->sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, para.graph->col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    
    //STEP 2：生成稀疏库对象
    int64_t Rows = nrow;
    int64_t Nnz  = nnz;
    ValueType  alpha = 1.0;
    ValueType  beta  = 1.0;
    size_t     bufferSize = 0;
    void*      d_buffer = nullptr;

    //生成矩阵
    timer.start(env.stream);
    cusparseSpMatDescr_t      SpMat;
    cusparseCreateCsr(&SpMat, Rows, Rows, Nnz, offset, para.graph->row_ind, para.graph->values, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, CudaDataType<ValueType>::value);

    //生成向量X
    cusparseDnVecDescr_t      Dn_VecX;
    cusparseCreateDnVec(&Dn_VecX, Rows, para.person, CudaDataType<ValueType>::value);
    
    //生成向量Y
    cusparseDnVecDescr_t      Dn_VecY;
    cusparseCreateDnVec(&Dn_VecY, Rows, para.pagerank_real, CudaDataType<ValueType>::value);
    
    //buffersize
    cusparseSpMV_bufferSize(env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG2, &bufferSize);
    cudaMalloc((void**)&d_buffer, bufferSize);
    cudaDeviceSynchronize();
    tim1 = timer.stop(env.stream);

    //计算
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd;++rnd)
    {
        cusparseSpMV(env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG2, d_buffer);
    }
    tim2 = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&CSR&&" << tim1 << "&" << tim2 / (para.maxiter * para.perrnd) << "\n";

    //回收内存
    CUDAFREE(offset);
    CUDAFREE(d_buffer);
    cusparseDestroySpMat(SpMat);
    cusparseDestroyDnVec(Dn_VecX);
    cusparseDestroyDnVec(Dn_VecY);
}
template void Fun_CSRMV<float>(const ENV& env, PARAMETER<float>& para);
template void Fun_CSRMV<double>(const ENV& env, PARAMETER<double>& para);


template<typename ValueType>
void Fun_COOMV(const ENV& env, PARAMETER<ValueType>& para)
{
    //变量定义
    const int nrow = para.graph->nrow;
    const int nnz = para.graph->nnz;
    Timer   timer;
    float   tim1(0), tim2(0);

    //STEP 2：生成稀疏库对象
    para.graph->sort(false);
    int64_t Rows = nrow;
    int64_t Nnz  = nnz;
    ValueType   alpha = 1.0;
    ValueType   beta  = 0.0;
    size_t      bufferSize = 0;
    void*       d_buffer = nullptr;

    //生成矩阵
    timer.start(env.stream);
    cusparseSpMatDescr_t      SpMat;
    cusparseCreateCoo(&SpMat, Rows, Rows, Nnz, para.graph->col_ind, para.graph->row_ind, para.graph->values, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, CudaDataType<ValueType>::value);

    //生成向量X
    cusparseDnVecDescr_t      Dn_VecX;
    cusparseCreateDnVec(&Dn_VecX, Rows, para.person, CudaDataType<ValueType>::value);

    //生成向量Y
    cusparseDnVecDescr_t      Dn_VecY;
    cusparseCreateDnVec(&Dn_VecY, Rows, para.pagerank_real, CudaDataType<ValueType>::value);
    
    //buffersize
    cusparseSpMV_bufferSize(env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_COO_ALG2, &bufferSize);
    cudaMalloc((void**)&d_buffer, bufferSize);
    cudaDeviceSynchronize();
    tim1 = timer.stop(env.stream);

    //计算
    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd;++rnd)
    {
        cusparseSpMV(env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_COO_ALG2, d_buffer);
    }
    tim2 = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&COO&&" << tim1 << "&" << tim2 / (para.maxiter * para.perrnd) << "\n";

    //回收内存
    CUDAFREE(d_buffer);
}
template void Fun_COOMV<float>(const ENV& env, PARAMETER<float>& para);
template void Fun_COOMV<double>(const ENV& env, PARAMETER<double>& para);

//merge
template<typename ValueType> 
void Fun_MERGEMV(const ENV& env, PARAMETER<ValueType>& para)
{
    //变量赋值
    Timer   timer;
    float   tim1(0), tim2(0);

    //生成offset
    para.graph->sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (para.graph->nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, para.graph->col_ind, para.graph->nnz, para.graph->nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //step 3：计算
    timer.start(env.stream);
    void*   d_temp_storage = nullptr;
    size_t  temp_storage_bytes = 0;
    cub::DeviceSpmv::CsrMV<ValueType>(d_temp_storage, temp_storage_bytes, para.graph->values, offset, para.graph->row_ind, para.person, para.pagerank_real, para.graph->nrow, para.graph->nrow, para.graph->nnz);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    tim1 = timer.stop(env.stream);

    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd;++rnd)
    {
       cub::DeviceSpmv::CsrMV<ValueType>(d_temp_storage, temp_storage_bytes, para.graph->values, offset, para.graph->row_ind, para.person, para.pagerank_real, para.graph->nrow, para.graph->nrow, para.graph->nnz, env.stream);
    }
    tim2 = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&MERGE&&" << tim1 << "&" << tim2 / (para.maxiter * para.perrnd) << "\n";

    CUDAFREE(offset);
    CUDAFREE(d_temp_storage);
}
template void Fun_MERGEMV<float>(const ENV& env, PARAMETER<float>& para);
template void Fun_MERGEMV<double>(const ENV& env, PARAMETER<double>& para);

//ginkgo csr
template<typename ValueType>
void Fun_GKOCSR(const ENV& env, PARAMETER<ValueType>& para)
{
    //STEP 1：变量赋值
    Timer timer;
    float tim(0.0); 
    
    //生成offset
    para.graph->sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (para.graph->nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, para.graph->col_ind, para.graph->nnz, para.graph->nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //step 2：生成GINKGO所需数据结构
    auto ref = gko::ReferenceExecutor::create();
    auto exec = gko::CudaExecutor::create(env.device_id, ref); 
    auto gko_offset = gko::array<int>::view(exec, para.graph->nrow + 1, offset);
    auto gko_row_ind = gko::array<int>::view(exec, para.graph->nnz, para.graph->row_ind);
    auto gko_values = gko::array<ValueType>::view(exec, para.graph->nnz, para.graph->values);
    auto matrix = gko::matrix::Csr<ValueType, gko::int32>::create(exec, gko::dim<2>(para.graph->nrow, para.graph->nrow), gko_values, gko_row_ind, gko_offset, nullptr);
    auto gko_x = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    auto gko_y = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    
    //STEP 3：开始运行
    cudaMemcpy(gko_x->get_values(), para.person, sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);
    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd; ++rnd)
    {
        matrix->apply(gko_x, gko_y);
    }
    exec->synchronize();
    cudaMemcpy(para.pagerank_real, gko_y->get_values(), sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);
    tim = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&GKOCSR&&&" << tim / (para.maxiter * para.perrnd) << "\n";
        
    CUDAFREE(offset);
}
template void Fun_GKOCSR<float>(const ENV& env, PARAMETER<float>& para);
template void Fun_GKOCSR<double>(const ENV& env, PARAMETER<double>& para);

//ginkgo coo
template<typename ValueType>
void Fun_GKOCOO(const ENV& env, PARAMETER<ValueType>& para)
{
    //STEP 1：变量赋值
    Timer timer;
    float tim(0.0); 
    
    //step 2：生成GINKGO所需数据结构
    auto ref = gko::ReferenceExecutor::create();
    auto exec = gko::CudaExecutor::create(env.device_id, ref); 
    auto gko_row_ind = gko::array<int>::view(exec, para.graph->nnz, para.graph->col_ind);
    auto gko_col_ind = gko::array<int>::view(exec, para.graph->nnz, para.graph->row_ind);
    auto gko_values = gko::array<ValueType>::view(exec, para.graph->nnz, para.graph->values);
    auto matrix = gko::matrix::Coo<ValueType, gko::int32>::create(exec, gko::dim<2>(para.graph->nrow, para.graph->nrow), gko_values, gko_col_ind, gko_row_ind);
    auto gko_x = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    auto gko_y = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    cudaMemcpy(gko_x->get_values(), para.person, sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);

    //STEP 3：开始运行
    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd; ++rnd)
    {
        matrix->apply(gko_x, gko_y);
    }
    exec->synchronize();
    cudaMemcpy(para.pagerank_real, gko_y->get_values(), sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);
    tim = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&GKOCOO&&&" << tim / (para.maxiter * para.perrnd) << "\n";
}
template void Fun_GKOCOO<float>(const ENV& env, PARAMETER<float>& para);
template void Fun_GKOCOO<double>(const ENV& env, PARAMETER<double>& para);

//ginkgo sellp
template<typename ValueType>
void Fun_GKOSELLP(const ENV& env, PARAMETER<ValueType>& para)
{
    //STEP 1：变量赋值
    Timer timer;
    float tim1(0), tim2(0); 
    
    //生成offset
    para.graph->sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (para.graph->nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, para.graph->col_ind, para.graph->nnz, para.graph->nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //step 2：生成GINKGO所需数据结构
    auto ref = gko::ReferenceExecutor::create();
    auto exec = gko::CudaExecutor::create(env.device_id, ref); 
    auto gko_offset = gko::array<int>::view(exec, para.graph->nrow + 1, offset);
    auto gko_row_ind = gko::array<int>::view(exec, para.graph->nnz, para.graph->row_ind);
    auto gko_values = gko::array<ValueType>::view(exec, para.graph->nnz, para.graph->values);
    auto matrix_csr = gko::matrix::Csr<ValueType, gko::int32>::create(exec, gko::dim<2>(para.graph->nrow, para.graph->nrow), gko_values, gko_row_ind, gko_offset, nullptr);
    timer.start(env.stream);
    auto matrix =  gko::matrix::Sellp<ValueType, gko::int32>::create(exec, gko::dim<2>(para.graph->nrow, para.graph->nrow));
    matrix_csr->convert_to(matrix);
    exec->synchronize();
    tim1 = timer.stop(env.stream);
    auto gko_x = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    auto gko_y = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    cudaMemcpy(gko_x->get_values(), para.person, sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);

    //STEP 3：开始运行
    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd; ++rnd)
    {
        matrix->apply(gko_x, gko_y);
    }
    exec->synchronize();
    cudaMemcpy(para.pagerank_real, gko_y->get_values(), sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);
    tim2 = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&GKOSELLP&&" << tim1 << "&" << tim2 / (para.maxiter * para.perrnd) << "\n";
    
    CUDAFREE(offset);
}
template void Fun_GKOSELLP<float>(const ENV& env, PARAMETER<float>& para);
template void Fun_GKOSELLP<double>(const ENV& env, PARAMETER<double>& para);

//ginkgo hyb
template<typename ValueType>
void Fun_GKOHYB(const ENV& env, PARAMETER<ValueType>& para)
{
    //STEP 1：变量赋值
    Timer timer;
    float tim1(0), tim2(0); 
    
    //生成offset
    para.graph->sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (para.graph->nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, para.graph->col_ind, para.graph->nnz, para.graph->nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //step 2：生成GINKGO所需数据结构
    auto ref = gko::ReferenceExecutor::create();
    auto exec = gko::CudaExecutor::create(env.device_id, ref); 
    auto gko_offset = gko::array<int>::view(exec, para.graph->nrow + 1, offset);
    auto gko_row_ind = gko::array<int>::view(exec, para.graph->nnz, para.graph->row_ind);
    auto gko_values = gko::array<ValueType>::view(exec, para.graph->nnz, para.graph->values);
    auto matrix_csr = gko::matrix::Csr<ValueType, gko::int32>::create(exec, gko::dim<2>(para.graph->nrow, para.graph->nrow), gko_values, gko_row_ind, gko_offset, nullptr);
    timer.start(env.stream);
    auto matrix =  gko::matrix::Hybrid<ValueType, gko::int32>::create(exec, gko::dim<2>(para.graph->nrow, para.graph->nrow));
    matrix_csr->convert_to(matrix);
    exec->synchronize();
    tim1 = timer.stop(env.stream);
    auto gko_x = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    auto gko_y = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(para.graph->nrow, 1), 1);
    cudaMemcpy(gko_x->get_values(), para.person, sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);

    //STEP 3：开始运行
    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd; ++rnd)
    {
        matrix->apply(gko_x, gko_y);
    }
    exec->synchronize();
    cudaMemcpy(para.pagerank_real, gko_y->get_values(), sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);
    tim2 = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&GKOHYB&&" << tim1 << "&" << tim2 / (para.maxiter * para.perrnd) << "\n";
    
    CUDAFREE(offset);
}
template void Fun_GKOHYB<float>(const ENV& env, PARAMETER<float>& para);
template void Fun_GKOHYB<double>(const ENV& env, PARAMETER<double>& para);


//csr5
template<typename ValueType>
void Fun_CSR5(const ENV& env, PARAMETER<ValueType>& para)
{
    //STEP 1：变量赋值
    Timer timer;
    float tim1(0), tim2(0); 
    ValueType alpha = 1.0;
    
    //生成offset
    para.graph->sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (para.graph->nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, para.graph->col_ind, para.graph->nnz, para.graph->nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //STEP 2：生成CSR5对象
    //生成矩阵
    anonymouslibHandle<int, unsigned int, ValueType>  A(para.graph->nrow, para.graph->nrow); 
    A.inputCSR(para.graph->nnz, offset, para.graph->row_ind, para.graph->values);
    A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
    timer.start(env.stream);
    A.asCSR5();
    A.setX(para.person);
    tim1 = timer.stop(env.stream);
    
    //STEP 3：开始运行
    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd; ++rnd)
    {
        cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
        A.spmv(alpha, para.pagerank_real);
    }
    cudaDeviceSynchronize();
    tim2 = timer.stop(env.stream);

    FSTR << " " << para.graph->graph_id << "&CSR5&&" << tim1 << "&" << tim2 / (para.maxiter * para.perrnd) << "\n";
    
    CUDAFREE(offset);
}
template void Fun_CSR5<double>(const ENV& env, PARAMETER<double>& para);
template void Fun_CSR5<float>(const ENV& env, PARAMETER<float>& para);

//hola
template<typename ValueType>
void Fun_HOLA(const ENV& env, PARAMETER<ValueType>& para)
{
    //STEP 1：变量赋值
    Timer timer;
    float tim1(0), tim2(0); 
    
    //生成offset
    para.graph->sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (para.graph->nrow + 1));
    cusparseXcoo2csr(env.handle_sparse, para.graph->col_ind, para.graph->nnz, para.graph->nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //step 2：生成hola所需数据结构
    dCSR<ValueType>         dcsr_mat(para.graph->nrow, para.graph->nrow, para.graph->nnz, para.graph->values, offset, para.graph->row_ind);
    dDenseVector<ValueType> dinput(para.graph->nrow);
    dDenseVector<ValueType> dres(para.graph->nrow);
    cudaMemcpy(dinput.data, para.person, sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);
    
    timer.start(env.stream);
    size_t holatemp_req;
    void*  dholatemp;
	hola_spmv(nullptr, holatemp_req, dres, dcsr_mat, dinput, HolaMode::Default, false, false);
	cudaMalloc(&dholatemp, holatemp_req);
    cudaDeviceSynchronize();
    tim1 = timer.stop(env.stream);
    
    //STEP 3：开始运行
    cudaMemset(para.pagerank_real, 0, sizeof(ValueType) * para.graph->nrow);
    timer.start(env.stream);
    for(int rnd = 0; rnd<para.maxiter * para.perrnd; ++rnd)
    {
        hola_spmv(dholatemp, holatemp_req, dres, dcsr_mat, dinput, HolaMode::Default, false, false);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(para.pagerank_real, dres.data, sizeof(ValueType) * para.graph->nrow, cudaMemcpyDeviceToDevice);
    tim2 = timer.stop(env.stream);
    
    FSTR << " " << para.graph->graph_id << "&HOLA&&" << tim1 << "&" << tim2 / (para.maxiter * para.perrnd) << "\n";
    
    CUDAFREE(offset);
    CUDAFREE(dholatemp);
}
template void Fun_HOLA<double>(const ENV& env, PARAMETER<double>& para);
template void Fun_HOLA<float>(const ENV& env, PARAMETER<float>& para);