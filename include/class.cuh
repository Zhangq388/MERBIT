#ifndef __CLASS__
#define __CLASS__
#include "./utils.cuh"
//时间
struct Timer 
{
    cudaEvent_t start_event, stop_event;
    Timer()
    {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    void start(cudaStream_t stream) 
    {
        cudaDeviceSynchronize();
        cudaEventRecord(start_event, stream);
    }
    float stop(cudaStream_t stream)
    {
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
    ~Timer()
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
};



//环境
struct ENV
{   
    const int            device_id = 0;
    const std::string    config; //配置文件
    const std::string    log;    //结果文件
    cudaStream_t         stream;
    cusparseHandle_t     handle_sparse;
    cublasHandle_t       handle_cublas;
    ENV(const int id, const std::string& config, const std::string log): device_id(id), config(config), log(log)
    {
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream);
        cusparseCreate(&handle_sparse);
        cublasCreate_v2(&handle_cublas);
        cusparseSetStream(handle_sparse, stream);
        cublasSetStream_v2(handle_cublas, stream);
    };
    ~ENV()
    {
        cudaStreamDestroy(stream);
        cusparseDestroy(handle_sparse);
        cublasDestroy_v2(handle_cublas);
    };
};

//图
template<typename ValueType> 
struct GRAPH
{
    int             graph_id;
    int             nrow;
    int             nnz;
    bool            owner = true;
    int*            row_ind;
    int*            col_ind;
    ValueType*      values;
    GRAPH(); //默认构造函数
    GRAPH(const bool own) : owner(own) {};
    GRAPH(const std::string& config, const int gid); //构造函数1
    GRAPH(const GRAPH& graph); //拷贝构造函数
    void sort(const bool dir);
    void dangling(const ENV& env);
    void desc(const ENV& env);
    void show(); //打印矩阵
    ~GRAPH(); //析构函数
};


//变量
template<typename ValueType>
struct VARIABLE
{
    GRAPH<ValueType>*     graph;
    ValueType*            prval;
    ValueType*            val1;
    ValueType*            val2;
    ValueType*            res1;
    ValueType*            res2;
    VARIABLE(GRAPH<ValueType>* graph): graph(graph)
    {
        cudaMalloc((void**)&prval, sizeof(ValueType) * graph->nrow);
        cudaMalloc((void**)&val1,  sizeof(ValueType) * graph->nrow);
        cudaMalloc((void**)&val2,  sizeof(ValueType) * graph->nrow);
        cudaMalloc((void**)&res1,  sizeof(ValueType) * graph->nrow);
        cudaMalloc((void**)&res2,  sizeof(ValueType) * graph->nrow);
        cudaMemset(prval, 0, sizeof(ValueType) * graph->nrow);
        cudaMemset(val1, 0, sizeof(ValueType) * graph->nrow);
        cudaMemset(val2, 0, sizeof(ValueType) * graph->nrow);
        cudaMemset(res1, 0, sizeof(ValueType) * graph->nrow);
        cudaMemset(res2, 0, sizeof(ValueType) * graph->nrow);
    };
    ~VARIABLE() //自己孩子自己抱
    {
        CUDAFREE(prval);
        CUDAFREE(val1);
        CUDAFREE(val2);
        CUDAFREE(res1);
        CUDAFREE(res2);
    };
};


//参数
template<typename ValueType>
struct PARAMETER
{
    GRAPH<ValueType>* graph;
    double            damp;
    int               perrnd;
    int               maxiter;
    int               iter;
    int               BOUND;  //临时用的
    char              TYPE;
    ValueType*        person;
    ValueType*        pagerank_real;
    PARAMETER(GRAPH<ValueType>* graph, double damp, int perrnd, int maxiter) : 
    graph(graph), damp(damp), perrnd(perrnd), maxiter(maxiter)
    {
        iter = 0;
        cudaMalloc((void**)&person, sizeof(ValueType) * graph->nrow);
        cudaMalloc((void**)&pagerank_real, sizeof(ValueType) * graph->nrow);
        thrust::device_ptr<ValueType> ptr_person(person);
        //thrust::fill(ptr_person, ptr_person+graph->nrow, (ValueType)(1.0 / graph->nrow));
        thrust::fill(ptr_person, ptr_person+graph->nrow, 1.0);
    };
    ~PARAMETER() //析构函数
    {
        CUDAFREE(person);
        CUDAFREE(pagerank_real);
    };
};


//结果
template<typename ValueType>
struct RESULT
{
    const GRAPH<ValueType>*  graph;
    std::string              alg;
    int*                     rnd;
    float*                   tim;
    double*                  norm;   //host变量
    double*                  maxerr; //host变量
    RESULT(const GRAPH<ValueType>* graph, const std::string alg, const int maxiter): graph(graph), alg(alg)
    {
        rnd = (int*)calloc(maxiter, sizeof(int));
        tim = (float*)calloc(maxiter, sizeof(float));
        norm = (double*)calloc(maxiter, sizeof(double));
        maxerr = (double*)calloc(maxiter, sizeof(double));
    };
    ~RESULT() //析构函数
    {
        FREE(rnd);
        FREE(tim);
        FREE(norm);
        FREE(maxerr);
    };
};

template<typename ValueType> struct CudaDataType;
template<> struct CudaDataType<half>{ static const cudaDataType value = cudaDataType::CUDA_R_16F;};
template<> struct CudaDataType<float>{ static const cudaDataType value = cudaDataType::CUDA_R_32F;};
template<> struct CudaDataType<double>{ static const cudaDataType value = cudaDataType::CUDA_R_64F;};
#endif