#include "../include/fun.cuh"
std::fstream      FSTR;
int main(int argc, char* argv[])
{  
    //变量赋值
    const int device_id = 2;
    const std::string config = "/home/ta/zhangq388/SPMV/config/config5.txt";
    const std::string log = "/home/ta/zhangq388/SPMV/log.txt";
    FSTR.open(log, std::ios_base::app);
    //Timer   timer;
    //float   tim(0);
    
    int gid = std::stoi(argv[1]);
    
    //环境
    ENV env(device_id, config, log);
    
    //预热
    format_warmup<int>();

    using ValueType = VALUETYPE;
    
    //开始运行
    //读取图
    GRAPH<ValueType>   graph(config, gid); 
    //graph.show();
    //graph.desc(env);
    //graph.dangling(env);
    
    //SPMV
    //定义参数
    PARAMETER<ValueType>  para(&graph, 0.85, 20, 20);
    
    const int BLOCK = 256;
    const int SIGMA = VSIGMA;
    
    //SPMV
    //ValueType* Y0 = (ValueType*)calloc(graph.nrow, sizeof(ValueType));
    //ValueType* Y1 = (ValueType*)calloc(graph.nrow, sizeof(ValueType));
    
    //Fun_CSRMV<ValueType>(env, para);
    //cudaMemcpy(Y0, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
    
    //Fun_COOMV<ValueType>(env, para);
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);

    //TRYCATCH(Fun_GKOCSR<ValueType>(env, para));
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
    
    TRYCATCH(Fun_GKOCOO<ValueType>(env, para));
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);

    //Fun_GKOSELLP<ValueType>(env, para);
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
    
    //TRYCATCH(Fun_GKOHYB<ValueType>(env, para));
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);

    //Fun_MERGEMV<ValueType>(env, para);
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
    
    //Fun_HOLA<ValueType>(env, para);
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);

    //Fun_Merbit<ValueType, BLOCK, SIGMA>(env, para);
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);

    //Fun_CSR5<ValueType>(env, para);
    //cudaMemcpy(Y1, para.pagerank_real, graph.nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);

    //std::cout << "Y0:\n";
    //SHOW(Y0, 0, para.graph->nrow);
    //SHOW(Y0, 0, 15);
    //std::cout << "Y1:\n";
    //SHOW(Y1, 0, para.graph->nrow);
    //SHOW(Y1, 0, 15);
    
    std::cout << "graph = " << para.graph->graph_id << ", block = " << BLOCK << ", sigma = " << SIGMA << std::endl;
    //Fun_Check<ValueType>(Y0, Y1, graph.nrow);
    
    FSTR.close();

    return 0;
}