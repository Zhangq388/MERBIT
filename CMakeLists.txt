cmake_minimum_required(VERSION 3.18)
project(SPMV LANGUAGES CXX CUDA)

set(VBLOCK     256   CACHE STRING "Value for -DVBLOCK")
set(VSIGMA     14    CACHE STRING "Value for -DVSIGMA")
set(VALUETYPE float CACHE STRING "Value for -DVALUETYPE")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(main
               src/main.cu
               src/class.cu
               src/spmv.cu)

target_include_directories(main PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(main PRIVATE
                           VBLOCK=${VBLOCK}
                           VSIGMA=${VSIGMA}
                           VALUETYPE=${VALUETYPE})

target_compile_options(main PRIVATE
                       $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
                       $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
                       $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
                       $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
                       $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g;-G>
                       $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g>)

set_target_properties(main PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON 
                      POSITION_INDEPENDENT_CODE ON) 

#CUDA库                      
find_package(CUDAToolkit REQUIRED)
target_link_libraries(main PRIVATE
                      CUDA::cudart 
                      CUDA::cublas
                      CUDA::cusparse)

#Ginkgo库
find_package(Ginkgo CONFIG REQUIRED)
target_link_libraries(main PRIVATE
                      Ginkgo::ginkgo
                      Ginkgo::ginkgo_omp
                      Ginkgo::ginkgo_cuda
                      Ginkgo::ginkgo_device
                      Ginkgo::ginkgo_reference
                      Ginkgo::ginkgo_hip
                      Ginkgo::ginkgo_dpcpp)
