#ifndef __FUN__
#define __FUN__
#include "./tile.cuh"
template<typename ValueType> void Fun_CSRMV(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_COOMV(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_MERGEMV(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_GKOCSR(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_GKOCOO(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_GKOSELLP(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_GKOHYB(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_CSR5(const ENV& env, PARAMETER<ValueType>& para);
template<typename ValueType> void Fun_HOLA(const ENV& env, PARAMETER<ValueType>& para);
#endif
