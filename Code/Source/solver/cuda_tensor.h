#ifndef CUDA_TENSOR_H
#define CUDA_TENSOR_H

// When compiling with a non-CUDA host compiler the CUDA qualifiers
// `__host__` and `__device__` are not defined. Define them as empty
// so headers using them can be included from .cpp files compiled by g++.
#if !defined(__CUDACC__)
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

template<size_t nsd>
class CudaTensor4 {
private:
    double data[nsd][nsd][nsd][nsd];

public:
    __host__ __device__ CudaTensor4() {
        for (int i = 0; i < nsd; i++)
            for (int j = 0; j < nsd; j++)
                for (int k = 0; k < nsd; k++)
                    for (int l = 0; l < nsd; l++)
                        data[i][j][k][l] = 0.0;
    }

    __host__ __device__ double& operator()(int i, int j, int k, int l) {
        return data[i][j][k][l];
    }

    __host__ __device__ const double& operator()(int i, int j, int k, int l) const {
        return data[i][j][k][l];
    }

    __host__ __device__ void setZero() {
        for (int i = 0; i < nsd; i++)
            for (int j = 0; j < nsd; j++)
                for (int k = 0; k < nsd; k++)
                    for (int l = 0; l < nsd; l++)
                        data[i][j][k][l] = 0.0;
    }
};

#endif // CUDA_TENSOR_H