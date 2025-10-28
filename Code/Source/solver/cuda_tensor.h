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

    // Assign all elements to a scalar value (allows `CC = 0.0` style)
    __host__ __device__ CudaTensor4& operator=(double v) {
        for (int i = 0; i < nsd; ++i)
            for (int j = 0; j < nsd; ++j)
                for (int k = 0; k < nsd; ++k)
                    for (int l = 0; l < nsd; ++l)
                        data[i][j][k][l] = v;
        return *this;
    }

    // Elementwise addition
    __host__ __device__ CudaTensor4& operator+=(const CudaTensor4& o) {
        for (int i = 0; i < nsd; ++i)
            for (int j = 0; j < nsd; ++j)
                for (int k = 0; k < nsd; ++k)
                    for (int l = 0; l < nsd; ++l)
                        data[i][j][k][l] += o.data[i][j][k][l];
        return *this;
    }

    // Elementwise subtraction
    __host__ __device__ CudaTensor4& operator-=(const CudaTensor4& o) {
        for (int i = 0; i < nsd; ++i)
            for (int j = 0; j < nsd; ++j)
                for (int k = 0; k < nsd; ++k)
                    for (int l = 0; l < nsd; ++l)
                        data[i][j][k][l] -= o.data[i][j][k][l];
        return *this;
    }

    // Elementwise multiply-add by scalar (inplace)
    __host__ __device__ CudaTensor4& scale_add(double s, const CudaTensor4& o) {
        for (int i = 0; i < nsd; ++i)
            for (int j = 0; j < nsd; ++j)
                for (int k = 0; k < nsd; ++k)
                    for (int l = 0; l < nsd; ++l)
                        data[i][j][k][l] += s * o.data[i][j][k][l];
        return *this;
    }
};

// Non-member helpers: +, -, scalar * tensor
template <size_t nsd>
__host__ __device__ inline CudaTensor4<nsd> operator+(const CudaTensor4<nsd>& a, const CudaTensor4<nsd>& b) {
    CudaTensor4<nsd> r;
    for (int i = 0; i < nsd; ++i)
        for (int j = 0; j < nsd; ++j)
            for (int k = 0; k < nsd; ++k)
                for (int l = 0; l < nsd; ++l)
                    r(i,j,k,l) = a(i,j,k,l) + b(i,j,k,l);
    return r;
}

template <size_t nsd>
__host__ __device__ inline CudaTensor4<nsd> operator-(const CudaTensor4<nsd>& a, const CudaTensor4<nsd>& b) {
    CudaTensor4<nsd> r;
    for (int i = 0; i < nsd; ++i)
        for (int j = 0; j < nsd; ++j)
            for (int k = 0; k < nsd; ++k)
                for (int l = 0; l < nsd; ++l)
                    r(i,j,k,l) = a(i,j,k,l) - b(i,j,k,l);
    return r;
}

template <size_t nsd>
__host__ __device__ inline CudaTensor4<nsd> operator*(double s, const CudaTensor4<nsd>& a) {
    CudaTensor4<nsd> r;
    for (int i = 0; i < nsd; ++i)
        for (int j = 0; j < nsd; ++j)
            for (int k = 0; k < nsd; ++k)
                for (int l = 0; l < nsd; ++l)
                    r(i,j,k,l) = s * a(i,j,k,l);
    return r;
}

template <size_t nsd>
__host__ __device__ inline CudaTensor4<nsd> operator*(const CudaTensor4<nsd>& a, double s) {
    return s * a;
}

// Unary negation
template <size_t nsd>
__host__ __device__ inline CudaTensor4<nsd> operator-(const CudaTensor4<nsd>& a) {
    CudaTensor4<nsd> r;
    for (int i = 0; i < nsd; ++i)
        for (int j = 0; j < nsd; ++j)
            for (int k = 0; k < nsd; ++k)
                for (int l = 0; l < nsd; ++l)
                    r(i,j,k,l) = -a(i,j,k,l);
    return r;
}

#endif // CUDA_TENSOR_H