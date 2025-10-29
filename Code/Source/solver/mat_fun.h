// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAT_FUN_H 
#define MAT_FUN_H 
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include <stdexcept>

#include "Array.h"
#include "Tensor4.h"
#include "Vector.h"

/// @brief The classes defined here duplicate the data structures in the 
/// Fortran MATFUN module defined in MATFUN.f. 
///
/// This module defines data structures for generally performed matrix and tensor operations.
///
/// \todo [TODO:DaveP] this should just be a namespace?
//
namespace mat_fun {
    // Define templated type aliases for Eigen matrices and tensors for convenience
    template<size_t nsd>
    using Matrix = Eigen::Matrix<double, nsd, nsd>;

        template<size_t nsd>
    class RawTensor {
    public:
        RawTensor() {
            data_.fill(0.0);
        }

        RawTensor(const RawTensor<nsd>& other) {
            std::copy(other.data_.begin(), other.data_.end(), data_.begin());
        }

        RawTensor& operator=(const RawTensor<nsd>& other) {
            if (this != &other) {
                std::copy(other.data_.begin(), other.data_.end(), data_.begin());
            }
            return *this;
        }

        RawTensor& operator+=(const RawTensor<nsd>& other) {
            for (size_t i = 0; i < nsd*nsd*nsd*nsd; ++i) {
                data_[i] += other.data_[i];
            }
            return *this;
        }

        RawTensor operator+(const RawTensor<nsd>& other) const {
            RawTensor<nsd> result(*this);
            result += other;
            return result;
        }

        RawTensor operator-(const RawTensor<nsd>& other) const {
            RawTensor<nsd> result(*this);
            for (size_t i = 0; i < nsd*nsd*nsd*nsd; ++i) {
                result.data_[i] -= other.data_[i];
            }
            return result;
        }

        RawTensor& operator-=(const RawTensor<nsd>& other) {
            for (size_t i = 0; i < nsd*nsd*nsd*nsd; ++i) {
                data_[i] -= other.data_[i];
            }
            return *this;
        }

        double& operator()(size_t i, size_t j, size_t k, size_t l) {
            return data_[i + nsd*(j + nsd*(k + nsd*l))];
        }

        double operator()(size_t i, size_t j, size_t k, size_t l) const {
            return data_[i + nsd*(j + nsd*(k + nsd*l))];
        }

        void setZero() {
            data_.fill(0.0);
        }

        // Friend functions for scalar multiplication
        friend RawTensor<nsd> operator*(const RawTensor<nsd>& tensor, double scalar) {
            RawTensor<nsd> result;
            for (size_t i = 0; i < nsd*nsd*nsd*nsd; ++i) {
                result.data_[i] = tensor.data_[i] * scalar;
            }
            return result;
        }

        friend RawTensor<nsd> operator*(double scalar, const RawTensor<nsd>& tensor) {
            return tensor * scalar;
        }

    private:
        std::array<double, nsd*nsd*nsd*nsd> data_;
    };

    template<size_t nsd>
    using Tensor = RawTensor<nsd>;

    // Function to convert Array<double> to Eigen::Matrix
    template <typename MatrixType>
    MatrixType convert_to_eigen_matrix(const Array<double>& src) {
        MatrixType mat;
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                mat(i, j) = src(i, j);
        return mat;
    }

    // Function to convert Eigen::Matrix to Array<double>
    template <typename MatrixType>
    void convert_to_array(const MatrixType& mat, Array<double>& dest) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                dest(i, j) = mat(i, j);
    }

    // Function to convert a higher-dimensional array like Dm
    template <typename MatrixType>
    void copy_Dm(const MatrixType& mat, Array<double>& dest, int rows, int cols) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                dest(i, j) = mat(i, j);
    }

    template <int nsd>
    Eigen::Matrix<double, nsd, 1> cross_product(const Eigen::Matrix<double, nsd, 1>& u, const Eigen::Matrix<double, nsd, 1>& v) {
        if constexpr (nsd == 2) {
            return Eigen::Matrix<double, 2, 1>(v(1), - v(0));
        }
        else if constexpr (nsd == 3) {
            return u.cross(v);
        }
        else {
            throw std::runtime_error("[cross_product] Invalid number of spatial dimensions '" + std::to_string(nsd) + "'. Valid dimensions are 2 or 3.");
        }
    }

    double mat_ddot(const Array<double>& A, const Array<double>& B, const int nd);
    
    template <int nsd>
    double double_dot_product(const Matrix<nsd>& A, const Matrix<nsd>& B) {
        return A.cwiseProduct(B).sum();
    }
    
    double mat_det(const Array<double>& A, const int nd);
    Array<double> mat_dev(const Array<double>& A, const int nd);

    Array<double> mat_dyad_prod(const Vector<double>& u, const Vector<double>& v, const int nd);

    Array<double> mat_id(const int nsd);
    Array<double> mat_inv(const Array<double>& A, const int nd, bool debug = false);
    Array<double> mat_inv_ge(const Array<double>& A, const int nd, bool debug = false);
    Array<double> mat_inv_lp(const Array<double>& A, const int nd);

    Vector<double> mat_mul(const Array<double>& A, const Vector<double>& v);
    Array<double> mat_mul(const Array<double>& A, const Array<double>& B);
    void mat_mul(const Array<double>& A, const Array<double>& B, Array<double>& result);
    void mat_mul6x3(const Array<double>& A, const Array<double>& B, Array<double>& C);

    Array<double> mat_symm(const Array<double>& A, const int nd);
    Array<double> mat_symm_prod(const Vector<double>& u, const Vector<double>& v, const int nd);

    double mat_trace(const Array<double>& A, const int nd);

    Tensor4<double> ten_asym_prod12(const Array<double>& A, const Array<double>& B, const int nd);
    Tensor4<double> ten_ddot(const Tensor4<double>& A, const Tensor4<double>& B, const int nd);
    Tensor4<double> ten_ddot_2412(const Tensor4<double>& A, const Tensor4<double>& B, const int nd);
    Tensor4<double> ten_ddot_3424(const Tensor4<double>& A, const Tensor4<double>& B, const int nd);

    /**
     * @brief Contracts two 4th order tensors A and B over two dimensions, 
     * 
     */
    template <int nsd>
    Tensor<nsd>
    double_dot_product(const Tensor<nsd>& A, const std::array<int, 2>& dimsA,
                        const Tensor<nsd>& B, const std::array<int, 2>& dimsB) {
        Tensor<nsd> C;
        
        // Manual contraction implementation
        for(int i = 0; i < nsd; ++i) {
            for(int j = 0; j < nsd; ++j) {
                for(int k = 0; k < nsd; ++k) {
                    for(int l = 0; l < nsd; ++l) {
                        for(int m = 0; m < nsd; ++m) {
                            for(int n = 0; n < nsd; ++n) {
                                if(dimsA == std::array<int,2>{2,3} && dimsB == std::array<int,2>{1,3}) {
                                    // Contract over indices (2,3) of A and (1,3) of B
                                    C(i,j,k,l) += A(i,j,m,n) * B(k,m,n,l);
                                } else if(dimsA == std::array<int,2>{1,3} && dimsB == std::array<int,2>{0,1}) {
                                    // Contract over indices (1,3) of A and (0,1) of B
                                    C(i,j,k,l) += A(i,m,n,j) * B(m,n,k,l);
                                }
                            }
                        }
                    }
                }
            }
        }
        return C;
    }

    Tensor4<double> ten_dyad_prod(const Array<double>& A, const Array<double>& B, const int nd);
    
    /**
     * @brief Compute the dyadic product of two 2nd order tensors A and B, C_ijkl = A_ij * B_kl
     * 
     * @tparam nsd, the number of spatial dimensions
     * @param A, the first 2nd order tensor
     * @param B, the second 2nd order tensor
     * @return Tensor<nsd>
     */
    template <int nsd>
    Tensor<nsd> 
    dyadic_product(const Matrix<nsd>& A, const Matrix<nsd>& B) {
        // Initialize the result tensor
        Tensor<nsd> C;

        // Compute the dyadic product: C_ijkl = A_ij * B_kl
        for (int i = 0; i < nsd; ++i) {
            for (int j = 0; j < nsd; ++j) {
                for (int k = 0; k < nsd; ++k) {
                    for (int l = 0; l < nsd; ++l) {
                        C(i,j,k,l) = A(i,j) * B(k,l);
                    }
                }
            }
        }
        // For some reason, in this case the Eigen::Tensor contract function is 
        // slower than the for loop implementation

        return C;
    }

    Tensor4<double> ten_ids(const int nd);

    /**
     * @brief Create a 4th order identity tensor:
     * I_ijkl = 0.5 * (δ_ik * δ_jl + δ_il * δ_jk)
     * 
     * @tparam nsd, the number of spatial dimensions
     * @return Tensor<nsd> 
     */
    template <int nsd>
    Tensor<nsd>
    fourth_order_identity() {
        // Initialize as zero
        Tensor<nsd> I;
        I.setZero();

        // Set only non-zero entries
        for (int i = 0; i < nsd; ++i) {
            for (int j = 0; j < nsd; ++j) {
                I(i,j,i,j) = 0.5;
                I(i,j,j,i) = 0.5;
            }
        }

        return I;
    }

    Array<double> ten_mddot(const Tensor4<double>& A, const Array<double>& B, const int nd);

    Tensor4<double> ten_symm_prod(const Array<double>& A, const Array<double>& B, const int nd);
    
    /// @brief Create a 4th order tensor from symmetric outer product of two matrices: C_ijkl = 0.5 * (A_ik * B_jl + A_il * B_jk)
    ///
    /// Reproduces 'FUNCTION TEN_SYMMPROD(A, B, nd) RESULT(C)'.
    //
    template <int nsd>
    Tensor<nsd>
    symmetric_dyadic_product(const Matrix<nsd>& A, const Matrix<nsd>& B) {
        // Initialize the result tensor
        Tensor<nsd> C;
        C.setZero();

        // Compute the symmetric product: C_ijkl = 0.5 * (A_ik * B_jl + A_il * B_jk)
        for (int i = 0; i < nsd; ++i) {
            for (int j = 0; j < nsd; ++j) {
                for (int k = 0; k < nsd; ++k) {
                    for (int l = 0; l < nsd; ++l) {
                        C(i,j,k,l) = 0.5 * (A(i,k) * B(j,l) + A(i,l) * B(j,k));
                    }
                }
            }
        }

        return C;
    }

    Tensor4<double> ten_transpose(const Tensor4<double>& A, const int nd);

    /**
     * @brief Performs a tensor transpose operation on a 4th order tensor A, B_ijkl = A_klij
     * 
     * @tparam nsd, the number of spatial dimensions
     * @param A, the input 4th order tensor
     * @return Tensor<nsd>
     */
    template <int nsd>
    Tensor<nsd>
    transpose(const Tensor<nsd>& A) {

        // Initialize the result tensor
        Tensor<nsd> B;

        // Permute the tensor indices to perform the transpose operation
        for (int i = 0; i < nsd; ++i) {
            for (int j = 0; j < nsd; ++j) {
                for (int k = 0; k < nsd; ++k) {
                    for (int l = 0; l < nsd; ++l) {
                        B(i,j,k,l) = A(k,l,i,j);
                    }
                }
            }
        }

        return B;
    }

    Array<double> transpose(const Array<double>& A);

    void ten_init(const int nd);

};

#endif

