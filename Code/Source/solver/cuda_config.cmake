# CUDA Configuration
# Find CUDA
find_package(CUDA 12.0 REQUIRED)

# Configure CUDA
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_SEPARABLE_COMPILATION ON)
    
    # Set nvcc flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++17)
    
    # Set compute capability
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -gencode arch=compute_75,code=sm_75
        -gencode arch=compute_80,code=sm_80
        -gencode arch=compute_86,code=sm_86)
endif()

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})
