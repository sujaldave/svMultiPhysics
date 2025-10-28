#ifndef GPU_DIAGNOSTIC_HPP
#define GPU_DIAGNOSTIC_HPP

#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdlib>

namespace gpu_diagnostic {

// Run a small GPU computation to verify device execution
void verify_gpu_compute(int argc=0, char* argv[]=nullptr) {
  // Initialize Kokkos with CUDA device
  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings settings;
    settings.set_device_id(0);                    // Use first GPU
    settings.set_num_threads(1);                  // Single CPU thread since we use GPU
    settings.set_disable_warnings(false);         // Show warnings for debugging
    Kokkos::initialize(settings);
  }
  Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  // Print pre-compute diagnostics
  std::cout << "\n[GPU Diagnostic] Pre-compute execution space: "
            << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

  // Explicitly check if CUDA is enabled 
  #ifdef KOKKOS_ENABLE_CUDA
  std::cout << "[GPU Diagnostic] CUDA support is enabled\n";
  #else
  std::cout << "[GPU Diagnostic] CUDA support is NOT enabled\n";
  #endif

  // Create a small device View and try to force GPU allocation
  Kokkos::View<double*, Kokkos::CudaSpace> d_data("test_data", 1024);
  
  // Fill with a simple kernel
  Kokkos::parallel_for("test_kernel", 1024, KOKKOS_LAMBDA(const int i) {
    d_data(i) = static_cast<double>(i * 2);
  });

  // Force synchronization 
  Kokkos::fence();

  // Get result back to host to verify computation occurred
  auto h_data = Kokkos::create_mirror_view(d_data);
  Kokkos::deep_copy(h_data, d_data);

  // Print post-compute diagnostics  
  std::cout << "[GPU Diagnostic] Kernel execution complete\n";
  std::cout << "[GPU Diagnostic] Sample values: " 
            << h_data(0) << ", " << h_data(10) << ", " << h_data(100) << "\n";
  
  std::cout << "[GPU Diagnostic] Current execution space: "
            << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

  #ifdef KOKKOS_ENABLE_CUDA 
  std::cout << "[GPU Diagnostic] CUDA enabled" << std::endl;
  #endif

  std::cout << "[GPU Diagnostic] Sample computed values: "
            << h_data(0) << ", " << h_data(10) << ", " << h_data(100) << std::endl;
}

} // namespace gpu_diagnostic

#endif