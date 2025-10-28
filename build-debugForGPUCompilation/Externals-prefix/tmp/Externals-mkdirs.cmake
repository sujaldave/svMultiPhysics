# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/sujal/research/svGPU/svMultiPhysics/Externals"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-build"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-prefix"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-prefix/tmp"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-prefix/src/Externals-stamp"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-prefix/src"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-prefix/src/Externals-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-prefix/src/Externals-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/Externals-prefix/src/Externals-stamp${cfgdir}") # cfgdir has leading slash
endif()
