# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/sujal/research/svGPU/svMultiPhysics/Code"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-build"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-prefix"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-prefix/tmp"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-prefix/src/svMultiPhysics-stamp"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-prefix/src"
  "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-prefix/src/svMultiPhysics-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-prefix/src/svMultiPhysics-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-prefix/src/svMultiPhysics-stamp${cfgdir}") # cfgdir has leading slash
endif()
