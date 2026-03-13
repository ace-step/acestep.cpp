#!/bin/bash

rm -rf build
mkdir build
cd build

cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON \
         -DVulkan_LIBRARY=/system/lib64/libvulkan.so
cmake --build . --config Release -j "$(nproc)"
