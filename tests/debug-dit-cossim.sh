#!/bin/bash

cd ..
./build.sh
cd tests
./debug-dit-cossim.py --mode turbo --quant BF16 > CUDA-BF16.log
./debug-dit-cossim.py --mode turbo --quant Q8_0 > CUDA-Q8_0.log
./debug-dit-cossim.py --mode turbo --quant Q6_K > CUDA-Q6_K.log
./debug-dit-cossim.py --mode turbo --quant Q5_K_M > CUDA-Q5_K_M.log
./debug-dit-cossim.py --mode turbo --quant Q4_K_M > CUDA-Q4_K_M.log

cd ..
./buildvulkan.sh
cd tests
./debug-dit-cossim.py --mode turbo --quant BF16 > Vulkan-BF16.log
./debug-dit-cossim.py --mode turbo --quant Q8_0 > Vulkan-Q8_0.log
./debug-dit-cossim.py --mode turbo --quant Q6_K > Vulkan-CPU_Q6_K.log
./debug-dit-cossim.py --mode turbo --quant Q5_K_M > Vulkan-Q5_K_M.log
./debug-dit-cossim.py --mode turbo --quant Q4_K_M > Vulkan-Q4_K_M.log

cd ..
./buildcpu.sh
cd tests
./debug-dit-cossim.py --mode turbo --quant BF16 > CPU-BF16.log
./debug-dit-cossim.py --mode turbo --quant Q8_0 > CPU-Q8_0.log
./debug-dit-cossim.py --mode turbo --quant Q6_K > CPU-Q6_K.log
./debug-dit-cossim.py --mode turbo --quant Q5_K_M > CPU-Q5_K_M.log
./debug-dit-cossim.py --mode turbo --quant Q4_K_M > CPU-Q4_K_M.log
