CUDA_DIR ?= /usr/local/cuda

CC_SRCS := $(wildcard src/*.cc)
CC_OBJS := ${CC_SRCS:src/%.cc=build/obj/%.o}
CUDA_SRCS := $(wildcard src/*.cu)
CUDA_OBJS := ${CUDA_SRCS:src/%.cu=build/obj/%.o}
OBJS := $(CC_OBJS) $(CUDA_OBJS)

CC = g++
WARNINGS = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
CC_FLAGS = -std=c++11 -O3 -fPIC $(WARNINGS) -I$(CUDA_DIR)/include
LD_FLAGS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas

NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -std=c++11 -O3 -lineinfo --compiler-options '-fPIC'

# Auto-detect installed GPU SM architecture via nvidia-smi (fallback to sm_80)
DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
ifeq ($(DETECTED_SM),)
ARCH = -gencode arch=compute_70,code=sm_70 \
       -gencode arch=compute_80,code=[sm_80,compute_80]
else
ARCH = -gencode arch=compute_$(DETECTED_SM),code=[sm_$(DETECTED_SM),compute_$(DETECTED_SM)]
endif

all: build/lib/libc_runtime_api.so

build/lib/libc_runtime_api.so: $(OBJS)
	@mkdir -p build/lib
	$(CC) -shared $^ -o $@ $(LD_FLAGS)

build/obj/%.o: src/%.cc
	@mkdir -p build/obj
	$(CC) $(CC_FLAGS) -c $< -o $@

build/obj/%.o: src/%.cu
	@mkdir -p build/obj
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: all clean

