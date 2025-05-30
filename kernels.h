#ifndef _ROCBENCH_KERNELS_H
#define _ROCBENCH_KERNELS_H

using float16 = _Float16;

// Vector types. Useful for packed math (where supported) and MFMA inputs.
template<typename T, uint32_t Rank>
using vecT = T __attribute__((ext_vector_type(Rank)));

template<typename T> using vec4 = vecT<T, 4>;
template<typename T> using vec8 = vecT<T, 8>;


// Kernels


template<typename T> __global__ void fma_throughput(vec4<T>* buffer, int count)
{
    const T k = 1.0;

    const int grid_size = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<T>* ptr = buffer;

    vec4<T> value0 = ptr[0 * grid_size + tid];
    vec4<T> value1 = ptr[1 * grid_size + tid];
    vec4<T> value2 = ptr[2 * grid_size + tid];
    vec4<T> value3 = ptr[3 * grid_size + tid];

    for(int j = 0; j < count; j++) {
        for(int j = 0; j < 64; j++) {

            // 16 FMA ops
            value0 = value0 * value0 + k;
            value1 = value1 * value1 + k;
            value2 = value2 * value2 + k;
            value3 = value3 * value3 + k;
        }
    }

    ptr[tid] = value0 + value1 + value2 + value3;
}

__global__ void matmul_fp16_throughput(vec4<float16>* inputs, vec4<float>* outputs, int count)
{
    int grid_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<float16>* ptr = inputs;

    vec4<float16> value0 = ptr[0 * grid_size + tid];
    vec4<float16> value1 = ptr[1 * grid_size + tid];
    vec4<float16> value2 = ptr[2 * grid_size + tid];
    vec4<float16> value3 = ptr[3 * grid_size + tid];

    vec4<float> accum0;
    vec4<float> accum1;
    vec4<float> accum2;
    vec4<float> accum3;
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < 64; j++) {
            // 4 MFMA ops
            accum0 = __builtin_amdgcn_mfma_f32_16x16x16f16(value0, value0, accum0, 0, 0, 0);
            accum1 = __builtin_amdgcn_mfma_f32_16x16x16f16(value1, value1, accum1, 0, 0, 0);
            accum2 = __builtin_amdgcn_mfma_f32_16x16x16f16(value2, value2, accum2, 0, 0, 0);
            accum3 = __builtin_amdgcn_mfma_f32_16x16x16f16(value3, value3, accum3, 0, 0, 0);
        }
    }

    outputs[tid] = accum0 + accum1 + accum2 + accum3;
}

__global__ void sparse_matmul_fp16_throughput(vec4<float16>* input0, vec8<float16>* input1, vec4<float>* outputs, int count)
{
    int grid_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<float16>* x_ptr = input0;
    vec8<float16>* y_ptr = input1;

    vec4<float16> x0 = x_ptr[0 * grid_size + tid];
    vec4<float16> x1 = x_ptr[1 * grid_size + tid];
    vec4<float16> x2 = x_ptr[2 * grid_size + tid];
    vec4<float16> x3 = x_ptr[3 * grid_size + tid];
    
    vec8<float16> y0 = y_ptr[0 * grid_size + tid];
    vec8<float16> y1 = y_ptr[1 * grid_size + tid];
    vec8<float16> y2 = y_ptr[2 * grid_size + tid];
    vec8<float16> y3 = y_ptr[3 * grid_size + tid];
    
    vec4<float> accum0;
    vec4<float> accum1;
    vec4<float> accum2;
    vec4<float> accum3;
   
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < 64; j++) {
            // 4 SMFMAC ops
            accum0 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x0, y0, accum0, 0, 0, 0);
            accum1 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x1, y1, accum1, 0, 0, 0);
            accum2 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x2, y2, accum2, 0, 0, 0);
            accum3 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x3, y3, accum3, 0, 0, 0);
        }
    }

    outputs[tid] = accum0 + accum1 + accum2 + accum3;
}

__global__ void matmul_fp32_throughput(float* inputs, vec4<float>* outputs, int count)
{
    int grid_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float* ptr = inputs;

    float value0 = ptr[0 * grid_size + tid];
    float value1 = ptr[1 * grid_size + tid];
    float value2 = ptr[2 * grid_size + tid];
    float value3 = ptr[2 * grid_size + tid];

    vec4<float> accum0;
    vec4<float> accum1;
    vec4<float> accum2;
    vec4<float> accum3;
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < 64; j++) {
            // 4 MFMA ops
            accum0 = __builtin_amdgcn_mfma_f32_16x16x4f32(value0, value0, accum0, 0, 0, 0);
            accum1 = __builtin_amdgcn_mfma_f32_16x16x4f32(value1, value1, accum1, 0, 0, 0);
            accum2 = __builtin_amdgcn_mfma_f32_16x16x4f32(value2, value2, accum2, 0, 0, 0);
            accum3 = __builtin_amdgcn_mfma_f32_16x16x4f32(value3, value3, accum3, 0, 0, 0);
        }
    }

    outputs[tid] = accum0 + accum1 + accum2 + accum3;
}

#endif