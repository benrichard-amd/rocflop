#include <iostream>

#include<hip/hip_runtime.h>
#include<hip/hip_fp16.h>
#include<unistd.h>
#include <type_traits>

using float16 = _Float16;

void HIP_CALL(hipError_t err)
{
    if(err != hipSuccess) {
        std::cout << "HIP Error: " << (int)err << " " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

class HIPTimer {

private:
    hipEvent_t m_start;
    hipEvent_t m_stop;

public:
    HIPTimer()
    {
        HIP_CALL(hipEventCreate(&m_start));
        HIP_CALL(hipEventCreate(&m_stop));
    }

    void start()
    {
        HIP_CALL(hipEventRecord(m_start));
    }

    void stop()
    {
        HIP_CALL(hipEventRecord(m_stop));
    }

    double elapsed()
    {
        float ms;
        HIP_CALL(hipEventElapsedTime(&ms, m_start, m_stop));

        return (double)ms / 1000.0;
    }
};

template<typename T, uint32_t Rank>
using vecT = T __attribute__((ext_vector_type(Rank)));

template<typename T> using vec4 = vecT<T, 4>;

template<typename T> __global__ void fma_throughput(vec4<T>* buffer, int count)
{
    const T k = 1.0;

    const int grid_size = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<T>* ptr = buffer;

    vec4<T> value0 = ptr[0 * grid_size + tid];
    vec4<T> value1 = ptr[1 * grid_size + tid];
    vec4<T> value2 = ptr[2 * grid_size + tid];

    for(int j = 0; j < count; j++) {
        for(int j = 0; j < 64; j++) {

            // 12 MFA ops
            value0 = value0 * value0 + k;
            value1 = value1 * value1 + k;
            value2 = value2 * value2 + k;
        }
    }

    ptr[tid] = value0 + value1 + value2;
}

__global__ void matmul_fp16_throughput(vec4<float16>* inputs, vec4<float>* outputs, int count)
{
    int grid_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<float16>* ptr = inputs;

    vec4<float16> value0 = ptr[0 * grid_size + tid];
    vec4<float16> value1 = ptr[1 * grid_size + tid];
    vec4<float16> value2 = ptr[2 * grid_size + tid];

    vec4<float> accum0;
    vec4<float> accum1;
    vec4<float> accum2;
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < 64; j++) {
            // 3 MFMA ops
            accum0 = __builtin_amdgcn_mfma_f32_16x16x16f16(value0, value0, accum0, 0, 0, 0);
            accum1 = __builtin_amdgcn_mfma_f32_16x16x16f16(value1, value1, accum1, 0, 0, 0);
            accum2 = __builtin_amdgcn_mfma_f32_16x16x16f16(value2, value2, accum2, 0, 0, 0);
        }
    }

    outputs[tid] = accum0 + accum1 + accum2;
}


template<typename T> void fma_throughput_test(int count, int runs = 1)
{
    vec4<T>* buffer = nullptr;

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));

    int blocks = props.multiProcessorCount * 512;
    int threads_per_block = 64;
    int total_threads = blocks * threads_per_block;

    HIP_CALL(hipMalloc(&buffer, sizeof(vec4<T>) * total_threads * 3));

    HIPTimer t;
    t.start();
    for(int i = 0; i < runs; i++) {
        fma_throughput<T><<<blocks, threads_per_block>>>(buffer, count);
    }
    t.stop();
    HIP_CALL(hipDeviceSynchronize());

    double elapsed = t.elapsed();
    double ops = (double)total_threads * count * 64 * 12 * runs;
    double tflops = (double)ops / 1e12 / elapsed;
    printf("%.2fT FMA ops/sec (%.2f TFLOPS)\n", tflops, tflops * 2.0);

    HIP_CALL(hipFree(buffer));
}

template<typename matT, typename accumT> void matmul_throughput_test(int count, int runs = 1)
{
    const int wave_size = 64;
    int k;
    int m;
    int n;

    if(std::is_same<matT, float16>::value) {
        k = 16;
        m = 16;
        n = 16;
    }
    
    int matrix_ops = k * m * n * 2;

    void* buffer = nullptr;
    void* accum = nullptr;

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));

    int blocks = props.multiProcessorCount * 512;
    int threads_per_block = wave_size;
    int total_threads = blocks * threads_per_block;

    HIP_CALL(hipMalloc(&buffer, sizeof(matT) * m * k * total_threads * 3));
    HIP_CALL(hipMalloc(&accum, sizeof(accumT) * m * n * total_threads));

    HIPTimer t;
    t.start();
    for(int i = 0; i < runs; i++) {
        if(std::is_same<matT, float16>::value && std::is_same<accumT, float>::value) {
            matmul_fp16_throughput<<<blocks, threads_per_block>>>((vec4<float16>*)buffer, (vec4<float>*)accum, count);
        }
    }
    t.stop();
    HIP_CALL(hipDeviceSynchronize());

    double elapsed = t.elapsed();
    double ops = (double)blocks * count * 64 * 3 * runs;
    double tflops = (double)ops / 1e12 / elapsed;
    printf("%.2fT MFMA ops/sec (%.2f TFLOPS)\n", tflops, tflops * matrix_ops);

    HIP_CALL(hipFree(buffer));
    HIP_CALL(hipFree(accum));
}


int main(int argc, char** argv)
{
    int runs = 1;

    bool all = false;
    bool fp16 = false;
    bool fp32 = false;
    bool fp64 = false;
    bool matfp16 = false;

    int device = 0;

    int i = 1;
    while(i < argc) {
        std::string arg = std::string(argv[i]);

        if(arg == "--device") {
            device = atoi(argv[i + 1]);

            // Skip next 
            i++;
        } else if(arg == "--runs") {
            runs = atoi(argv[i + 1]);

            // Skip next
            i++;
        } else if(arg == "--all") {
            all = true;
        } else if(arg == "--fp32") {
            fp32 = true;
        } else if(arg == "--fp64") {
            fp64 = true;
        } else if(arg == "--fp16") {
            fp16 = true;
        } else if(arg == "--matfp32") {
            matfp16 = true;
        } else {
            std::cout << "Invalid argument " << arg << std::endl;
            return 1;
        }

        i++;
    }

    all |= !fp32 && !fp32 && !fp16 && !matfp16;

    HIP_CALL(hipSetDevice(device));

    if(all || fp16) {
        std::cout << "FP16:" << std::endl;
        fma_throughput_test<float16>(4096, runs);
    }

    if(all || fp32) {
        std::cout << "FP32:" << std::endl;
        fma_throughput_test<float>(4096, runs);
    }

    if(all || fp64) {
        std::cout << "FP64:" << std::endl;
        fma_throughput_test<double>(4096);
    }

    if(all || matfp16) {
        std::cout << "MFMA FP16:" << std::endl;
        matmul_throughput_test<float16, float>(1024);
    }

    return 0;
}


