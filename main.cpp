#include <iostream>
#include <cstring>
#include<hip/hip_runtime.h>
#include<hip/hip_fp16.h>
#include<unistd.h>
#include <type_traits>
#include <vector>
#include <sys/wait.h>
#include <fcntl.h>

using float16 = _Float16;

void HIP_CALL(hipError_t err)
{
    if(err != hipSuccess) {
        std::cout << "HIP Error: " << (int)err << " " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

// Timer for measuring kernel duration
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

// Vector types. Useful for packed math (where supported) and MFMA inputs.
template<typename T, uint32_t Rank>
using vecT = T __attribute__((ext_vector_type(Rank)));

template<typename T> using vec4 = vecT<T, 4>;


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
    vec4<float16> value3 = ptr[2 * grid_size + tid];

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


// Host code


template<typename T> double fma_throughput_test(int count, int runs = 1)
{
    vec4<T>* buffer = nullptr;

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));
    
    int blocks = props.multiProcessorCount * 512;
    int threads_per_block = 64;
    int total_threads = blocks * threads_per_block;

    HIP_CALL(hipMalloc(&buffer, sizeof(vec4<T>) * total_threads * 4));
    
    HIPTimer t;
    t.start();
    for(int i = 0; i < runs; i++) {
        fma_throughput<T><<<blocks, threads_per_block>>>(buffer, count);
    }
    t.stop();
    HIP_CALL(hipDeviceSynchronize());

    double elapsed = t.elapsed();
    double ops = (double)total_threads * count * 64 * 16 * runs;
    double flops = (double)ops * 2.0 / elapsed;

    HIP_CALL(hipFree(buffer));

    return flops;
}

template<typename matT, typename accumT> double matmul_throughput_test(int count, int runs = 1)
{
    const int wave_size = 64;
    int k;
    int m;
    int n;

    if(std::is_same<matT, float16>::value) {
        m = 16;
        n = 16;
        k = 16;
    } else if(std::is_same<matT, float>::value) {
        m = 16;
        n = 16;
        k = 4;
    } else {
        assert(false);
    }
    
    int ops_per_matmul = k * m * n * 2;

    void* buffer = nullptr;
    void* accum = nullptr;

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));

    int blocks = props.multiProcessorCount * 512;
    int threads_per_block = wave_size;
    int total_threads = blocks * threads_per_block;

    HIP_CALL(hipMalloc(&buffer, 4 * sizeof(matT) * m * k * total_threads));
    HIP_CALL(hipMalloc(&accum, sizeof(accumT) * m * n * total_threads));

    HIPTimer t;
    t.start();
    for(int i = 0; i < runs; i++) {
        if(std::is_same<matT, float16>::value && std::is_same<accumT, float>::value) {
            matmul_fp16_throughput<<<blocks, threads_per_block>>>((vec4<float16>*)buffer, (vec4<float>*)accum, count);
        } else if(std::is_same<matT,float>::value && std::is_same<accumT, float>::value) {
            matmul_fp32_throughput<<<blocks, threads_per_block>>>((float*)buffer, (vec4<float>*)accum, count);
        }
    }
    t.stop();
    HIP_CALL(hipDeviceSynchronize());

    double elapsed = t.elapsed();
    double ops = (double)blocks * count * 64 * 4 * runs;
    double flops = (double)ops * ops_per_matmul / elapsed;

    HIP_CALL(hipFree(buffer));
    HIP_CALL(hipFree(accum));

    return flops;
}


enum : uint32_t {
    VALU_FP32 = 1 << 0,
    VALU_FP16 = 1 << 1,
    VALU_FP64 = 1 << 2,
    MFMA_FP16 = 1 << 3,
    MFMA_FP32 = 1 << 4,
    ALL = (uint32_t)-1
};

struct Result {
    int device;
    double valu_fp16;
    double valu_fp32;
    double valu_fp64;
    double mfma_fp16;
    double mfma_fp32;

    bool operator<(const Result& other) {
        return device < other.device;
    }
};

void print_result(const Result& res, uint32_t mask)
{
    std::cout << std::endl << "GPU " << res.device << std::endl;

    if(mask & VALU_FP16) {
        printf("VALU FP16: %.2f TFLOPS\n", res.valu_fp16 / 1e12);
    }
    if(mask & VALU_FP32) {
        printf("VALU FP32: %.2f TFLOPS\n", res.valu_fp32 / 1e12);
    }
    if(mask & VALU_FP64) {
        printf("VALU FP64: %.2f TFLOPS\n", res.valu_fp64 / 1e12);
    }
    if(mask & MFMA_FP16) {
        printf("MFMA FP16: %.2f TFLOPS\n", res.mfma_fp16 / 1e12);
    }
    if(mask & MFMA_FP32) {
        printf("MFMA FP32: %.2f TFLOPS\n", res.mfma_fp32 / 1e12);
    }
}

Result run_tests(int device, int runs, uint32_t mask)
{
    int device_count;

    HIP_CALL(hipGetDeviceCount(&device_count));

    if(device >= device_count) {
        std::cout << "Device " << device << " does not exist. Skipping..." << std::endl;
        exit(1);
    }

    HIP_CALL(hipSetDevice(device));

    Result res = {.device = device};

    if(mask & VALU_FP16) {
        res.valu_fp16 = fma_throughput_test<float16>(4096, runs);
    }

    if(mask & VALU_FP32) {
        res.valu_fp32 = fma_throughput_test<float>(4096, runs);
    }

    if(mask & VALU_FP64) {
        res.valu_fp64 = fma_throughput_test<double>(4096, runs);
    }

    if(mask & MFMA_FP16) {
        res.mfma_fp16 =matmul_throughput_test<float16, float>(4096, runs);
    }
    
    if(mask & MFMA_FP32) {
        res.mfma_fp32 = matmul_throughput_test<float, float>(4096, runs);
    }

    return res;
}

void run(std::vector<int>& devices, int runs, uint32_t mask)
{
    std::vector<pid_t> pids;
    std::vector<Result> results;

    int fd[2];

    if(pipe(fd)) {
        std::cout << std::strerror(errno) << std::endl;
        exit(1);
    }

    // Start a new process for each GPU
    for(auto d : devices) {
        pid_t pid = fork();

        if(pid == 0) {
            Result res = run_tests(d, runs, mask);

            write(fd[1], &res, sizeof(res));
            return;
        }
        pids.push_back(pid);
    }

    // Wait for all processes to finish
    for(auto pid : pids) {
        int status;
        waitpid(pid, &status, 0);
    }

    // Set reads to non-blocking in case one or more child processes
    // fails and does not write its results to the pipe
    int flags = fcntl(fd[0], F_GETFL, 0);
    fcntl(fd[0], F_SETFL, flags | O_NONBLOCK);

    for(auto pid : pids) {
        Result res;
        if(read(fd[0], &res, sizeof(res)) > 0) {
            results.push_back(res);
        }
    }

    // Sort results by GPU id
    std::sort(results.begin(), results.end());
 
    // Print results
    for(auto r : results) {
        print_result(r, mask);
    }
}

int main(int argc, char** argv)
{
    int runs = 1;

    uint32_t mask = 0;
    std::vector<int> devices;
    int device = 0;

    int i = 1;
    while(i < argc) {
        std::string arg = std::string(argv[i]);

        if(arg == "--device") {
            devices.push_back(atoi(argv[i + 1]));
            // Skip next 
            i++;
        } else if(arg == "--devices") {
            std::string s(argv[i + 1]);
            std::stringstream ss(s);
            std::string r;
            while(getline(ss, r, ',')) {
                devices.push_back(std::stoi(r));
            }
            // Skip next 
            i++;
        } else if(arg == "--runs") {
            runs = atoi(argv[i + 1]);

            // Skip next
            i++;
        } else if(arg == "--fp32") {
            mask |= VALU_FP32;
        } else if(arg == "--fp64") {
            mask |= VALU_FP64;
        } else if(arg == "--fp16") {
            mask |= VALU_FP16;
        } else if(arg == "--matfp16") {
            mask |= MFMA_FP16;
        } else if(arg == "--matfp32") {
            mask |= MFMA_FP32;
        } else {
            std::cout << "Invalid argument " << arg << std::endl;
            return 1;
        }

        i++;
    }

    if(devices.size() == 0) {
        devices.push_back(0);
    }

    if(mask == 0) {
        mask = ALL;
    }

    run(devices, runs, mask);

    return 0;
}


