#include <cuda_runtime_api.h>
#include <iostream>

// Declaration of a device variable in constant memory
__constant__ int cFoo;

__global__ void ReadConstantMemory()
{
    printf("GPU: Reading constant memory --> %x\n", cFoo);
}

// Definition of a device variable in global memory
__device__ const int dFoo = 42;

__global__ void ReadGlobalMemory(const int* __restrict dBarPtr)
{
    printf("GPU: Reading global memory --> %d %x\n", dFoo, *dBarPtr);
}

__global__ void WriteGlobalMemory(int* __restrict dOutPtr)
{
    *dOutPtr = dFoo * dFoo;
}

__device__ void WriteAndPrintSharedMemory(int* sFoo)
{
    // Write a computed result to shared memory for other threads to see
    sFoo[threadIdx.x] = 42 * (threadIdx.x + 1);
    // We make sure that no thread prints while the other still writes (parallelism!)
    __syncwarp();
    // Print own computed result and result by neighbor
    printf("ThreadID: %d, sFoo[0]: %d, sFoo[1]: %d\n", threadIdx.x, sFoo[0], sFoo[1]);
}

__device__ void clock_block(clock_t *d_o, clock_t clock_count)
{
    clock_t start_clock = clock64();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock64() - start_clock;
    }
     d_o[0] = clock_offset;
}

__device__ void WriteAndPrintSharedMemoryFake(int* val) {
    if (3 == threadIdx.x) {
        clock_t now_clock;
        clock_block(&now_clock, 5000000000);
    }
    val[threadIdx.x] = threadIdx.x;
    for (int i = 0; i < blockDim.x; i++) 
        printf("ThreadID: %d, idx: %d, val: %d\n", threadIdx.x, i, val[i]);
}

__global__ void WriteAndPrintSharedMemoryFixed()
{
    // Fixed allocation of two integers in shared memory
    __shared__ int sFoo[32];
    // Use it for efficient exchange of information
    WriteAndPrintSharedMemoryFake(sFoo);
}

__global__ void WriteAndPrintSharedMemoryDynamic()
{
    // Use dynamically allocated shared memory
    extern __shared__ int sFoo[];
    // Use it for efficient exchange of information
    WriteAndPrintSharedMemory(sFoo);
}

__device__  int d_val;
__global__ void ConstantToGlobal(int* val) {
    d_val = cFoo;
    *val = d_val;
}
__global__ void PrintGlobalVal() {
    printf("global value= %d\n", d_val);
}

int main()
{
    std::cout << "==== Sample 06 - Memory Basics ====\n" << std::endl;
    /*
     Expected output:
        GPU: Reading constant memory --> caffe
        GPU: Reading global memory --> 42 caffe
        CPU: Copied back from GPU --> 1764

        Using static shared memory to share computed results
        ThreadID: 0, sFoo[0]: 42, sFoo[1]: 84
        ThreadID: 1, sFoo[0]: 42, sFoo[1]: 84

        Using dynamic shared memory to share computed results
        ThreadID: 0, sFoo[0]: 42, sFoo[1]: 84
        ThreadID: 1, sFoo[0]: 42, sFoo[1]: 84
    */

    const int bar = 0xcaffe;
    /*
     Uniform variables should best be placed in constant
     GPU memory. Can be updated with cudaMemcpyToSymbol.
     This syntax is unusual, but this is how it should be
    */
    cudaMemcpyToSymbol(cFoo, &bar, sizeof(int));
    ReadConstantMemory<<<1, 1>>>();
    cudaDeviceSynchronize();

    /*
     Larger or read-write data is easiest provisioned by
     global memory. Can be allocated with cudaMalloc and
     updated with cudaMemcpy. Must be free'd afterward.
    */
    int* dBarPtr;
    cudaMalloc((void**)&dBarPtr, sizeof(int));
    cudaMemcpy(dBarPtr, &bar, sizeof(int), cudaMemcpyHostToDevice);
    ReadGlobalMemory<<<1, 1>>>(dBarPtr);
    cudaDeviceSynchronize();
    cudaFree(dBarPtr);

    /*
     The CPU may also read back updates from the GPU by
     copying the relevant data from global memory after
     running the kernel. Notice that here, we do not use
     cudaDeviceSynchronize: cudaMemcpy will synchronize
     with the CPU automatically.
    */
    int out, *dOutPtr;
    cudaMalloc((void**)&dOutPtr, sizeof(int));
    WriteGlobalMemory<<<1,1>>>(dOutPtr);
    cudaMemcpy(&out, dOutPtr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dOutPtr);
    std::cout << "CPU: Copied back from GPU --> " << out << std::endl;

    /*
    For information that is shared only within a single threadblock,
    we can also use shared memory, which is usually more efficient than
    global memory. Shared memory for a block may be statically allocated
    inside the kernel, or dynamically allocated at the kernel launch. In
    the latter case, the size of the required shared memory is provided as
    the third launch parameter, and the kernel will be able to access the 
    allocated shared memory via an array with the "extern" decoration. 
    Below, we use both methods to provide shared memory for a kernel with 
    two threads that exchange computed integers. 
    */
    std::cout << "\nUsing static shared memory to share computed results" << std::endl;
    WriteAndPrintSharedMemoryFixed<<<1, 2>>>();
    cudaDeviceSynchronize();

    std::cout << "\nUsing dynamic shared memory to share computed results" << std::endl;
    WriteAndPrintSharedMemoryDynamic<<<1, 2, 1 * sizeof(int)>>>();
    cudaDeviceSynchronize();

    // exercise 1
    int* new_ptr;
    cudaMalloc((void**)&new_ptr, sizeof(int));
    ConstantToGlobal<<<1,1>>>(new_ptr);
    int new_val;
    cudaMemcpy(&new_val, new_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(new_ptr);
    printf("new value = 0x%x\n", new_val);
    PrintGlobalVal<<<1,1>>>();
    PrintGlobalVal<<<1,1>>>();
    cudaDeviceSynchronize();

    // exercise 2
    WriteAndPrintSharedMemoryFixed<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}