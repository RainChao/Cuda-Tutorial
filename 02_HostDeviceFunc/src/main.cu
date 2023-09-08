#include <cuda_runtime_api.h>
#include <iostream>

__host__ void HostOnly() {
  std::cout << "This function may only be called from host" << std::endl;
}

__device__ void DeviceOnly() {
  printf("This function may only be called from device\n");
}

__host__ __device__ void HostDevicePrint() {
#if defined(__CUDA_ARCH__)
  printf("host & device print\n");
#else
  std::cout << "host & device print" << std::endl;
#endif
}

__host__ __device__ float SquareAnywhere(float x)
{
    return x * x;
}

__global__ void RunGPU(float x)
{
    DeviceOnly();
    HostDevicePrint();
    printf("%f\n", SquareAnywhere(x));
}

void RunCPU(float x)
{
    HostOnly();
    HostDevicePrint();
    std::cout << SquareAnywhere(x) << std::endl;
}

int main()
{
    std::cout << "==== Sample 02 - Host / Device Functions ====\n" << std::endl;
    /*
     Expected output:
     "This function may only be called from the host"
     1764
     "This function may only be called from the device"
     1764.00
    */

    RunCPU(42);
    RunGPU<<<1, 1>>>(42);
    cudaDeviceSynchronize();
    return 0;
}
