#include <iostream>
#include <cuda_runtime.h>

void retrieve_data_from_gpu(cudaIpcMemHandle_t d_tensor_address, size_t size)
{
    cudaSetDevice(0); // Optionally set the CUDA device

    // Cast the given address to a void pointer
    void *d_tensor;
    cudaError_t status = cudaIpcOpenMemHandle(&d_tensor, d_tensor_address, cudaIpcMemLazyEnablePeerAccess);

    // Allocate CPU memory to receive the data
    void *h_data = malloc(size);
    if (!h_data)
    {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return;
    }
    // log the inputs
    std::cout << "Data pointer: " << d_tensor_address << std::endl;
    std::cout << "Size: " << size << std::endl;

    // Copy the data from GPU to CPU
    cudaError_t memcpy_status = cudaMemcpy(h_data, d_tensor, size, cudaMemcpyDeviceToHost);
    if (memcpy_status != cudaSuccess)
    {
        std::cerr << "cudaMemcpy failed with error: " << cudaGetErrorString(memcpy_status) << std::endl;
    }
    else
    {
        std::cout << "Data successfully copied from GPU to CPU." << std::endl;
        // Here you can cast `h_data` to the appropriate type and use it
        // For example, if the data is an array of integers:
        int *int_data = static_cast<int *>(h_data);
        for (size_t i = 0; i < size / sizeof(int); ++i)
        {
            std::cout << "Data[" << i << "] = " << int_data[i] << std::endl;
        }
    }

    // Free the allocated host memory
    free(h_data);
}

int main()
{
    cudaIpcMemHandle_t data_ptr = 0x7f2937e00000; // Replace with your actual pointer
    size_t size = 4000;                  // Replace with your actual size

    // Retrieve the data from GPU
    retrieve_data_from_gpu(data_ptr, size);

    return 0;
}