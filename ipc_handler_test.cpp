#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

bool loadIpcHandleFromFile(const std::string &filename, cudaIpcMemHandle_t &ipc_handle)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&ipc_handle), sizeof(cudaIpcMemHandle_t));
    if (!file)
    {
        std::cerr << "Failed to read IPC handle from file: " << filename << std::endl;
        return false;
    }

    file.close();
    return true;
}

void printGpuData(void *d_tensor, size_t size)
{
    // Allocate memory on the host to copy the data from the GPU
    float *h_data = new float[size / sizeof(float)];

    // Copy data from GPU to host
    cudaError_t err = cudaMemcpy(h_data, d_tensor, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        delete[] h_data;
        return;
    }

    // Print the data
    std::cout << "Data in GPU memory:" << std::endl;
    for (size_t i = 0; i < size / sizeof(float); ++i)
    {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_data;
}

int main()
{
    cudaIpcMemHandle_t ipc_handle;
    if (!loadIpcHandleFromFile("bn1.bias_ipc_handle.bin", ipc_handle))
    {
        std::cerr << "Failed to load IPC handle from file." << std::endl;
        return -1;
    }

    // Open the IPC memory handle to access the GPU memory
    void *d_tensor;
    cudaError_t status = cudaIpcOpenMemHandle(&d_tensor, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    if (status != cudaSuccess)
    {
        std::cerr << "cudaIpcOpenMemHandle failed: " << cudaGetErrorString(status) << std::endl;
        return -1;
    }

    std::cout << "Imported GPU memory at address: " << &ipc_handle << std::endl;

    // Assuming you know the size of the data
    size_t data_size = 256; // Replace with actual size

    // Print the data from the GPU
    printGpuData(d_tensor, data_size);

    // Close the IPC handle
    cudaIpcCloseMemHandle(d_tensor);

    return 0;
}
