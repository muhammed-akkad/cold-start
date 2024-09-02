#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>

int connectToDaemon()
{
    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        std::cerr << "Socket creation error" << std::endl;
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0)
    {
        std::cerr << "Invalid address / Address not supported" << std::endl;
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        std::cerr << "Connection Failed" << std::endl;
        return -1;
    }

    return sock;
}

std::string sendCommandToDaemon(int sock, const std::string &command)
{
    char buffer[1024] = {0};
    send(sock, command.c_str(), command.size(), 0);
    read(sock, buffer, 1024);
    return std::string(buffer);
}

bool receiveCudaIpcHandle(int sock, cudaIpcMemHandle_t &ipc_handle)
{
    // Receive the CUDA IPC handle from the daemon
    if (read(sock, &ipc_handle, sizeof(cudaIpcMemHandle_t)) != sizeof(cudaIpcMemHandle_t))
    {
        std::cerr << "Failed to receive IPC handle from daemon" << std::endl;
        return false;
    }
    return true;
}

// Function to save IPC handle to a file
void saveIpcHandleToFile(const std::string &filename, const cudaIpcMemHandle_t &ipc_handle)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char *>(&ipc_handle), sizeof(cudaIpcMemHandle_t));
    if (!file)
    {
        std::cerr << "Failed to write IPC handle to file: " << filename << std::endl;
    }

    file.close();
}

std::map<std::string, uint64_t> save_tensors_cpp(const std::vector<std::string> &tensor_names,
                                                           const std::map<std::string, std::pair<uint64_t, size_t>> &tensor_data_index)
{
    std::map<std::string, uint64_t> tensor_offsets;

    std::cout << "Starting save_tensors_cpp" << std::endl;

    int sock = connectToDaemon();
    if (sock < 0)
    {
        std::cerr << "Failed to connect to the daemon" << std::endl;
        return tensor_offsets;
    }

    for (const auto &name : tensor_names)
    {
        std::cout << "Processing tensor: " << name << std::endl;

        uint64_t data_ptr = tensor_data_index.at(name).first;
        size_t size = tensor_data_index.at(name).second;

        std::cout << "Data pointer: " << data_ptr << ", Size: " << size << std::endl;

        // Send allocate command to the daemon
        std::string command = "ALLOCATE " + name + " " + std::to_string(size);
        std::string response = sendCommandToDaemon(sock, command);

        if (response.find("ALLOCATED") == 0)
        {

            std::cout << "Allocated memory: " << data_ptr << " size: " << size << std::endl;

            cudaIpcMemHandle_t ipc_handle;
            if (!receiveCudaIpcHandle(sock, ipc_handle))
            {
                std::cerr << "Failed to receive IPC handle for tensor: " << name << std::endl;
                continue;
            }

            // Save the IPC handle to a file
            std::string filename = name + "_ipc_handle.bin";
            saveIpcHandleToFile(filename, ipc_handle);

            // Import the memory on the client side
            void *d_tensor;
            cudaError_t status = cudaIpcOpenMemHandle(&d_tensor, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            if (status != cudaSuccess)
            {
                std::cerr << "cudaIpcOpenMemHandle failed: " << cudaGetErrorString(status) << std::endl;
                continue;
            }

            std::cout << "Imported GPU memory for tensor: " << name << " at address: " << d_tensor << std::endl;

            // Copy data from host to the imported GPU memory
            cudaMemcpy(d_tensor, reinterpret_cast<void *>(data_ptr), size, cudaMemcpyHostToDevice);

            std::cout << "Copied data to GPU memory for tensor: " << name << std::endl;
            tensor_offsets[name] = reinterpret_cast<uint64_t>(d_tensor);
        }
        else
        {
            std::cerr << "Memory allocation failed for tensor: " << name << std::endl;
        }
    }

    close(sock);
    std::cout << "Finished save_tensors_cpp" << std::endl;
    return tensor_offsets;
}
torch::Tensor load_tensor_from_gpu(uint64_t ptr, std::vector<int64_t> shape, std::vector<int64_t> stride, std::string dtype_str)
{
    torch::ScalarType dtype;

    if (dtype_str == "torch.float32")
    {
        dtype = torch::kFloat32;
    }
    else if (dtype_str == "torch.float64")
    {
        dtype = torch::kFloat64;
    }
    else if (dtype_str == "torch.int32")
    {
        dtype = torch::kInt32;
    }
    else if (dtype_str == "torch.int64")
    {
        dtype = torch::kInt64;
    }
    else
    {
        throw std::runtime_error("Unsupported dtype: " + dtype_str);
    }

    void *data_ptr = reinterpret_cast<void *>(ptr);

    torch::TensorOptions options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);

    torch::Tensor tensor = torch::from_blob(data_ptr, shape, stride, options);

    return tensor;
}

PYBIND11_MODULE(cuda_loader, m)
{
}
PYBIND11_MODULE(cuda_saver, m)
{
    m.def("load_tensor_from_gpu", &load_tensor_from_gpu, "Load a tensor from a GPU memory pointer");

    m.def("save_tensors_cpp", &save_tensors_cpp, "Save tensors to GPU memory");
}
