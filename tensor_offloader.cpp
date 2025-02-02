#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>
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
#include <fcntl.h>    // for O_CREAT, O_RDWR
#include <sys/mman.h> // for mmap, shm_open
#include <sys/stat.h> // for ftruncate
#include <cstring>    // for memcpy
#include <thread>
#include <future>
#include <nlohmann/json.hpp>

// For convenience
using json = nlohmann::json;

int connectToServer()
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

std::string sendCommandToServer(int sock, const std::string &command)
{
    send(sock, command.c_str(), command.size(), 0);
    char buffer[1024] = {0};
    int bytes_received = recv(sock, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received > 0)
    {
        buffer[bytes_received] = '\0';
        return std::string(buffer);
    }
    else
    {
        perror("recv");
        return "";
    }
}

bool receiveCudaIpcHandle(int sock, cudaIpcMemHandle_t &ipc_handle)
{
    // Receive the CUDA IPC handle from the server
    if (read(sock, &ipc_handle, sizeof(cudaIpcMemHandle_t)) != sizeof(cudaIpcMemHandle_t))
    {
        std::cerr << "Failed to receive IPC handle from server" << std::endl;
        return false;
    }
    return true;
}

// Function to save IPC handle to a file
void saveIpcHandleToFile(const std::string &filename, const cudaIpcMemHandle_t &ipc_handle)
{
    std::string folder = "handlers_gpu/";
    std::string full_filename = folder + filename;

    std::ofstream file(full_filename, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << full_filename << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char *>(&ipc_handle), sizeof(cudaIpcMemHandle_t));
    if (!file)
    {
        std::cerr << "Failed to write IPC handle to file: " << full_filename << std::endl;
    }

    file.close();
}

std::map<std::string, uint64_t> saveTensorsToGpu(const std::vector<std::string> &tensor_names,
                                                 const std::map<std::string, std::pair<uint64_t, size_t>> &tensor_data_index)
{
    std::map<std::string, uint64_t> tensor_offsets;
    std::cout << "Starting saveTensorsToGpu" << std::endl;

    int sock = connectToServer();
    if (sock < 0)
    {
        std::cerr << "Failed to connect to the server" << std::endl;
        return tensor_offsets;
    }

    for (const auto &name : tensor_names)
    {
        std::cout << "Processing tensor: " << name << std::endl;

        auto it = tensor_data_index.find(name);
        if (it == tensor_data_index.end())
        {
            std::cerr << "Tensor data not found for: " << name << std::endl;
            continue;
        }

        uint64_t data_ptr = it->second.first;
        size_t size = it->second.second;

        std::cout << "Data pointer: " << data_ptr << ", Size: " << size << std::endl;

        // Send allocate command to the server with the new structure: "ALLOCATE GPU"
        std::string command = "ALLOCATE GPU " + name + " " + std::to_string(size);
        std::string response = sendCommandToServer(sock, command);

        if (response.find("ALLOCATED GPU") == 0)
        {
            std::cout << "Allocated GPU memory for tensor: " << name << std::endl;

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
            void *d_tensor = nullptr;
            cudaError_t status = cudaIpcOpenMemHandle(&d_tensor, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            if (status != cudaSuccess)
            {
                std::cerr << "cudaIpcOpenMemHandle failed for tensor: " << name << " with error: " << cudaGetErrorString(status) << std::endl;
                continue;
            }

            std::cout << "Imported GPU memory for tensor: " << name << " at address: " << d_tensor << std::endl;

            // Copy data from host to the imported GPU memory
            status = cudaMemcpy(d_tensor, reinterpret_cast<void *>(data_ptr), size, cudaMemcpyHostToDevice);
            if (status != cudaSuccess)
            {
                std::cerr << "cudaMemcpy failed for tensor: " << name << " with error: " << cudaGetErrorString(status) << std::endl;
                cudaIpcCloseMemHandle(d_tensor);
                continue;
            }

            std::cout << "Copied data to GPU memory for tensor: " << name << std::endl;
            tensor_offsets[name] = reinterpret_cast<uint64_t>(d_tensor);
        }
        else
        {
            std::cerr << "Memory allocation failed for tensor: " << name << " with response: " << response << std::endl;
        }
    }

    close(sock);
    std::cout << "Finished save_tensors_cpp" << std::endl;
    return tensor_offsets;
}

std::map<std::string, uint64_t> saveTensorsToCpu(const std::vector<std::string> &tensor_names,
                                                 const std::map<std::string, std::pair<uint64_t, size_t>> &tensor_data_index)
{
    std::map<std::string, uint64_t> tensor_offsets;
    std::cout << "Starting saveTensorsToCpu" << std::endl;

    int sock = connectToServer();
    if (sock < 0)
    {
        std::cerr << "Failed to connect to the server" << std::endl;
        return tensor_offsets;
    }

    for (const auto &name : tensor_names)
    {
        std::cout << "Processing tensor: " << name << std::endl;

        auto it = tensor_data_index.find(name);
        if (it == tensor_data_index.end())
        {
            std::cerr << "Tensor data not found for: " << name << std::endl;
            continue;
        }

        uint64_t data_ptr = it->second.first;
        size_t size = it->second.second;

        std::cout << "Data pointer: " << data_ptr << ", Size: " << size << std::endl;

        // Send allocate command to the server
        std::string command = "ALLOCATE HOST " + name + " " + std::to_string(size);
        std::string response = sendCommandToServer(sock, command);

        if (response.find("ALLOCATED HOST") == 0)
        {
            std::cout << "Allocated shared host memory for tensor: " << name << std::endl;
            // Open the shared memory using the tensor name
            std::string shm_name = "/" + name;
            int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
            if (shm_fd == -1)
            {
                std::cerr << "shm_open failed for tensor: " << name << " with error: " << strerror(errno) << std::endl;
                continue;
            }

            // Map the shared memory into the process's address space
            void *h_memory = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            if (h_memory == MAP_FAILED)
            {
                std::cerr << "mmap failed for tensor: " << name << " with error: " << strerror(errno) << std::endl;
                close(shm_fd);
                continue;
            }

            // Close the file descriptor after mapping
            close(shm_fd);

            std::cout << "Mapped shared host memory for tensor: " << name << " at address: " << h_memory << std::endl;
            std::cout << "data_ptr address: " << reinterpret_cast<void *>(data_ptr) << std::endl;
            //cudaError_t status = cudaMemcpy(h_memory, reinterpret_cast<void *>(data_ptr), size, cudaMemcpyDeviceToHost);
            std::memcpy(h_memory, reinterpret_cast<void *>(data_ptr), size);


            std::cout << "Copied data to shared host memory for tensor: " << name << std::endl;
            tensor_offsets[name] = reinterpret_cast<uint64_t>(h_memory);
        }
        else
        {
            std::cerr << "Memory allocation failed for tensor: " << name << " with response: " << response << std::endl;
        }
    }

    close(sock);
    std::cout << "Finished save_tensors_cpu_cpp" << std::endl;
    return tensor_offsets;
}

std::map<std::string, uint64_t> saveTensorsToDisk(const std::vector<std::string> &tensor_names,
                                                  const std::map<std::string, std::pair<uint64_t, size_t>> &tensor_data_index)
{
    std::map<std::string, uint64_t> tensor_offsets;
    std::cout << "Starting saveTensorsToDisk" << std::endl;

    // Create the directory for tensor data
    const std::string directory = "tensors_data";
    mkdir(directory.c_str(), 0777);

    for (const auto &name : tensor_names)
    {
        std::cout << "Processing tensor: " << name << std::endl;

        auto it = tensor_data_index.find(name);
        if (it == tensor_data_index.end())
        {
            std::cerr << "Tensor data not found for: " << name << std::endl;
            continue;
        }

        uint64_t data_ptr = it->second.first;
        size_t size = it->second.second;

        std::cout << "Data pointer: " << data_ptr << ", Size: " << size << std::endl;

        // Allocate host memory to copy the tensor data from GPU
        void *host_buffer = malloc(size);
        if (host_buffer == nullptr)
        {
            std::cerr << "Failed to allocate host memory for tensor: " << name << std::endl;
            continue;
        }

        // Copy data from GPU to host
        //cudaError_t status = cudaMemcpy(host_buffer, reinterpret_cast<void *>(data_ptr), size, cudaMemcpyDeviceToHost);
        std::memcpy(host_buffer, reinterpret_cast<void *>(data_ptr), size);


        // Create a filename for the tensor data
        std::string filename = directory + "/" + name + "_data.bin";

        // Write the tensor data to the file
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (!file)
        {
            std::cerr << "Failed to open file for tensor: " << name << std::endl;
            free(host_buffer);
            continue;
        }

        file.write(reinterpret_cast<const char *>(host_buffer), size);
        if (!file)
        {
            std::cerr << "Failed to write data to file for tensor: " << name << std::endl;
            file.close();
            free(host_buffer);
            continue;
        }

        file.close();
        std::cout << "Saved tensor data to file: " << filename << std::endl;

        // Free the host buffer
        free(host_buffer);

        tensor_offsets[name] = data_ptr;
    }
    std::cout << "Finished saveTensorsToFile" << std::endl;
    return tensor_offsets;
}

cudaIpcMemHandle_t load_ipc_handle(const std::string &filename)
{
    cudaIpcMemHandle_t ipc_handle;
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open())
    {
        throw std::runtime_error("Failed to open IPC handle file: " + filename);
    }
    f.read(reinterpret_cast<char *>(&ipc_handle), sizeof(cudaIpcMemHandle_t));
    f.close();
    return ipc_handle;
}

torch::ScalarType dtype_string_to_scalar_type(const std::string &dtype_str)
{
    if (dtype_str == "torch.float32")
    {
        return torch::kFloat32;
    }
    else if (dtype_str == "torch.float64")
    {
        return torch::kFloat64;
    }
    else if (dtype_str == "torch.int32")
    {
        return torch::kInt32;
    }
    else if (dtype_str == "torch.int64")
    {
        return torch::kInt64;
    }
    else
    {
        throw std::runtime_error("Unsupported dtype string: " + dtype_str);
    }
}

torch::Tensor tensor_from_ipc_handle(const cudaIpcMemHandle_t &ipc_handle,
                                     const std::vector<int64_t> &shape,
                                     torch::ScalarType dtype)
{
    // Open the IPC memory handle
    void *dev_ptr = nullptr;
    cudaError_t status = cudaIpcOpenMemHandle(&dev_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    if (status != cudaSuccess)
    {
        throw std::runtime_error("cudaIpcOpenMemHandle failed: " + std::string(cudaGetErrorString(status)));
    }

    // Create a torch::Tensor from the device pointer
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);

    // Create the tensor without copying data
    torch::Tensor tensor = torch::from_blob(dev_ptr, shape, options);

    // Important: Ensure dev_ptr remains valid for the tensor's lifetime
    // Optionally, store dev_ptr and close the handle later

    return tensor;
}

torch::Tensor load_model_tensor(const std::string &filename, const std::vector<int64_t> &shape,  torch::ScalarType dtype)
{
    cudaIpcMemHandle_t ipc_handle;
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open())
    {
        throw std::runtime_error("Failed to open IPC handle file: " + filename);
    }
    f.read(reinterpret_cast<char *>(&ipc_handle), sizeof(cudaIpcMemHandle_t));
    f.close();
    torch::Tensor tensor = tensor_from_ipc_handle(ipc_handle, shape, dtype);

    return tensor;
}

py::dict load_model_from_ipc(const std::string &tensor_index_file,
                             const std::string &ipc_handles_dir)
{
    // Load tensor metadata from JSON
    std::ifstream f(tensor_index_file);
    if (!f.is_open())
    {
        throw std::runtime_error("Failed to open tensor index file: " + tensor_index_file);
    }
    json tensor_index;
    f >> tensor_index;
    f.close();

    py::dict state_dict;

    for (auto &[name, meta] : tensor_index.items())
    {
        try
        {
            std::string tensor_name = name;
            std::string ipc_handle_file = ipc_handles_dir + "/" + tensor_name + "_ipc_handle.bin";

            // Extract metadata
            if (meta.size() < 5)
            {
                throw std::runtime_error("Incomplete metadata for tensor: " + tensor_name);
            }

            std::vector<int64_t> shape = meta[2].get<std::vector<int64_t>>();
            std::string dtype_str = meta[4].get<std::string>();
            torch::ScalarType dtype = dtype_string_to_scalar_type(dtype_str);

            // Load IPC handle
            cudaIpcMemHandle_t ipc_handle = load_ipc_handle(ipc_handle_file);

            // Reconstruct tensor
            torch::Tensor tensor = tensor_from_ipc_handle(ipc_handle, shape, dtype);

            state_dict[py::str(name)] = tensor;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error processing tensor " << name << ": " << e.what() << std::endl;
            throw;
        }
    }

    return state_dict;
}

PYBIND11_MODULE(cuda_saver, m)
{

    m.def("save_tensors_to_gpu", &saveTensorsToGpu, "Save tensors to GPU memory");
    m.def("save_tensors_to_cpu", &saveTensorsToCpu, "Save tensors to CPU memory");
    m.def("save_tensors_to_disk", &saveTensorsToDisk, "Save tensors to Disk memory");
    m.def("load_model_tensor", &load_model_tensor, " Create a torch::Tensor from the device pointer");
    m.def("load_model_from_ipc", &load_model_from_ipc, "Load model state dict from IPC handles",
          py::arg("tensor_index_file"),
          py::arg("ipc_handles_dir"));
}