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
#include <fcntl.h>    // for O_CREAT, O_RDWR
#include <sys/mman.h> // for mmap, shm_open
#include <sys/stat.h> // for ftruncate
#include <cstring>    // for memcpy

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

// Function to send the command to the daemon and receive the response
std::string sendCommandToDaemonCpu(int sock, const std::string &command)
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

void saveSharedMemoryHandleToFile(const std::string &filename, int shm_fd)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char *>(&shm_fd), sizeof(int));
    if (!file)
    {
        std::cerr << "Failed to write shared memory file descriptor to file: " << filename << std::endl;
    }

    file.close();
}

bool receiveSharedMemoryHandle(int sock, int &shm_fd)
{
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char buf[1024]; // Buffer to receive any actual data sent alongside the control message
    char cmsg_buf[CMSG_SPACE(sizeof(int))];
    int received_fd = -1;

    // Prepare the iovec
    struct iovec io = {.iov_base = buf, .iov_len = sizeof(buf)};
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;

    // Set up the control message buffer
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);

    // Receive the message
    ssize_t recv_len = recvmsg(sock, &msg, 0);
    if (recv_len == -1)
    {
        perror("recvmsg");
        return false; // Handle error appropriately
    }

    std::cout << "recvmsg returned " << recv_len << " bytes." << std::endl;
    std::cout << "msg.msg_controllen: " << msg.msg_controllen << std::endl;
    // Get the first control message header
    cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg)
    {
        std::cout << "cmsg_level: " << cmsg->cmsg_level << ", cmsg_type: " << cmsg->cmsg_type << std::endl;
    }
    else
    {
        std::cout << "cmsg is NULL. No control message received." << std::endl;
    }

    if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
    {
        memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(received_fd));
        shm_fd = received_fd;
        std::cout << "Received file descriptor: " << received_fd << std::endl;
        return true;
    }
    else
    {
        std::cout << "Failed to receive control message or no file descriptor passed." << std::endl;
        return false;
    }
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

        auto it = tensor_data_index.find(name);
        if (it == tensor_data_index.end())
        {
            std::cerr << "Tensor data not found for: " << name << std::endl;
            continue;
        }

        uint64_t data_ptr = it->second.first;
        size_t size = it->second.second;

        std::cout << "Data pointer: " << data_ptr << ", Size: " << size << std::endl;

        // Send allocate command to the daemon with the new structure: "ALLOCATE GPU"
        std::string command = "ALLOCATE GPU " + name + " " + std::to_string(size);
        std::string response = sendCommandToDaemon(sock, command);

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

std::map<std::string, uint64_t> save_tensors_cpu_cpp(const std::vector<std::string> &tensor_names,
                                                     const std::map<std::string, std::pair<uint64_t, size_t>> &tensor_data_index)
{
    std::map<std::string, uint64_t> tensor_offsets;
    std::cout << "Starting save_tensors_cpu_cpp" << std::endl;

    int sock = connectToDaemon();
    if (sock < 0)
    {
        std::cerr << "Failed to connect to the daemon" << std::endl;
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

        // Send allocate command to the daemon with the new structure: "ALLOCATE HOST"
        std::string command = "ALLOCATE HOST " + name + " " + std::to_string(size);
        std::string response = sendCommandToDaemonCpu(sock, command);

        if (response.find("ALLOCATED HOST") == 0)
        {
            std::cout << "Allocated shared host memory for tensor: " << name << std::endl;

            int shm_fd;
            // Receive the shared memory file descriptor from the daemon
            if (receiveSharedMemoryHandle(sock, shm_fd))
            {
                std::cout << "Received shared memory file descriptor: " << shm_fd << std::endl;
            }
            else
            {
                std::cerr << "Failed to receive shared memory file descriptor for tensor: " << name << std::endl;
            }
            // Save the shared memory handle to a file
            std::string filename = name + "_shm_handle.bin";
            saveSharedMemoryHandleToFile(filename, shm_fd);
            if (shm_fd < 0)
            {
                std::cerr << "Received an invalid file descriptor for tensor: " << name << std::endl;
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

            std::cout << "Mapped shared host memory for tensor: " << name << " at address: " << h_memory << std::endl;

            // Copy data from the provided host pointer to the shared memory
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

PYBIND11_MODULE(cuda_saver, m)
{
    m.def("load_tensor_from_gpu", &load_tensor_from_gpu, "Load a tensor from a GPU memory pointer");

    m.def("save_tensors_cpp", &save_tensors_cpp, "Save tensors to GPU memory");
    m.def("save_tensors_cpu_cpp", &save_tensors_cpu_cpp, "Save tensors to CPU memory");
}
