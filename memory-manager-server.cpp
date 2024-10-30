#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h> // for close()
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <ctime>
#include <unordered_map>
#include <cuda.h>
#include <thread>     // For std::this_thread::sleep_for
#include <chrono>     // For std::chrono::milliseconds
#include <sys/mman.h> // For mmap
#include <fcntl.h>    // For O_CREAT, O_RDWR
#include <sys/stat.h> // For ftruncate
#include <sstream>    // For std::istringstream
#include <errno.h>    // For errno

// ==================== HostMemoryManager Class ====================

class HostMemoryManager
{
public:
    HostMemoryManager()
    {
        log("Host Memory Manager initialized.");
    }

    ~HostMemoryManager()
    {
        log("Host Memory Manager destroyed.");
    }

    // Allocate shared memory and return the pointer
    void *allocateMemory(const std::string &name, size_t size)
    {
        std::string shm_name = "/" + name; // Shared memory name starting with '/'

        // Create a shared memory object
        int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1)
        {
            log("shm_open failed for " + shm_name + " with errno: " + std::to_string(errno));
            return nullptr;
        }

        // Set the size of the shared memory object
        if (ftruncate(shm_fd, size) == -1)
        {
            log("ftruncate failed for " + shm_name + " with errno: " + std::to_string(errno));
            close(shm_fd);
            shm_unlink(shm_name.c_str());
            return nullptr;
        }

        // Map the shared memory object into the process's address space
        void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (ptr == MAP_FAILED)
        {
            log("mmap failed for " + shm_name + " with errno: " + std::to_string(errno));
            close(shm_fd);
            shm_unlink(shm_name.c_str());
            return nullptr;
        }

        close(shm_fd); // Close the file descriptor after mapping

        log("Allocated " + std::to_string(size) + " bytes for " + shm_name + " in shared memory");
        return ptr;
    }

    // Free the shared memory
    bool freeMemory(void *ptr, const std::string &name, size_t size)
    {
        if (ptr != nullptr)
        {
            if (munmap(ptr, size) == -1)
            {
                log("munmap failed for " + name + " with errno: " + std::to_string(errno));
                return false;
            }
            std::string shm_name = "/" + name;
            if (shm_unlink(shm_name.c_str()) == -1)
            {
                log("shm_unlink failed for " + shm_name + " with errno: " + std::to_string(errno));
                return false;
            }
            log("Freed shared memory for " + shm_name);
            return true;
        }
        log("Shared memory for " + name + " not allocated.");
        return false;
    }

    static void log(const std::string &message)
    {
        std::ofstream log_file;
        log_file.open("host_memory_manager.log", std::ios_base::app);
        std::time_t now = std::time(nullptr);
        log_file << std::ctime(&now) << ": " << message << std::endl;
        log_file.close();
    }
};

// ==================== GpuMemoryManager Class ====================

class GpuMemoryManager
{
public:
    GpuMemoryManager()
    {
        cudaSetDevice(0);
        log("GPU Memory Manager initialized.");
    }

    ~GpuMemoryManager()
    {
        log("GPU Memory Manager destroyed.");
    }

    void *allocateMemory(const std::string &name, size_t size, cudaIpcMemHandle_t &ipc_handle)
    {
        void *d_tensor;
        cudaError_t status = cudaMalloc(&d_tensor, size);
        if (status != cudaSuccess)
        {
            log("cudaMalloc failed for tensor " + name + " with error: " + std::string(cudaGetErrorString(status)));
            return nullptr;
        }

        // Create an IPC handle for the allocated memory
        status = cudaIpcGetMemHandle(&ipc_handle, d_tensor);
        if (status != cudaSuccess)
        {
            log("cudaIpcGetMemHandle failed for tensor " + name + " with error: " + std::string(cudaGetErrorString(status)));
            cudaFree(d_tensor);
            return nullptr;
        }

        log("Allocated " + std::to_string(size) + " bytes for tensor " + name);
        return d_tensor;
    }

    void *importMemory(cudaIpcMemHandle_t &ipc_handle)
    {
        void *d_tensor;
        cudaError_t status = cudaIpcOpenMemHandle(&d_tensor, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
        if (status != cudaSuccess)
        {
            log("cudaIpcOpenMemHandle failed with error: " + std::string(cudaGetErrorString(status)));
            return nullptr;
        }
        return d_tensor;
    }

    bool freeMemory(void *d_tensor, const std::string &name)
    {
        if (d_tensor != nullptr)
        {
            cudaError_t status = cudaFree(d_tensor);
            if (status != cudaSuccess)
            {
                log("cudaFree failed for tensor " + name + " with error: " + std::string(cudaGetErrorString(status)));
                return false;
            }
            log("Freed memory for tensor " + name);
            return true;
        }
        log("Memory for tensor " + name + " not allocated.");
        return false;
    }

    static void log(const std::string &message)
    {
        std::ofstream log_file;
        log_file.open("cuda_daemon.log", std::ios_base::app);
        std::time_t now = std::time(nullptr);
        log_file << std::ctime(&now) << ": " << message << std::endl;
        log_file.close();
    }
};




GpuMemoryManager gpuManager;
HostMemoryManager hostManager;

// ==================== Handle Client Function ====================

void handleClient(int client_socket)
{
    char buffer[1024] = {0};
    int read_size;
    void *memory = nullptr;
    cudaIpcMemHandle_t ipc_handle;
   
    while ((read_size = read(client_socket, buffer, 1024)) > 0)
    {
        std::string command(buffer, read_size);

        GpuMemoryManager::log("Received command: " + command);


        std::istringstream iss(command);
        std::string cmd_type, mem_type, name;
        size_t size = 0;

        iss >> cmd_type >> mem_type >> name >> size;

        if (cmd_type == "ALLOCATE" && mem_type == "GPU")
        {
            memory = gpuManager.allocateMemory(name, size, ipc_handle);
            if (memory)
            {
                GpuMemoryManager::log("Allocated GPU memory at pointer: " + std::to_string(reinterpret_cast<uint64_t>(memory)));
                std::string response = "ALLOCATED GPU " + name + "\n";
                send(client_socket, response.c_str(), response.size(), 0);
                std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 500 ms delay

                send(client_socket, &ipc_handle, sizeof(cudaIpcMemHandle_t), 0);
                GpuMemoryManager::log("Sent IPC handle for GPU tensor " + name);
            }
            else
            {
                std::string response = "FAILED GPU\n";
                send(client_socket, response.c_str(), response.size(), 0);
                GpuMemoryManager::log("GPU memory allocation failed for tensor " + name);
            }
        }
        else if (cmd_type == "ALLOCATE" && mem_type == "HOST")
        {
            memory = hostManager.allocateMemory(name, size);
            if (memory)
            {
                HostMemoryManager::log("Allocated shared host memory for: " + name);
                std::string response = "ALLOCATED HOST " + name + "\n";
                send(client_socket, response.c_str(), response.size(), 0);
                HostMemoryManager::log("Sent allocation confirmation for host memory " + name);
            }
            else
            {
                std::string response = "FAILED HOST\n";
                send(client_socket, response.c_str(), response.size(), 0);
                HostMemoryManager::log("Host memory allocation failed for " + name);
            }
        }
        else if (cmd_type == "IMPORT" && mem_type == "GPU")
        {
            read(client_socket, &ipc_handle, sizeof(cudaIpcMemHandle_t));
            memory = gpuManager.importMemory(ipc_handle);
            if (memory)
            {
                std::string response = "IMPORTED GPU\n";
                send(client_socket, response.c_str(), response.size(), 0);
                GpuMemoryManager::log("Imported GPU memory for tensor with IPC handle.");
            }
            else
            {
                std::string response = "IMPORT FAILED GPU\n";
                send(client_socket, response.c_str(), response.size(), 0);
                GpuMemoryManager::log("GPU memory import failed.");
            }
        }
        else if (cmd_type == "FREE" && mem_type == "GPU")
        {
            bool success = gpuManager.freeMemory(memory, name);
            std::string response = success ? "FREED GPU\n" : "NOT_ALLOCATED GPU\n";
            send(client_socket, response.c_str(), response.size(), 0);
            GpuMemoryManager::log("GPU memory free request for tensor " + name + " resulted in: " + response);
        }
        else
        {
            std::string response = "UNKNOWN COMMAND\n";
            send(client_socket, response.c_str(), response.size(), 0);
            GpuMemoryManager::log("Received unknown command: " + command);
        }

        // Clear the buffer for the next command
        memset(buffer, 0, sizeof(buffer));
    }

    if (read_size == 0)
    {
        GpuMemoryManager::log("Client disconnected.");
    }
    else if (read_size == -1)
    {
        GpuMemoryManager::log("Read error occurred.");
    }

    close(client_socket);
}
// ==================== Main Function  ====================

int main(int argc, char const *argv[])
{
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Create socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        GpuMemoryManager::log("Socket creation failed.");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        GpuMemoryManager::log("Setsockopt failed.");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    // Bind the socket to the network address and port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        GpuMemoryManager::log("Bind failed.");
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        GpuMemoryManager::log("Listen failed.");
        exit(EXIT_FAILURE);
    }

    GpuMemoryManager::log("Daemon started, listening for connections...");

    while (true)
    {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0)
        {
            perror("accept");
            GpuMemoryManager::log("Accept failed.");
            exit(EXIT_FAILURE);
        }

        handleClient(new_socket);
    }

    return 0;
}
