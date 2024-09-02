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
#include <thread> // For std::this_thread::sleep_for
#include <chrono> // For std::chrono::milliseconds

class GpuMemoryManager
{
public:
    GpuMemoryManager()
    {
        cudaSetDevice(0); // Optionally set the CUDA device
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
        cudaIpcGetMemHandle(&ipc_handle, d_tensor);

        log("Allocated " + std::to_string(size) + " bytes for tensor " + name);
        return d_tensor;
    }

    void *importMemory(cudaIpcMemHandle_t &ipc_handle)
    {
        void *d_tensor;
        cudaIpcOpenMemHandle(&d_tensor, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
        return d_tensor;
    }

    bool freeMemory(void *d_tensor, const std::string &name)
    {
        if (d_tensor != nullptr)
        {
            cudaFree(d_tensor);
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

void handleClient(int client_socket)
{
    char buffer[1024] = {0};
    int read_size;
    void *d_tensor = nullptr;
    cudaIpcMemHandle_t ipc_handle; // To store the IPC handle

    while ((read_size = read(client_socket, buffer, 1024)) > 0)
    {
        std::string command(buffer, read_size);

        GpuMemoryManager::log("Received command: " + command);

        if (command.find("ALLOCATE") == 0)
        {
            std::string name = command.substr(9, command.find(" ") - 9);
            size_t size = std::stoul(command.substr(command.find_last_of(" ") + 1));

            d_tensor = gpuManager.allocateMemory(name, size, ipc_handle);
            if (d_tensor)
            {
                GpuMemoryManager::log("Allocated memory at pointer: " + std::to_string(reinterpret_cast<uint64_t>(d_tensor)));
                // Send the "ALLOCATED" message to the client
                std::string response = "ALLOCATED " + name + "\n";
                send(client_socket, response.c_str(), response.size(), 0);
                // Add a delay before sending the IPC handle
                std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 500 ms delay

                // Send the IPC handle back to the client
                send(client_socket, &ipc_handle, sizeof(cudaIpcMemHandle_t), 0);

                GpuMemoryManager::log("Sent IPC handle for tensor " + name);
            }
            else
            {
                std::string response = "FAILED\n";
                send(client_socket, response.c_str(), response.size(), 0);
                GpuMemoryManager::log("Memory allocation failed for tensor " + name);
            }
        }
        else if (command.find("IMPORT") == 0)
        {
            // Assume the IPC handle is passed by the client
            read(client_socket, &ipc_handle, sizeof(cudaIpcMemHandle_t));
            d_tensor = gpuManager.importMemory(ipc_handle);
            if (d_tensor)
            {
                std::string response = "IMPORTED\n";
                send(client_socket, response.c_str(), response.size(), 0);
                GpuMemoryManager::log("Imported memory for tensor with IPC handle.");
            }
            else
            {
                std::string response = "IMPORT_FAILED\n";
                send(client_socket, response.c_str(), response.size(), 0);
                GpuMemoryManager::log("Memory import failed.");
            }
        }
        else if (command.find("FREE") == 0)
        {
            std::string name = command.substr(5);
            bool success = gpuManager.freeMemory(d_tensor, name);
            std::string response = success ? "FREED\n" : "NOT_ALLOCATED\n";
            send(client_socket, response.c_str(), response.size(), 0);
            GpuMemoryManager::log("Memory free request for tensor " + name + " resulted in: " + response);
            d_tensor = nullptr;
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

        // Handle client requests
        handleClient(new_socket);
    }

    return 0;
}
