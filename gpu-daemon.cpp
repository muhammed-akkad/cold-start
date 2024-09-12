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

bool sendSharedMemoryHandle(int client_socket, int shm_fd)
{
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char buf[CMSG_SPACE(sizeof(int))];
    memset(buf, '\0', sizeof(buf));

    struct iovec io = {.iov_base = (void *)"FD", .iov_len = 2};

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));

    memcpy(CMSG_DATA(cmsg), &shm_fd, sizeof(int));

    if (sendmsg(client_socket, &msg, 0) == -1)
    {
        std::cerr << "Failed to send file descriptor: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}
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

    void *allocateMemory(const std::string &name, size_t size, int &shm_fd)
    {
        // Create shared memory object
        shm_fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1)
        {
            log("shm_open failed for memory " + name);
            return nullptr;
        }

        // Set the size of the shared memory object
        if (ftruncate(shm_fd, size) == -1)
        {
            log("ftruncate failed for memory " + name);
            close(shm_fd);
            shm_unlink(name.c_str());
            return nullptr;
        }

        // Map the shared memory object to the process's address space
        void *h_memory = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (h_memory == MAP_FAILED)
        {
            log("mmap failed for memory " + name);
            close(shm_fd);
            shm_unlink(name.c_str());
            return nullptr;
        }

        log("Allocated " + std::to_string(size) + " bytes of shared host memory for " + name);
        return h_memory;
    }

    void *importMemory(const std::string &name, size_t size, int &shm_fd)
    {
        // Open the shared memory object
        shm_fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (shm_fd == -1)
        {
            log("shm_open failed for memory " + name);
            return nullptr;
        }

        // Map the shared memory object to the process's address space
        void *h_memory = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (h_memory == MAP_FAILED)
        {
            log("mmap failed for memory " + name);
            close(shm_fd);
            return nullptr;
        }

        log("Imported shared host memory for " + name);
        return h_memory;
    }

    bool freeMemory(void *h_memory, const std::string &name, size_t size, int shm_fd)
    {
        if (h_memory != nullptr)
        {
            // Unmap the shared memory
            munmap(h_memory, size);
            // Close the shared memory file descriptor
            close(shm_fd);
            // Unlink the shared memory object
            shm_unlink(name.c_str());

            log("Freed shared host memory for " + name);
            return true;
        }
        log("Shared host memory for " + name + " not allocated.");
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

void handleClient(int client_socket)
{
    char buffer[1024] = {0};
    int read_size;
    void *memory = nullptr;
    cudaIpcMemHandle_t ipc_handle; // To store the IPC handle
    int shm_fd = -1;               // Shared memory file descriptor

    while ((read_size = read(client_socket, buffer, 1024)) > 0)
    {
        std::string command(buffer, read_size);

        GpuMemoryManager::log("Received command: " + command);

        // Split the command into parts
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
            memory = hostManager.allocateMemory(name, size, shm_fd);
            if (memory)
            {
                HostMemoryManager::log("Allocated shared host memory at pointer: " + std::to_string(reinterpret_cast<uint64_t>(memory)));
                std::string response = "ALLOCATED HOST " + name + "\n";
                send(client_socket, response.c_str(), response.size(), 0);
                // Prepare to send the shared memory file descriptor to the client
                std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 500 ms delay

                struct msghdr msg = {0};
                struct cmsghdr *cmsg;
                char cmsg_buf[CMSG_SPACE(sizeof(int))];
                char dummy_buf[1] = {' '}; // You need to send at least 1 byte of real data

                // Prepare the iovec
                struct iovec io = {.iov_base = dummy_buf, .iov_len = sizeof(dummy_buf)};
                msg.msg_iov = &io;
                msg.msg_iovlen = 1;

                // Set up the control message buffer
                msg.msg_control = cmsg_buf;
                msg.msg_controllen = sizeof(cmsg_buf);

                // Get the first control message header
                cmsg = CMSG_FIRSTHDR(&msg);
                cmsg->cmsg_level = SOL_SOCKET;
                cmsg->cmsg_type = SCM_RIGHTS;
                cmsg->cmsg_len = CMSG_LEN(sizeof(int));

                // Copy the file descriptor to the control message
                // Step 2: Print the dummy buffer, file descriptor, and control message details
                HostMemoryManager::log("Sending dummy buffer: " + std::string(dummy_buf, sizeof(dummy_buf)));
                HostMemoryManager::log("File descriptor to be sent: " + std::to_string(shm_fd));
                HostMemoryManager::log("Control message length: " + std::to_string(cmsg->cmsg_len));
                HostMemoryManager::log("Control message level: " + std::to_string(cmsg->cmsg_level));
                HostMemoryManager::log("Control message type: " + std::to_string(cmsg->cmsg_type));

                // Send the message with the file descriptor
                if (sendmsg(client_socket, &msg, 0) == -1)
                {
                    perror("sendmsg");
                    HostMemoryManager::log("Failed to send file descriptor.");
                }
                else
                {
                    HostMemoryManager::log("Sent shared memory descriptor for host memory " + name);
                }
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
        else if (cmd_type == "IMPORT" && mem_type == "HOST")
        {
            read(client_socket, &shm_fd, sizeof(shm_fd));
            memory = hostManager.importMemory(name, size, shm_fd);
            if (memory)
            {
                std::string response = "IMPORTED HOST\n";
                send(client_socket, response.c_str(), response.size(), 0);
                HostMemoryManager::log("Imported shared host memory for " + name);
            }
            else
            {
                std::string response = "IMPORT FAILED HOST\n";
                send(client_socket, response.c_str(), response.size(), 0);
                HostMemoryManager::log("Host memory import failed.");
            }
        }
        else if (cmd_type == "FREE" && mem_type == "GPU")
        {
            bool success = gpuManager.freeMemory(memory, name);
            std::string response = success ? "FREED GPU\n" : "NOT_ALLOCATED GPU\n";
            send(client_socket, response.c_str(), response.size(), 0);
            GpuMemoryManager::log("GPU memory free request for tensor " + name + " resulted in: " + response);
        }
        else if (cmd_type == "FREE" && mem_type == "HOST")
        {
            bool success = hostManager.freeMemory(memory, name, size, shm_fd);
            std::string response = success ? "FREED HOST\n" : "NOT_ALLOCATED HOST\n";
            send(client_socket, response.c_str(), response.size(), 0);
            HostMemoryManager::log("Host memory free request for " + name + " resulted in: " + response);
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
