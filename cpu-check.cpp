#include <iostream>
#include <fcntl.h>    // For shm_open
#include <sys/mman.h> // For mmap
#include <unistd.h>
#include <cstring>
#include <errno.h>

int main()
{
    std::string name = "layer2.0.bn1.running_var";
    size_t size = 512; // Ensure this matches the size used during allocation

    // Open the shared memory object
    std::string shm_name = "/" + name;
    int shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (shm_fd == -1)
    {
        std::cerr << "shm_open failed for tensor: " << name << " with error: " << strerror(errno) << std::endl;
        return -1;
    }

    // Map the shared memory into the process's address space
    void* h_memory = mmap(0, size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (h_memory == MAP_FAILED)
    {
        std::cerr << "mmap failed for tensor: " << name << " with error: " << strerror(errno) << std::endl;
        close(shm_fd);
        return -1;
    }

    // Close the file descriptor after mapping
    close(shm_fd);

    // Read data from shared memory
    float* shared_data = static_cast<float*>(h_memory);
    size_t num_elements = size / sizeof(float);

    // Print the first few elements to verify
    std::cout << "Data in shared memory for tensor: " << name << std::endl;
    for (size_t i = 0; i < std::min(num_elements, size_t(10)); ++i)
    {
        std::cout << "Element " << i << ": " << shared_data[i] << std::endl;
    }

    // Unmap the shared memory when done
    munmap(h_memory, size);

    return 0;
}
