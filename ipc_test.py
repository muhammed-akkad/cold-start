import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# The IPC handle as a raw byte string (received from the daemon)
# Example: ipc_handle_bytes = <received from daemon>
ipc_handle_bytes = b'0x7f83415f5000'  # This should be the actual IPC handle received from the daemon

# Convert the byte string to a CUDA IPC handle
ipc_handle = cuda.IpcMemHandle(ipc_handle_bytes)

# Import the IPC memory (assuming the data type is float and size is known)
size_in_bytes = 37632 * 4  # Example: 1024 floats
d_tensor = ipc_handle.open(size_in_bytes)

# Now, d_tensor is a device pointer to the memory shared via IPC
# You can work with this memory using PyCUDA or transfer it to host memory

# For example, to copy this data to the host and print it:
h_tensor = np.empty(1024, dtype=np.float32)  # Assuming the memory holds 1024 floats
cuda.memcpy_dtoh(h_tensor, d_tensor)

print("Data in shared memory:", h_tensor)

# When done, close the IPC memory handle
ipc_handle.close()