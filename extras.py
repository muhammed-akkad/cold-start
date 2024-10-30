""" 
def save_tensors_py(tensor_names, tensor_data_index):
    tensor_offsets = {}

    print("Starting save_tensors_py")

    sock = connect_to_daemon()
    if not sock:
        print("Failed to connect to the daemon")
        return tensor_offsets

    for name in tensor_names:
        print(f"Processing tensor: {name}")

        data_ptr = tensor_data_index[name][0]
        size = tensor_data_index[name][1]

        print(f"Data pointer: {data_ptr}, Size: {size}")
        print("running_var")
        if name == "layer2.0.bn1.running_var":
            print("running_var")
            print(tensor_data_index[name])
        # Send allocate command to the daemon
        command = f"ALLOCATE {name} {size}"
        response = send_command_to_daemon(sock, command)

        if response and response.startswith("ALLOCATED"):
            print(f"Received response: {response}")

            # Prepare the COPY command
            copy_command = f"COPY {name} {data_ptr} {size}"
            copy_response = send_command_to_daemon(sock, copy_command)

            if copy_response and copy_response.startswith("COPIED"):
                print(f"Copied data to GPU memory for tensor: {name}")

                # Assuming daemon returns the GPU pointer as part of the response
                gpu_pointer = int(copy_response.split()[-1])
                tensor_offsets[name] = gpu_pointer
            else:
                print(f"Copy failed for tensor: {name}")
                send_command_to_daemon(sock, f"FREE {name}")
        else:
            print(f"Memory allocation failed for tensor: {name}")

    sock.close()
    print("Finished save_tensors_py")
    return tensor_offsets """
    
    
    def connect_to_daemon():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 8080))
        return sock
    except socket.error as e:
        print(f"Failed to connect to the daemon: {e}")
        return None

def send_command_to_daemon(sock, command):
    try:
        sock.sendall(command.encode('utf-8'))
        response = sock.recv(1024).decode('utf-8')
        return response
    except socket.error as e:
        print(f"Failed to send command to daemon: {e}")
        return None
    
    start_time_ipc = time.time()
    model_class = models.mobilenet_v3_large  
    model = load_model_from_ipc(model_class, 'tensor_index.json', 'handlers_gpu')
    #torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    end_time_ipc = time.time()
    # Now you can use the model
    model.eval()
    start_time_normal = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Example usage:
    normal_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).to(device)
    end_time_normal = time.time()
    normal_load_time = end_time_normal - start_time_normal
    ipc_load_time = end_time_ipc - start_time_ipc

    normal_model.eval()
    print(f"Model loaded normally in {normal_load_time:.6f} seconds")
    print(f"Model loaded using IPC in {ipc_load_time:.6f} seconds")