import socket
import torch

def get_gpu_count():
    gpu_count = torch.cuda.device_count()
    return (gpu_count)

def get_node_ip():
    try:
        # Get the hostname
        hostname = socket.gethostname()
        # Get the IP address
        ip_address = socket.gethostbyname(hostname)
        gpu_count = get_gpu_count()
        #print(f"Node IP Address: {ip_address} ")
        #print(f"Number of GPUs: {gpu_count}")
        print(f"Node IP Address: {ip_address}, Number of GPUs: {gpu_count}")

        # List the names of the GPUs
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i + 1}: {gpu_name}")

    except socket.error as e:
        print(f"Unable to get IP Address: {e}")

if __name__ == "__main__":
    get_node_ip()
