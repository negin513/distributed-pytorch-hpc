import socket

def get_node_ip():
    try:
        # Get the hostname
        hostname = socket.gethostname()
        # Get the IP address
        ip_address = socket.gethostbyname(hostname)
        print(f"Node IP Address: {ip_address} ")

    except socket.error as e:
        print(f"Unable to get IP Address: {e}")

if __name__ == "__main__":
    get_node_ip()
