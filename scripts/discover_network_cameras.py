import ipaddress
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


PORTS = (80, 554, 5000, 8080, 8899)


def local_ipv4_networks():
    output = subprocess.run(["ipconfig"], capture_output=True, text=True).stdout
    addresses = []
    masks = []
    for line in output.splitlines():
        text = line.strip()
        if text.startswith("IPv4 Address"):
            addresses.append(text.split(":")[-1].strip())
        if text.startswith("Subnet Mask"):
            masks.append(text.split(":")[-1].strip())
    for address, mask in zip(addresses, masks):
        try:
            yield ipaddress.ip_network(f"{address}/{mask}", strict=False)
        except ValueError:
            continue


def port_open(host, port, timeout=0.35):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((str(host), port)) == 0


def scan_host(host):
    open_ports = [port for port in PORTS if port_open(host, port)]
    return str(host), open_ports


def main():
    if len(sys.argv) > 1:
        networks = []
        for arg in sys.argv[1:]:
            try:
                if "/" in arg:
                    networks.append(ipaddress.ip_network(arg, strict=False))
                else:
                    networks.append(ipaddress.ip_network(f"{arg}/32", strict=False))
            except ValueError:
                print(f"Skipping invalid network/IP: {arg}")
    else:
        networks = list(local_ipv4_networks())
    if not networks:
        print("No IPv4 networks found.")
        return 1

    hosts = []
    for network in networks:
        if network.num_addresses == 1:
            hosts.append(network.network_address)
        elif network.num_addresses <= 512:
            hosts.extend(network.hosts())

    print(f"Scanning {len(hosts)} LAN addresses for camera-ish ports {PORTS}...")
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(scan_host, host) for host in hosts]
        for future in as_completed(futures):
            host, open_ports = future.result()
            if open_ports:
                print(f"{host}: open {', '.join(map(str, open_ports))}")

    print("\nTry Yoosee RTSP URLs with VLC/OpenCV:")
    print("  rtsp://admin:<nvr-password>@<ip>:554/onvif1")
    print("  rtsp://admin:<nvr-password>@<ip>:554/onvif2")
    print("Some Yoosee cameras expose NVR/ONVIF on port 5000 or require enabling NVR Connection in the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
