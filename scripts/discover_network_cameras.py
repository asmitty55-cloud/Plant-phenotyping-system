import ipaddress
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pt.device.network_camera_discovery import (
    CAMERA_PORTS,
    MAX_ADDRESSES_PER_NETWORK,
    MAX_TOTAL_ADDRESSES,
    is_rtsp_server,
    local_ipv4_networks,
    read_neighbor_macs,
    scan_host,
)


def parse_networks(arguments):
    if not arguments:
        return local_ipv4_networks()
    networks = []
    for argument in arguments:
        try:
            networks.append(ipaddress.ip_network(argument, strict=False))
        except ValueError:
            print(f"Skipping invalid network/IP: {argument}")
    return networks


def main():
    networks = parse_networks(sys.argv[1:])
    if not networks:
        print("No private IPv4 networks found.")
        return 1

    hosts = []
    for network in networks:
        if network.num_addresses > MAX_ADDRESSES_PER_NETWORK:
            print(f"Skipping oversized network: {network}")
            continue
        hosts.extend([str(network.network_address)] if network.num_addresses == 1 else map(str, network.hosts()))
    hosts = list(dict.fromkeys(hosts))
    if not hosts:
        print("No scan targets within the configured address limit.")
        return 1
    if len(hosts) > MAX_TOTAL_ADDRESSES:
        print(f"Refusing to scan more than {MAX_TOTAL_ADDRESSES} addresses at once.")
        return 1

    print(f"Scanning {len(hosts)} addresses for camera ports {CAMERA_PORTS}...")
    results = []
    with ThreadPoolExecutor(max_workers=min(64, len(hosts))) as pool:
        futures = [pool.submit(scan_host, host) for host in hosts]
        for future in as_completed(futures):
            result = future.result()
            if result["open_ports"]:
                results.append(result)

    neighbors = read_neighbor_macs()
    for result in sorted(results, key=lambda item: ipaddress.ip_address(item["host"])):
        host = result["host"]
        mac_address = neighbors.get(host, "unknown MAC")
        rtsp_ports = [
            port for port in (554, 8554)
            if port in result["open_ports"] and is_rtsp_server(host, port)
        ]
        print(f"{host}: open {', '.join(map(str, result['open_ports']))}; MAC {mac_address}; RTSP {rtsp_ports or 'not confirmed'}")

    print("\nESCAM/Yoosee examples (use your camera's local credentials):")
    print("  rtsp://<user>:<password>@<ip>:554/onvif1")
    print("  rtsp://<user>:<password>@<ip>:554/onvif2")
    print("A responding RTSP port does not confirm credentials, path, or frame decoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
