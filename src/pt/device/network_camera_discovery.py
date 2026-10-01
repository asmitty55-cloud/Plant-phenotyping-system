import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


CAMERA_PORTS = (554, 8554, 80, 5000, 8080, 8899)
RTSP_PORTS = (554, 8554)
MAX_ADDRESSES_PER_NETWORK = 512
MAX_TOTAL_ADDRESSES = 1024
MAX_WORKERS = 64
DEFAULT_TIMEOUT = 0.2


def normalize_mac(value):
    raw = str(value or "").strip()
    if not re.fullmatch(r"(?:[0-9a-fA-F]{12}|(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2})", raw):
        raise ValueError("Enter a valid 6-byte MAC address.")
    compact = re.sub(r"[:-]", "", raw)
    compact = compact.lower()
    return ":".join(compact[index:index + 2] for index in range(0, 12, 2))


def _usable_network(address, prefix_length):
    try:
        interface = ipaddress.ip_interface(f"{address}/{int(prefix_length)}")
    except (TypeError, ValueError):
        return None
    network = interface.network
    if (
        not interface.ip.is_private
        or interface.ip.is_loopback
        or interface.ip.is_link_local
        or network.num_addresses > MAX_ADDRESSES_PER_NETWORK
    ):
        return None
    return network


def _windows_ipv4_networks():
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return []
    command = (
        "Get-NetIPAddress -AddressFamily IPv4 | "
        "Select-Object IPAddress, PrefixLength | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=12,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        records = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    if isinstance(records, dict):
        records = [records]
    networks = []
    for record in records:
        network = _usable_network(record.get("IPAddress"), record.get("PrefixLength"))
        if network is not None:
            networks.append(network)
    return networks


def _ip_command_networks():
    ip_command = shutil.which("ip")
    if not ip_command:
        return []
    try:
        result = subprocess.run(
            [ip_command, "-j", "-4", "address", "show", "scope", "global"],
            capture_output=True,
            text=True,
            timeout=12,
        )
        if result.returncode != 0:
            return []
        interfaces = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    networks = []
    for interface in interfaces:
        for address in interface.get("addr_info", []):
            if address.get("family") != "inet":
                continue
            network = _usable_network(address.get("local"), address.get("prefixlen"))
            if network is not None:
                networks.append(network)
    return networks


def local_ipv4_networks():
    if os.name == "nt":
        networks = _windows_ipv4_networks()
    else:
        networks = _ip_command_networks()
    return sorted(set(networks), key=lambda network: (int(network.network_address), network.prefixlen))


def _network_hosts(networks):
    hosts = []
    seen = set()
    for network in networks:
        if network.num_addresses > MAX_ADDRESSES_PER_NETWORK:
            continue
        for host in network.hosts():
            value = str(host)
            if value not in seen:
                seen.add(value)
                hosts.append(value)
    if len(hosts) > MAX_TOTAL_ADDRESSES:
        raise ValueError("Local camera scan exceeds the address limit; narrow the connected subnets.")
    return hosts


def scan_host(host, ports=CAMERA_PORTS, timeout=DEFAULT_TIMEOUT):
    open_ports = []
    for port in ports:
        try:
            with socket.create_connection((str(host), int(port)), timeout=timeout):
                open_ports.append(int(port))
        except OSError:
            continue
    return {"host": str(host), "open_ports": open_ports}


def is_rtsp_server(host, port, timeout=0.5):
    request = f"OPTIONS rtsp://{host}:{int(port)}/ RTSP/1.0\r\nCSeq: 1\r\n\r\n"
    try:
        with socket.create_connection((str(host), int(port)), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(request.encode("ascii"))
            response = connection.recv(512).decode("ascii", errors="replace")
    except OSError:
        return False
    return bool(re.match(r"^RTSP/1\.0\s+\d{3}", response))


def _read_windows_neighbors():
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return {}
    command = (
        "Get-NetNeighbor -AddressFamily IPv4 | "
        "Select-Object IPAddress, LinkLayerAddress | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        records = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    if isinstance(records, dict):
        records = [records]
    neighbors = {}
    for record in records:
        address = record.get("IPAddress")
        link_layer = record.get("LinkLayerAddress")
        if not address or not link_layer:
            continue
        try:
            mac = normalize_mac(link_layer)
        except ValueError:
            continue
        if mac != "00:00:00:00:00:00" and mac != "ff:ff:ff:ff:ff:ff":
            neighbors[str(address)] = mac
    return neighbors


def _read_ip_neighbors():
    ip_command = shutil.which("ip")
    if ip_command:
        try:
            result = subprocess.run(
                [ip_command, "neighbor", "show"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        neighbors = {}
        for line in result.stdout.splitlines():
            match = re.match(r"^(\S+)\s+dev\s+\S+.*\blladdr\s+([0-9a-fA-F:.-]+)", line)
            if not match:
                continue
            try:
                neighbors[match.group(1)] = normalize_mac(match.group(2))
            except ValueError:
                continue
        return neighbors
    return {}


def read_neighbor_macs():
    if os.name == "nt":
        return _read_windows_neighbors()
    return _read_ip_neighbors()


def discover_camera(expected_mac, networks=None, progress_callback=None):
    target_mac = normalize_mac(expected_mac)
    networks = list(local_ipv4_networks() if networks is None else networks)
    hosts = _network_hosts(networks)
    if not hosts:
        return {"status": "no_networks", "candidates": [], "match": None}

    scanned = []
    workers = min(MAX_WORKERS, len(hosts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan_host, host) for host in hosts]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["open_ports"]:
                scanned.append(result)
            if progress_callback:
                progress_callback(completed, len(hosts))

    neighbors = read_neighbor_macs()
    candidates = []
    for result in scanned:
        host = result["host"]
        mac_address = neighbors.get(host)
        if not mac_address:
            continue
        rtsp_port = next(
            (
                port
                for port in RTSP_PORTS
                if port in result["open_ports"] and is_rtsp_server(host, port)
            ),
            None,
        )
        if rtsp_port is None:
            continue
        candidates.append({
            "host": host,
            "mac_address": mac_address,
            "open_ports": result["open_ports"],
            "rtsp_port": rtsp_port,
        })

    matches = [candidate for candidate in candidates if candidate["mac_address"] == target_mac]
    match = matches[0] if len(matches) == 1 else None
    status = "matched" if match else "ambiguous" if matches else "not_found"
    return {"status": status, "candidates": candidates, "match": match}