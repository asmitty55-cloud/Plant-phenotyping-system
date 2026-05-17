#!/usr/bin/env python
"""
Start Plant Observatory without a command prompt and open it in an app window.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:5000/"


def port_is_open(host="127.0.0.1", port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_open():
            try:
                with urllib.request.urlopen(URL, timeout=2) as response:
                    if response.status == 200:
                        return True
            except OSError:
                pass
        time.sleep(0.5)
    return False


def pythonw_path():
    exe = sys.executable
    candidate = exe.replace("python.exe", "pythonw.exe")
    if os.path.exists(candidate):
        return candidate
    return exe


def start_server_if_needed():
    if port_is_open():
        return

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

    subprocess.Popen(
        [pythonw_path(), os.path.join(ROOT, "main.py")],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )


def browser_candidates():
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    return [
        os.path.join(program_files, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(program_files_x86, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(local_app_data, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(program_files_x86, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe"),
    ]


def open_app_window():
    profile_dir = os.path.join(ROOT, ".observatory-browser")
    for browser in browser_candidates():
        if os.path.exists(browser):
            subprocess.Popen(
                [
                    browser,
                    f"--app={URL}",
                    "--new-window",
                    f"--user-data-dir={profile_dir}",
                    "--no-first-run",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return

    import webbrowser
    webbrowser.open(URL)


def main():
    start_server_if_needed()
    if wait_for_server():
        open_app_window()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
