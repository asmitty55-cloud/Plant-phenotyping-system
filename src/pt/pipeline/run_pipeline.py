from pt.core.analysis import process_latest_captures
from pt.core.utils.path_utils import get_captures_dir
from pt.device.capture_service.capture import capture_all_devices

def main():
    print("Starting phenotyping pipeline...")

    captures = capture_all_devices()
    results = process_latest_captures(get_captures_dir())

    print(f"Captured {len(captures)} frame(s).")
    print("Analysis complete:", results)

if __name__ == "__main__":
    main()
