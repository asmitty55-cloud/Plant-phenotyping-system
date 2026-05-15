import errno
import os
import time
from pt.core.utils.path_utils import get_debug_dir, get_user_data_root

class PhoneLogger:
    def __init__(self, device_id):
        self.device_id = device_id
        self.log_dir = get_debug_dir()
        self.log_path = os.path.join(self.log_dir, f"{device_id}.txt")
        self.rotated_paths = [
            os.path.join(self.log_dir, f"{device_id}_1.txt"),
            os.path.join(self.log_dir, f"{device_id}_2.txt"),
            os.path.join(self.log_dir, f"{device_id}_3.txt"),
        ]
        self.max_size = 1 * 1024 * 1024  # 1 MB
        self._ensure_log_path()

    def _ensure_log_path(self):
        try:
            with self._open_log_file("a"):
                pass
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EPERM):
                self._fallback_to_user_debug_dir()
            else:
                raise

    def _fallback_to_user_debug_dir(self):
        fallback_root = get_user_data_root()
        self.log_dir = os.path.join(fallback_root, "debug")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{self.device_id}.txt")
        self.rotated_paths = [
            os.path.join(self.log_dir, f"{self.device_id}_1.txt"),
            os.path.join(self.log_dir, f"{self.device_id}_2.txt"),
            os.path.join(self.log_dir, f"{self.device_id}_3.txt"),
        ]

    def _open_log_file(self, mode="a"):
        try:
            return open(self.log_path, mode, encoding="utf-8")
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EPERM):
                self._fallback_to_user_debug_dir()
                return open(self.log_path, mode, encoding="utf-8")
            raise

    def _timestamp(self):
        return time.strftime("[%Y-%m-%d %H:%M:%S]")

    def _rotate_logs(self):
        # Delete oldest
        if os.path.exists(self.rotated_paths[2]):
            os.remove(self.rotated_paths[2])

        # Shift 2 → 3
        if os.path.exists(self.rotated_paths[1]):
            os.rename(self.rotated_paths[1], self.rotated_paths[2])

        # Shift 1 → 2
        if os.path.exists(self.rotated_paths[0]):
            os.rename(self.rotated_paths[0], self.rotated_paths[1])

        # Main → 1
        if os.path.exists(self.log_path):
            os.rename(self.log_path, self.rotated_paths[0])

    def _check_rotate_before_write(self):
        if os.path.exists(self.log_path):
            if os.path.getsize(self.log_path) >= self.max_size:
                self._rotate_logs()

    def log(self, message, major=False):
        """
        major=True → prepend timestamp
        major=False → no timestamp
        """
        self._check_rotate_before_write()

        if major:
            line = f"{self._timestamp()} {message}\n"
        else:
            line = f"{message}\n"

        with self._open_log_file("a") as f:
            f.write(line)

    def log_error_with_adb_output(self, message, adb_stdout, adb_stderr):
        """
        Used only when something fails.
        Includes raw ADB output.
        """
        self._check_rotate_before_write()

        entry = (
            f"{self._timestamp()} ERROR: {message}\n"
            f"ADB OUT:\n{adb_stdout}\n"
            f"ADB ERR:\n{adb_stderr}\n"
        )

        with self._open_log_file("a") as f:
            f.write(entry)
