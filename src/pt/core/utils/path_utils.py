import os
import tempfile


def _env_data_root():
    value = os.getenv("PT_DATA_ROOT")
    if value:
        return os.path.abspath(os.path.expanduser(value))
    return None


def _is_writable_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        test_path = os.path.join(path, ".path_check")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except OSError:
        return False


def get_default_app_root():
    env_root = _env_data_root()
    if env_root:
        return env_root

    return get_user_data_root()


def get_data_root(app_name="pt"):
    root = get_default_app_root()
    if _is_writable_dir(root):
        return root

    return get_user_data_root(app_name)


def get_user_data_root(app_name="pt"):
    fallback_home = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~")
    fallback_root = os.path.join(fallback_home, app_name)
    os.makedirs(fallback_root, exist_ok=True)
    if _is_writable_dir(fallback_root):
        return fallback_root

    temp_root = os.path.join(tempfile.gettempdir(), app_name)
    os.makedirs(temp_root, exist_ok=True)
    return temp_root


def _get_fallback_root(app_name="pt"):
    fallback_home = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~")
    fallback_root = os.path.join(fallback_home, app_name)
    os.makedirs(fallback_root, exist_ok=True)
    return fallback_root


def _get_writable_subdir(subdir_name):
    root = get_data_root()
    path = os.path.join(root, subdir_name)
    if _is_writable_dir(path):
        return path

    fallback_root = _get_fallback_root()
    path = os.path.join(fallback_root, subdir_name)
    if _is_writable_dir(path):
        return path

    temp_root = os.path.join(tempfile.gettempdir(), "pt")
    os.makedirs(temp_root, exist_ok=True)
    path = os.path.join(temp_root, subdir_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_debug_dir():
    return _get_writable_subdir("debug")


def get_captures_dir():
    return _get_writable_subdir("captures")


def get_images_dir():
    return _get_writable_subdir("images")
