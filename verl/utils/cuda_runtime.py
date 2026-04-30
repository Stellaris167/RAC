import os
import sys


def _prepend_library_path(env_var: str, path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False

    current = os.environ.get(env_var, "")
    parts = [part for part in current.split(":") if part]
    if path in parts:
        return False

    os.environ[env_var] = ":".join([path, *parts]) if parts else path
    return True


def _contains_nvrtc_builtins(path: str) -> bool:
    try:
        return any(name.startswith("libnvrtc-builtins.so") for name in os.listdir(path))
    except OSError:
        return False


def get_nvrtc_library_candidates() -> list[str]:
    candidates: list[str] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "lib"))

    python_prefix = os.path.dirname(os.path.dirname(sys.executable))
    candidates.append(os.path.join(python_prefix, "lib"))

    try:
        import torch

        site_packages_dir = os.path.dirname(os.path.dirname(torch.__file__))
        cuda_version = getattr(torch.version, "cuda", "") or ""
        if cuda_version:
            cuda_major = cuda_version.split(".", 1)[0]
            candidates.append(os.path.join(site_packages_dir, "nvidia", f"cu{cuda_major}", "lib"))
        candidates.append(os.path.join(site_packages_dir, "nvidia", "cuda_nvrtc", "lib"))
    except Exception:
        pass

    return list(dict.fromkeys(candidates))


def ensure_nvrtc_builtins_on_library_path() -> list[str]:
    added_paths = []
    for path in get_nvrtc_library_candidates():
        if _contains_nvrtc_builtins(path) and _prepend_library_path("LD_LIBRARY_PATH", path):
            added_paths.append(path)
    return added_paths