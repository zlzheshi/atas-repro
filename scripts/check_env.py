from __future__ import annotations

import importlib.util
import platform


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("python:", platform.python_version())

    for name in ["torch", "torchvision", "open_clip", "yaml", "tqdm", "PIL"]:
        print(f"{name}:", "ok" if has_module(name) else "missing")

    if has_module("torch"):
        import torch

        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        print("cuda device count:", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            print(f"gpu {index}:", torch.cuda.get_device_name(index))


if __name__ == "__main__":
    main()

