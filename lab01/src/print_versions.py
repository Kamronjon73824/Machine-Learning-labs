"""Print Python and library versions. Usage: python print_versions.py [output_file]"""
import platform
import sys
from importlib import metadata

PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi", "uvicorn",
    "pytest", "httpx", "locust", "requests", "pyarrow", "joblib", "tqdm", "pyyaml",
]


def versions_text():
    lines = [f"Python {sys.version.split()[0]} ({platform.platform()})"]
    for pkg in PACKAGES:
        try:
            lines.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{pkg}: not installed")
    return "\n".join(lines)


if __name__ == "__main__":
    text = versions_text()
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(text + "\n")
