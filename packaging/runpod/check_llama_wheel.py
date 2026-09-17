# Build-time check for the llama pod image: the CUDA wheel is installed, its
# shared libraries resolve, and the CUDA backend is actually in it.
import glob
import subprocess

libs = glob.glob("/usr/local/lib/python3.12/site-packages/llama_cpp/lib/*.so*")
assert libs, "no llama_cpp shared libraries installed"
# libcuda.so.1 (and libnvidia-ml) belong to the DRIVER: the container runtime
# mounts them from the host at run time, and no image ever carries them.
DRIVER = ("libcuda.so", "libnvidia-ml.so")


def _unresolved(lib):
    out = subprocess.run(["ldd", lib], capture_output=True, text=True).stdout
    return [ln.split()[0] for ln in out.splitlines()
            if "not found" in ln and not ln.strip().startswith(DRIVER)]


bad = {l: m for l in libs if (m := _unresolved(l))}
assert not bad, f"unresolved libraries in {bad}"
# Not `import llama_cpp`: loading libggml-cuda needs the driver, which a build
# host without a GPU does not have. Linkage is proven above; the version comes
# from the package metadata.
import importlib.metadata  # noqa: E402
print("llama-cpp-python", importlib.metadata.version("llama_cpp_python"),
      "with", len(libs), "libs, all resolved")
assert any("ggml-cuda" in l for l in libs), "the wheel was built WITHOUT CUDA"
