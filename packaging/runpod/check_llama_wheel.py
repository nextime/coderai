# Build-time check for the llama pod image: the CUDA wheel is installed, its
# shared libraries resolve, and the CUDA backend is actually in it.
import glob
import subprocess

libs = glob.glob("/usr/local/lib/python3.12/site-packages/llama_cpp/lib/*.so*")
assert libs, "no llama_cpp shared libraries installed"
bad = [l for l in libs
       if "not found" in subprocess.run(["ldd", l], capture_output=True, text=True).stdout]
assert not bad, f"unresolved libraries in {bad}"
import llama_cpp  # noqa: E402
print("llama-cpp-python", llama_cpp.__version__, "with", len(libs), "libs, all resolved")
assert any("ggml-cuda" in l for l in libs), "the wheel was built WITHOUT CUDA"
