import os
import tempfile

# Runtime data (a symlink on the VM) holds old deployment copies of tests; never collect it.
collect_ignore = ["data"]

# Keep Hugging Face caches private to the test run; the VM's shared cache is not writable.
_hf_home = tempfile.mkdtemp(prefix="forgetune-test-hf-")
os.environ["HF_HOME"] = _hf_home
os.environ["HF_DATASETS_CACHE"] = os.path.join(_hf_home, "datasets")
os.environ["HF_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_home, "transformers")
