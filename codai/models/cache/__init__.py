"""
Model Cache - Handles model caching, downloading, and management.

This module provides functions for:
- Model cache directory management
- Checking for cached models
- Downloading models from URLs and HuggingFace
- Listing and removing cached models
"""

import os
import hashlib
import pathlib
from typing import Optional, Dict, List, Tuple

# For type hints
import time


def get_model_cache_dir() -> str:
    """Get or create the model cache directory."""
    # Use XDG_CACHE_HOME if set, otherwise use ~/.cache/coderai
    cache_home = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    cache_dir = os.path.join(cache_home, 'coderai', 'models')
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_all_cache_dirs() -> dict:
    """Get all model cache directories."""
    caches = {}
    cache_home = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))

    # Coderai GGUF cache
    coderai_cache = os.path.join(cache_home, 'coderai', 'models')
    if os.path.exists(coderai_cache):
        caches['coderai'] = coderai_cache

    # HuggingFace cache (for .safetensors, PyTorch models, etc.)
    # Check both the main directory and the hub subdirectory
    hf_cache = os.path.join(cache_home, 'huggingface')
    hf_hub_cache = os.path.join(hf_cache, 'hub')
    if os.path.exists(hf_hub_cache):
        caches['huggingface'] = hf_hub_cache  # Use hub directory if it exists
    elif os.path.exists(hf_cache):
        caches['huggingface'] = hf_cache

    # Local diffusers cache (often stored locally by apps)
    local_diffusers = os.path.expanduser('~/.cache/diffusers')
    if os.path.exists(local_diffusers):
        caches['diffusers'] = local_diffusers

    return caches


def get_cached_model_path(url: str) -> Optional[str]:
    """Check if a model URL is already cached. Returns path if cached, None otherwise."""
    cache_dir = get_model_cache_dir()
    # Create a safe filename from the URL using SHA-256 hash
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    url_path = url.split('?')[0]  # Remove query params
    filename = os.path.basename(url_path)
    if not filename:
        filename = "model.gguf"
    # Match the format used in load_model: {hash}_{filename}
    cached_filename = f"{url_hash}_{filename}"
    cached_path = os.path.join(cache_dir, cached_filename)
    
    if os.path.exists(cached_path):
        print(f"Using cached model: {cached_path}")
        return cached_path
    return None


def is_huggingface_model_id(path: str) -> bool:
    """Check if the path is a Hugging Face model ID (e.g., 'Qwen/Qwen3-4B-Instruct-2507-Q3_K_S')."""
    # Must contain / but not be a URL
    return '/' in path and not path.startswith('http://') and not path.startswith('https://')


def download_huggingface_model(model_id: str, cache_dir: str, file_pattern: str = '.gguf') -> Optional[str]:
    """Download a model from Hugging Face by model ID. Returns cached path or None on failure."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        print(f"Downloading from Hugging Face: {model_id}")
        files = list_repo_files(model_id)
        
        # Filter files by extension pattern
        matching_files = [f for f in files if f.endswith(file_pattern)]
        if not matching_files:
            print(f"No {file_pattern} files found in {model_id}")
            return None
        
        # Download the first matching file
        filename = matching_files[0]
        print(f"Downloading: {filename}")
        model_path = hf_hub_download(repo_id=model_id, filename=filename, cache_dir=cache_dir)
        print(f"Downloaded model to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        return None


def download_model(url: str, cache_dir: str) -> str:
    """Download a model from URL with progress reporting. Returns cached path."""
    import requests
    
    url_path = url.split('?')[0]
    filename = os.path.basename(url_path)
    
    # Determine file extension
    if 'gguf' in url.lower():
        ext = '.gguf'
    elif 'bin' in url.lower():
        ext = '.bin'
    elif 'ggml' in url.lower():
        ext = '.ggml'
    else:
        ext = '.bin'
    
    if not filename.endswith(ext):
        filename = f"model{ext}"
    
    # Create safe filename in cache
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    cached_filename = f"{url_hash}_{filename}"
    model_path = os.path.join(cache_dir, cached_filename)
    
    # Check if already cached
    if os.path.exists(model_path):
        print(f"Using cached model: {model_path}")
        return model_path
    
    # Download
    print(f"Downloading model: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
    
    downloaded = 0
    start_time = time.time()
    
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    speed_mb = speed / (1024 * 1024)
                    dl_mb = downloaded / (1024 * 1024)
                    print(f"Downloaded: {percent:.1f}% ({dl_mb:.1f}/{total_mb:.1f} MB) at {speed_mb:.1f} MB/s", end='\r')
    
    print()  # New line after progress
    print(f"Downloaded and cached to: {model_path}")
    if total_mb > 0:
        print(f"File size: {total_mb:.1f} MB")
    
    return model_path


def list_cached_models() -> Tuple[List[Tuple[str, str, int]], int]:
    """
    List all cached models.
    
    Returns:
        Tuple of (list of (cache_name, filename, size), total_size_bytes)
    """
    caches = get_all_cache_dirs()
    all_files = []
    total_size = 0
    
    for cache_name, cache_dir in caches.items():
        if not os.path.exists(cache_dir):
            continue
            
        files = os.listdir(cache_dir)
        if not files:
            continue
            
        # For diffusers and huggingface, show directory structure
        if cache_name in ('diffusers', 'huggingface'):
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    filepath = os.path.join(root, f)
                    rel_path = os.path.relpath(filepath, cache_dir)
                    size = os.path.getsize(filepath)
                    all_files.append((cache_name, rel_path, size))
                    total_size += size
        else:
            for f in files:
                filepath = os.path.join(cache_dir, f)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    all_files.append((cache_name, f, size))
                    total_size += size
    
    return all_files, total_size


def remove_cached_model(match_term: str) -> List[Tuple[str, str, int]]:
    """
    Remove cached models matching the given term.
    
    Args:
        match_term: String to match against cached model names
        
    Returns:
        List of (cache_name, filename, size) for removed files
    """
    import shutil
    
    caches = get_all_cache_dirs()
    all_matching = []
    
    for cache_name, cache_dir in caches.items():
        if not os.path.exists(cache_dir):
            continue
            
        # For diffusers and huggingface, search recursively
        if cache_name in ('diffusers', 'huggingface'):
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    if match_term.lower() in f.lower():
                        filepath = os.path.join(root, f)
                        rel_path = os.path.relpath(filepath, cache_dir)
                        size = os.path.getsize(filepath)
                        all_matching.append((cache_name, rel_path, filepath, size))
        else:
            files = os.listdir(cache_dir)
            for f in files:
                if match_term.lower() in f.lower():
                    filepath = os.path.join(cache_dir, f)
                    if os.path.isfile(filepath):
                        size = os.path.getsize(filepath)
                        all_matching.append((cache_name, f, filepath, size))
    
    # Remove matching files
    removed = []
    for cache_name, filename, filepath, size in all_matching:
        try:
            os.remove(filepath)
            removed.append((cache_name, filename, size))
            print(f"  Deleted: [{cache_name}] {filename}")
        except Exception as e:
            print(f"  Error deleting [{cache_name}] {filename}: {e}")
    
    return removed


def list_cached_models_info() -> dict:
    """
    List cached models with proper model-level information.

    Returns:
        Dict with cache information containing:
        - 'coderai': List of (filename, size_mb) tuples for GGUF files
        - 'huggingface': List of (repo_id, size_gb, revision_count) tuples for HF models
        - 'total_models': Total number of models
        - 'total_size_gb': Total size in GB
    """
    caches = get_all_cache_dirs()
    result = {
        'coderai': [],
        'huggingface': [],
        'total_models': 0,
        'total_size_gb': 0.0
    }

    # List CoderAI GGUF files
    coderai_dir = caches.get('coderai')
    if coderai_dir and os.path.exists(coderai_dir):
        files = [f for f in os.listdir(coderai_dir) if os.path.isfile(os.path.join(coderai_dir, f))]
        for filename in sorted(files):
            filepath = os.path.join(coderai_dir, filename)
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            result['coderai'].append((filename, size_mb))
            result['total_size_gb'] += size_mb / 1024  # Convert MB to GB
        result['total_models'] += len(files)

    # List HuggingFace models using HF API
    hf_dir = caches.get('huggingface')
    if hf_dir and os.path.exists(hf_dir):
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir(hf_dir)

            for repo in sorted(cache_info.repos, key=lambda x: x.repo_id):
                # Calculate total size for this repo
                repo_size = sum(revision.size_on_disk for revision in repo.revisions)
                size_gb = repo_size / (1024 * 1024 * 1024)
                revision_count = len(repo.revisions)

                result['huggingface'].append((repo.repo_id, size_gb, revision_count))
                result['total_models'] += 1
                result['total_size_gb'] += size_gb

        except ImportError:
            # huggingface_hub not available
            pass
        except Exception as e:
            # Error listing HF cache
            pass

    return result


def remove_all_cached_models() -> int:
    """
    Remove all cached models.
    
    Returns:
        Number of items removed
    """
    import shutil
    
    caches = get_all_cache_dirs()
    total_removed = 0
    
    for cache_name, cache_dir in caches.items():
        if not os.path.exists(cache_dir):
            continue
            
        files = os.listdir(cache_dir)
        if not files:
            continue
            
        # For diffusers, remove entire directory tree
        if cache_name == 'diffusers':
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    total_removed += 1
        else:
            for f in files:
                filepath = os.path.join(cache_dir, f)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    total_removed += 1
    
    return total_removed
