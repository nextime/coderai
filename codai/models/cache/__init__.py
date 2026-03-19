"""
Model Cache - Unified model loading, caching, downloading, and management.

This module provides unified functions that handle three types of model sources:
1. Local files: Used directly without caching
2. URLs: Downloaded to CoderAI cache (~/.cache/coderai/models) if not cached
3. HF model IDs: Downloaded to HF cache (~/.cache/huggingface/hub) if not cached

All functions work seamlessly across:
- CoderAI GGUF cache for downloaded files
- HuggingFace cache for HF models and datasets
- Direct local file access

Functions available:
- Intelligent model loading with automatic source detection
- Cache directory management for all cache types
- Checking for cached models (URLs, HF IDs) and local file validation
- Downloading models from URLs or HuggingFace with caching
- Listing and removing cached models across all cache types
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


def get_cached_model_path(model_path: str) -> Optional[str]:
    """
    Check if a model is already cached. Works with URLs, HF model IDs, and local files.

    Args:
        model_path: Local path, URL, or HuggingFace model ID

    Returns:
        Path to cached model if found, local path if local file exists, None otherwise
    """
    # 1. Check if it's a local file - return directly if it exists
    if os.path.isfile(model_path):
        return model_path

    # 2. Check if it's a HuggingFace model ID
    if is_huggingface_model_id(model_path):
        # For HF models, check the HF cache
        caches = get_all_cache_dirs()
        hf_dir = caches.get('huggingface')
        if hf_dir:
            try:
                from huggingface_hub import scan_cache_dir
                cache_info = scan_cache_dir(hf_dir)

                # Look for the model in the cache
                for repo in cache_info.repos:
                    if repo.repo_id == model_path:
                        # Return the path to the first file (typically the model file)
                        if repo.revisions:
                            latest_rev = repo.revisions[0]  # Use first revision
                            if latest_rev.files:
                                file_path = os.path.join(hf_dir, latest_rev.files[0].file_path)
                                if os.path.exists(file_path):
                                    print(f"Using cached HF model: {file_path}")
                                    return file_path
            except (ImportError, Exception):
                pass
    else:
        # 3. For URLs, check the coderai cache
        cache_dir = get_model_cache_dir()
        # Create a safe filename from the URL using SHA-256 hash
        url_hash = hashlib.sha256(model_path.encode()).hexdigest()
        url_path = model_path.split('?')[0]  # Remove query params
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


def download_huggingface_model(model_id: str, cache_dir: Optional[str] = None, file_pattern: str = '.gguf') -> Optional[str]:
    """Download a model from Hugging Face by model ID. Returns cached path or None on failure."""
    if cache_dir is None:
        caches = get_all_cache_dirs()
        cache_dir = caches.get('huggingface')
        if not cache_dir:
            print("No HuggingFace cache directory found")
            return None
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


def load_model(model_path: str, cache_dir: Optional[str] = None, file_pattern: str = '.gguf') -> Optional[str]:
    """
    Load a model from various sources with intelligent caching.

    Handles three types of inputs:
    1. Local file path: Use directly (no caching)
    2. URL: Download if not cached, then use cached version
    3. HF model ID: Download via HF API if not cached, then use

    Args:
        model_path: Local path, URL, or HuggingFace model ID
        cache_dir: Specific cache directory (auto-detected if None)
        file_pattern: File pattern for HF downloads (default: '.gguf')

    Returns:
        Path to model file (local or cached), or None on failure
    """
    # 1. Check if it's a local file path
    if os.path.isfile(model_path):
        # Local file - use directly without caching
        return model_path

    # 2. Check if it's a URL (starts with http/https)
    if model_path.startswith('http://') or model_path.startswith('https://'):
        # URL - check cache first, download if needed
        cached_path = get_cached_model_path(model_path)
        if cached_path:
            return cached_path
        # Not cached - download
        return download_model(model_path, cache_dir, file_pattern)

    # 3. Assume it's a HuggingFace model ID
    # Check cache first, download if needed
    cached_path = get_cached_model_path(model_path)
    if cached_path:
        return cached_path
    # Not cached - download from HF
    return download_model(model_path, cache_dir, file_pattern)


def download_model(model_path: str, cache_dir: Optional[str] = None, file_pattern: str = '.gguf') -> Optional[str]:
    """
    Download a model from URL or HuggingFace. Works with both URLs and model IDs.

    Args:
        model_path: URL or HuggingFace model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
        cache_dir: Cache directory to use (auto-detected if None)
        file_pattern: File pattern to download for HF models (default: '.gguf')

    Returns:
        Path to downloaded model, or None on failure
    """
    # Check if it's a HuggingFace model ID
    if is_huggingface_model_id(model_path):
        # Use HF cache if no specific cache_dir provided
        if cache_dir is None:
            caches = get_all_cache_dirs()
            cache_dir = caches.get('huggingface')
            if not cache_dir:
                print("No HuggingFace cache directory found")
                return None

        return download_huggingface_model(model_path, cache_dir, file_pattern)
    else:
        # It's a URL - use coderai cache
        if cache_dir is None:
            cache_dir = get_model_cache_dir()

        # Download from URL
        import requests

        url = model_path
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
        cached_path = os.path.join(cache_dir, cached_filename)

        # Check if already cached (double-check)
        if os.path.exists(cached_path):
            print(f"Using cached model: {cached_path}")
            return cached_path

        # Download
        print(f"Downloading model: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0

        downloaded = 0
        start_time = time.time()

        with open(cached_path, 'wb') as f:
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
        print(f"Downloaded and cached to: {cached_path}")
        if total_mb > 0:
            print(f"File size: {total_mb:.1f} MB")

        return cached_path


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
            # First, try to match by repo ID for HuggingFace models
            if cache_name == 'huggingface':
                try:
                    from huggingface_hub import scan_cache_dir
                    cache_info = scan_cache_dir(cache_dir)

                    # Check if match_term matches any repo_id
                    for repo in cache_info.repos:
                        if match_term.lower() in repo.repo_id.lower():
                            # Found matching repo - remove the entire repo directory
                            # This is more thorough than removing individual files
                            repo_dir = os.path.join(cache_dir, f"models--{repo.repo_id.replace('/', '--')}")
                            if os.path.exists(repo_dir):
                                # Calculate total size of all files in the repo
                                total_size = 0
                                for root, dirs, files in os.walk(repo_dir):
                                    for f in files:
                                        filepath = os.path.join(root, f)
                                        try:
                                            total_size += os.path.getsize(filepath)
                                        except OSError:
                                            pass
                                # Add the entire directory for removal
                                all_matching.append((cache_name, f"models--{repo.repo_id.replace('/', '--')}", repo_dir, total_size))
                            break  # Only match one repo per search term
                except ImportError:
                    # huggingface_hub not available, fall back to filename search
                    pass
                except Exception:
                    # Error scanning HF cache, fall back to filename search
                    pass

            # Fall back to filename search for both diffusers and huggingface
            if not all_matching:  # Only if we didn't find repo matches
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

    # Remove matching files/directories
    removed = []
    for cache_name, filename, filepath, size in all_matching:
        try:
            if os.path.isdir(filepath):
                # Remove entire directory tree
                shutil.rmtree(filepath)
                print(f"  Deleted: [{cache_name}] {filename}/ (directory)")
            else:
                # Remove single file
                os.remove(filepath)
                print(f"  Deleted: [{cache_name}] {filename}")
            removed.append((cache_name, filename, size))
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


# Export all public functions
__all__ = [
    'get_model_cache_dir',
    'get_all_cache_dirs',
    'get_cached_model_path',
    'is_huggingface_model_id',
    'download_huggingface_model',
    'load_model',
    'download_model',
    'list_cached_models',
    'remove_cached_model',
    'list_cached_models_info',
    'remove_all_cached_models',
]
