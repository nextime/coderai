# README Update - 2026-05-05

## Summary

Updated the README.md to reflect the current configuration-based architecture implemented in the 2026-05-03 refactoring. The README was outdated and still documented the old CLI-heavy approach with numerous command-line flags.

## Key Changes

### 1. Updated Feature Section
- Reorganized into three subsections: Core Capabilities, GPU Backend Support, Advanced Features
- Emphasized the web admin dashboard and configuration-based approach
- Highlighted multi-modal support (text, image, audio, TTS)
- Added per-model configuration as a key feature

### 2. Installation Section
- Updated build script examples to show `./build.sh all` option
- Clarified that `all` installs support for all backends
- Maintained backward compatibility with `nvidia` and `vulkan` options

### 3. Usage Section - Major Overhaul
- **Removed**: All old CLI examples with `--model`, `--backend`, `--load-in-4bit`, etc.
- **Added**: 
  - Quick start guide with simple `python coderai` command
  - Access points (Admin Dashboard, Chat Interface, API, Docs)
  - First login credentials
  - Configuration files overview
  - Updated command-line options (only `--config`, `--debug`, `--dump`, model management, and utility flags)

### 4. Configuration Section - New Structure
- Added comprehensive configuration file examples:
  - `config.json` - Server, backend, and global settings
  - `models.json` - Model registry with per-model configurations
  - `auth.json` - Users, API tokens, and sessions
- Added "Managing Configuration" subsection:
  - Via Web Dashboard (recommended)
  - Via Configuration Files (manual editing)
- Added "Per-Model Configuration" with detailed settings for each backend
- Added "Backend Selection" and "Model Loading Modes" subsections

### 5. Backend-Specific Setup - Restructured
- **NVIDIA (CUDA)**: Removed CLI examples, added `models.json` configuration example
- **AMD and Intel (Vulkan)**: Removed CLI examples, added `models.json` and `config.json` configuration examples
- **CPU-Only**: Updated to show configuration-based approach
- **Low VRAM Configuration**: Changed from CLI flags to config file examples (global and per-model)
- **Multi-GPU with Vulkan**: Updated to use `config.json` settings instead of CLI flags

### 6. Removed Sections
- Removed "Reply Filters" section (not in current CLI)
- Removed "HuggingFace Chat Template" section (not in current CLI)
- Removed "Backend Selection" CLI examples
- Removed "Model Formats by Backend" CLI examples
- Removed all "Examples" subsection with CLI commands

### 7. Maintained Sections
- API Documentation (unchanged - still valid)
- Model Recommendations (unchanged - still valid)
- Troubleshooting (unchanged - examples are still helpful)
- License, Contributing, Acknowledgments (unchanged)

## Architecture Documented

### Before (Old README)
```
Command Line (many flags) → main.py → FastAPI API
```

### After (Updated README)
```
~/.coderai/
├── config.json       # Server, backend, global settings
├── models.json       # Per-model configs
├── auth.json         # Users, tokens, sessions
└── secret_key        # Session signing key
    ↓
ConfigManager → main.py → FastAPI (API + Admin UI + Chat)
```

## User Experience Improvements

1. **Simpler Getting Started**: Users now just run `python coderai` instead of memorizing complex CLI flags
2. **Web-Based Management**: All configuration through the admin dashboard at `http://localhost:8000/admin`
3. **Persistent Configuration**: Settings saved in JSON files, no need to remember CLI arguments
4. **Per-Model Settings**: Each model can have its own configuration (GPU layers, quantization, context size)
5. **Better Documentation**: Clear separation between installation, usage, and configuration

## Files Modified

- `/storage/coderai/README.md` - Complete overhaul (~1009 lines)

## Validation

- ✅ All sections updated to reflect configuration-based architecture
- ✅ Removed outdated CLI examples
- ✅ Added comprehensive configuration examples
- ✅ Maintained valid troubleshooting and model recommendation sections
- ✅ Preserved license and acknowledgments
- ✅ Structure is clear and easy to navigate

## Next Steps

Users should now:
1. Run `./build.sh all` to install
2. Run `python coderai` to start
3. Visit `http://localhost:8000/admin` to configure
4. Use the web dashboard for all model and settings management

No more memorizing CLI flags!
