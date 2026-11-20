from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.models.manager import MultiModelManager


def test_multitype_model_allows_text_when_registered_as_image_too():
    manager = MultiModelManager()
    manager.set_default_model("deepseek-ai/DeepSeek-V4", config={"load_mode": "on-request"})
    manager.set_image_model("deepseek-ai/DeepSeek-V4", config={"load_mode": "on-request"})

    assert manager.get_registered_model_type("DeepSeek-V4") is None
    assert manager.is_allowed_model("deepseek-ai/DeepSeek-V4", "text") is True
    assert manager.is_allowed_model("DeepSeek-V4", "text") is True


def test_single_type_image_model_still_rejects_text_requests():
    manager = MultiModelManager()
    manager.set_image_model("org/ImageOnly", config={"load_mode": "on-request"})

    assert manager.get_registered_model_type("ImageOnly") == "image"
    assert manager.is_allowed_model("org/ImageOnly", "text") is False


def test_text_model_uses_config_from_prefixed_multitype_entry():
    manager = MultiModelManager()
    manager.set_image_model(
        "deepseek-ai/DeepSeek-V4",
        config={"load_mode": "on-request", "load_in_4bit": True},
    )
    manager._remember_registered_type("deepseek-ai/DeepSeek-V4", "text")

    info = manager.request_model("deepseek-ai/DeepSeek-V4", model_type="text")

    assert info["model_key"] == "deepseek-ai/DeepSeek-V4"
    assert info["config"]["load_in_4bit"] is True
    assert info["already_loaded"] is False


def test_multitype_model_infers_text_support_from_saved_config(monkeypatch):
    from codai.admin import routes

    class DummyConfigManager:
        models_data = {
            "image_models": [
                {
                    "path": "mlx-community/DeepSeek-V4-Flash-4bit",
                    "model_type": "text_models",
                    "model_types": ["text_models", "image_models"],
                }
            ]
        }

    monkeypatch.setattr(routes, "config_manager", DummyConfigManager())
    manager = MultiModelManager()
    manager.set_image_model("mlx-community/DeepSeek-V4-Flash-4bit", config={"load_mode": "on-request"})

    assert manager.get_registered_model_type("mlx-community/DeepSeek-V4-Flash-4bit") is None
    assert manager.is_allowed_model("mlx-community/DeepSeek-V4-Flash-4bit", "text") is True
