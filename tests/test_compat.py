import json

from callgrader.llm import compat_config


def test_compat_config_restores_rope_theta_and_special_tokens(tmp_path):
    # what transformers 5 writes: rope_theta nested, extra_special_tokens as a list
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "qwen2", "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": ["<|im_start|>", "<|im_end|>"]}))
    compat_config(str(tmp_path))
    cfg = json.loads((tmp_path / "config.json").read_text())
    tok = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert cfg["rope_theta"] == 1000000.0 and cfg["rope_parameters"]["rope_theta"] == 1000000.0
    assert tok["additional_special_tokens"] == ["<|im_start|>", "<|im_end|>"] and "extra_special_tokens" not in tok
    compat_config(str(tmp_path))  # running twice changes nothing
    assert json.loads((tmp_path / "config.json").read_text()) == cfg
