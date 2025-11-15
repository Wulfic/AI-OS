import pytest

from aios.gui.utils.model_display import get_model_display_name


@pytest.mark.parametrize(
    "identifier,expected",
    [
        ("gpt2", "gpt2"),
        ("/models/English-v1", "English-v1"),
        (r"C:\\deep\\models\\brain.bin", "brain.bin"),
        ("brain_path=C:/Users/tyler/Repos/AI-OS/artifacts/brains/actv1/English-v1", "English-v1"),
        (r"model_path=\\share\\brains\\French-v2", "French-v2"),
    ],
)
def test_get_model_display_name(identifier: str, expected: str) -> None:
    assert get_model_display_name(identifier) == expected


def test_empty_identifier_returns_unknown() -> None:
    assert get_model_display_name("") == "Unknown"
    assert get_model_display_name(None) == "Unknown"
