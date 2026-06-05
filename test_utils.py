import pytest
from utils import _normalize_text, _normalize_npk, _score_numeric, _score_npk

def test_normalize_text():
    assert _normalize_text("  Kharif  ") == "kharif"
    assert _normalize_text(None) is None
    assert _normalize_text("") is None
    assert _normalize_text("   ") is None

def test_normalize_npk():
    assert _normalize_npk("High") == "high"
    assert _normalize_npk("h") == "high"
    assert _normalize_npk("Med") == "medium"
    assert _normalize_npk(100) == "high"
    assert _normalize_npk(50) == "medium"
    assert _normalize_npk(10) == "low"
    assert _normalize_npk(None) is None

def test_score_numeric():
    # Value in range
    assert _score_numeric(25, 20, 30) == 1.0
    # Value outside range
    assert _score_numeric(15, 20, 30) < 1.0
    assert _score_numeric(35, 20, 30) < 1.0
    # Penalty calculation
    assert _score_numeric(10, 20, 30) == 0.0 # (20-10)/10 = 1.0 penalty

def test_score_npk():
    assert _score_npk("high", "high") == 1.0
    assert _score_npk("medium", "high") == 0.65
    assert _score_npk("low", "high") == 0.3
    assert _score_npk(None, "high") == 0.4
