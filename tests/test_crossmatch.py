import sys
from pathlib import Path

# Add project root to path so tasni is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasni.crossmatch.crossmatch_full import get_completed_tiles, get_ready_tiles


class TestCrossmatch:
    def test_get_ready_tiles(self):
        result = get_ready_tiles()
        assert isinstance(result, (list, set))

    def test_get_completed_tiles(self):
        result = get_completed_tiles()
        assert isinstance(result, (list, set))
