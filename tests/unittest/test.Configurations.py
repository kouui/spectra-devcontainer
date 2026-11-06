import pytest

from spectra import Configurations as _CFG


class Test_jit_from_env:
    def test_unset_is_off(self):
        assert _CFG._jit_from_env_(None) is False

    def test_zero_is_off(self):
        assert _CFG._jit_from_env_("0") is False

    def test_one_is_on(self):
        assert _CFG._jit_from_env_("1") is True

    @pytest.mark.parametrize("value", ["true", "", "2", " 1"])
    def test_invalid_value_raises(self, value):
        with pytest.raises(ValueError, match="SPECTRA_JIT"):
            _CFG._jit_from_env_(value)
