"""Le noyau Rust est un prerequis : son absence est une erreur, pas un repli."""

import importlib
import sys

import pytest

from polytech import noyau


def test_l_extension_absente_leve_en_disant_comment_la_construire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "polytech_core", None)
    monkeypatch.delitem(sys.modules, "polytech.noyau", raising=False)
    with pytest.raises(ImportError, match=r"maturin develop"):
        importlib.import_module("polytech.noyau")
    monkeypatch.delitem(sys.modules, "polytech.noyau", raising=False)


def test_l_extension_presente_est_exposee() -> None:
    assert noyau.core.en_degres(0.0) == 0.0
