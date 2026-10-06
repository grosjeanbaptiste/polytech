"""`outils/muter.sh` : une mutation a la main, sans cache perime ni source abimee.

Le script nait d'une faute du 2026-10-06 : des mutations de meme taille,
restaurees dans la meme seconde, laissaient des `.pyc` que Python croyait
valides. Les mutants restaient actifs d'une mutation a la suivante.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

SCRIPT: Final = Path(__file__).resolve().parent.parent / "outils" / "muter.sh"
CODE: Final = "def somme(a, b):\n    return a + b\n"
TEST: Final = (
    "from module import somme\n\n\ndef test_somme():\n    assert somme(2, 3) == 5\n"
)
FAIBLE: Final = (
    "from module import somme\n\n\ndef test_somme():\n    assert somme(0, 0) == 0\n"
)


def _depot(tmp_path: Path, test: str) -> Path:
    (tmp_path / "module.py").write_text(CODE, encoding="utf-8")
    (tmp_path / "test_module.py").write_text(test, encoding="utf-8")
    return tmp_path


def _muter(depot: Path, motif: str, mutant: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- script du depot, arguments fixes
        ["/bin/bash", str(SCRIPT), "module.py", motif, mutant, "test_module.py"],
        cwd=depot,
        capture_output=True,
        text=True,
        check=False,
        env={"MUTER_PYTHON": sys.executable, "PATH": "/usr/bin:/bin"},
    )


def test_un_mutant_tue_rend_zero_et_le_dit(tmp_path: Path) -> None:
    """Les tests echouent sur le mutant : c'est le resultat attendu."""
    acheve = _muter(_depot(tmp_path, TEST), "a + b", "a - b")
    assert (acheve.returncode, "TUE" in acheve.stdout) == (0, True)


def test_un_mutant_survivant_rend_un_et_le_dit(tmp_path: Path) -> None:
    """Les tests passent sur le mutant : la garde ne garde rien."""
    acheve = _muter(_depot(tmp_path, FAIBLE), "a + b", "a - b")
    assert (acheve.returncode, "SURVIT" in acheve.stdout) == (1, True)


@pytest.mark.parametrize("test", [TEST, FAIBLE])
def test_la_source_est_rendue_a_l_identique(tmp_path: Path, test: str) -> None:
    """Contenu et date : un `.pyc` anterieur reste valide, aucun n'est a refaire."""
    depot = _depot(tmp_path, test)
    avant = (depot / "module.py").stat().st_mtime_ns
    _muter(depot, "a + b", "a - b")
    assert (depot / "module.py").read_text(encoding="utf-8") == CODE
    assert (depot / "module.py").stat().st_mtime_ns == avant
    assert not list(depot.glob("*.orig"))


def test_aucun_cache_du_mutant_n_est_ecrit(tmp_path: Path) -> None:
    """`python -B` : rien ne survit au mutant, quelle que soit sa taille."""
    depot = _depot(tmp_path, TEST)
    _muter(depot, "a + b", "a - b")
    assert not list(depot.rglob("module*.pyc"))


def test_un_motif_absent_est_refuse_sans_toucher_au_fichier(tmp_path: Path) -> None:
    """Une mutation qui ne s'applique pas ne passe pas pour une mutation tuee."""
    depot = _depot(tmp_path, TEST)
    acheve = _muter(depot, "a * b", "a - b")
    assert acheve.returncode == 2
    assert "motif absent" in acheve.stderr
    assert (depot / "module.py").read_text(encoding="utf-8") == CODE
