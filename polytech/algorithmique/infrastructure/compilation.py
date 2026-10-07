"""Adaptateur : compiler un programme C++ et le lancer sur une saisie.

C'est le seul endroit du contexte qui lance un processus. Le compilateur
est cherche sur le poste ; il est appele en C++17, tous les
avertissements traites comme des erreurs : un corrige qui compile avec
des avertissements n'est pas un corrige.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

CANDIDATS: Final = ("clang++", "g++")
OPTIONS: Final = ("-std=c++17", "-Wall", "-Wextra", "-Werror")
DELAI: Final = 60
"""Secondes accordees a la compilation, puis a l'execution."""


def compilateur(candidats: tuple[str, ...] = CANDIDATS) -> str | None:
    """Le premier compilateur C++ present sur le poste, ou rien."""
    for candidat in candidats:
        trouve = shutil.which(candidat)
        if trouve is not None:
            return trouve
    return None


@dataclass(frozen=True, slots=True)
class ProgrammeCpp:
    """Un fichier source C++ qu'on peut compiler et lancer."""

    source: Path
    candidats: tuple[str, ...] = CANDIDATS
    options: tuple[str, ...] = OPTIONS

    def executer(self, saisie: str, dossier: Path) -> str:
        """Compile dans `dossier`, lance avec cette saisie, rend ce qui s'affiche."""
        outil = compilateur(self.candidats)
        if outil is None:
            msg = f"aucun compilateur C++ parmi {self.candidats}"
            raise FileNotFoundError(msg)
        binaire = dossier / self.source.stem
        commande = [outil, *self.options, str(self.source), "-o", str(binaire)]
        compilation = subprocess.run(  # noqa: S603 -- commande batie ici, sans shell
            commande, capture_output=True, text=True, timeout=DELAI, check=False
        )
        if compilation.returncode != 0:
            msg = f"compilation refusee : {compilation.stderr.strip()}"
            raise RuntimeError(msg)
        execution = subprocess.run(  # noqa: S603 -- le binaire qu'on vient de produire
            [str(binaire)],
            input=saisie,
            capture_output=True,
            encoding="utf-8",
            errors="replace",  # un programme en Latin-1 : accents remplaces, visibles
            timeout=DELAI,
            check=False,
        )
        if execution.returncode != 0:
            msg = f"le programme a rendu le code {execution.returncode}"
            raise RuntimeError(msg)
        return execution.stdout
