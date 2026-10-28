"""Adaptateur : le port `Compilateur` tenu par clang++ (ou g++) du poste.

Deux accommodations du poste, dites ici et non cachees : si l'edition de
liens echoue, elle est retentee avec le SDK de repli de la macOS des
outils en ligne de commande ; si elle echoue encore, `EditionDeLiensImpossible`
est levee avec le message de l'editeur de liens. Sans compilateur,
`FileNotFoundError` -- les tests qui en dependent se suspendent en le disant.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from polytech.algorithmique.domain.compilateur import (
    Diagnostic,
    EditionDeLiensImpossible,
)
from polytech.algorithmique.infrastructure.compilation import (
    CANDIDATS,
    DELAI,
    compilateur,
)

SDK_DE_REPLI: Final = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk")
STANDARD: Final = "-std=c++17"


def compilateur_disponible(candidats: tuple[str, ...] = CANDIDATS) -> bool:
    """Vrai si un compilateur C++ est present sur le poste."""
    return compilateur(candidats) is not None


@dataclass(frozen=True, slots=True)
class CompilateurClang:
    """Le compilateur du poste, derriere le port `Compilateur`."""

    candidats: tuple[str, ...] = CANDIDATS

    def _outil(self) -> str:
        outil = compilateur(self.candidats)
        if outil is None:
            msg = f"aucun compilateur C++ parmi {self.candidats}"
            raise FileNotFoundError(msg)
        return outil

    def diagnostiquer(self, source: str) -> Diagnostic:
        """Verifie la seule syntaxe : l'ecart se joue a l'analyse, non au lien."""
        outil = self._outil()
        with tempfile.TemporaryDirectory() as dossier:
            fichier = Path(dossier) / "source.cpp"
            fichier.write_text(source, encoding="utf-8")
            resultat = _lancer([outil, STANDARD, "-fsyntax-only", str(fichier)])
        return Diagnostic(accepte=resultat.returncode == 0, messages=resultat.stderr)

    def executer(self, source: str) -> str:
        """Compile, lie (avec repli sur le SDK), lance, rend la sortie."""
        outil = self._outil()
        diagnostic = self.diagnostiquer(source)
        if not diagnostic.accepte:
            msg = f"compilation refusee : {diagnostic.messages}"
            raise RuntimeError(msg)
        with tempfile.TemporaryDirectory() as dossier:
            fichier, binaire = Path(dossier) / "exec.cpp", Path(dossier) / "exec"
            fichier.write_text(source, encoding="utf-8")
            commande = [outil, STANDARD, str(fichier), "-o", str(binaire)]
            _lier(commande)
            execution = subprocess.run(  # noqa: S603 -- le binaire qu'on vient de produire
                [str(binaire)],
                capture_output=True,
                text=True,
                timeout=DELAI,
                check=False,
            )
        if execution.returncode != 0:
            msg = f"le programme a rendu le code {execution.returncode}"
            raise RuntimeError(msg)
        return execution.stdout


def _lancer(
    commande: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- commande batie ici, sans shell
        commande, capture_output=True, text=True, timeout=DELAI, check=False, env=env
    )


def _lier(commande: list[str]) -> None:
    lien = _lancer(commande)
    if lien.returncode != 0 and SDK_DE_REPLI.exists():
        lien = _lancer(commande, {**os.environ, "SDKROOT": str(SDK_DE_REPLI)})
    if lien.returncode != 0:
        msg = f"edition de liens impossible : {lien.stderr.strip()[-300:]}"
        raise EditionDeLiensImpossible(msg)
