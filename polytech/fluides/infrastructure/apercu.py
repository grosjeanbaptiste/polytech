r"""Adaptateur : compose un apercu lisible des macros publiees.

`resultats.tex` ne peut pas donner de PDF : c'est un *fragment*, une suite
de ``\newcommand`` sans classe de document, que les trois documents
chargent par ``\input``. Le rendre autonome demanderait d'y placer un
``\documentclass`` en tete, ce qui interdirait justement cet ``\input``.

Ce module produit donc le document qui manquait : un apercu qui charge le
fragment et compose chaque macro avec son nom, afin qu'un lecteur puisse
verifier d'un coup d'oeil toutes les valeurs que la bibliotheque calcule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_PREAMBULE: Final = (
    "% Fichier genere par `python -m polytech.fluides`"
    " -- NE PAS EDITER.\n"
    r"""
\documentclass[10pt,a4paper,landscape]{article}
\usepackage{iftex}
\ifPDFTeX\usepackage[utf8]{inputenc}\fi
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage[a4paper,landscape,margin=1.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{multicol}
\DeclareSIUnit{\bar}{bar}
\sisetup{output-decimal-marker={,},per-mode=symbol}
\input{resultats}
\pagestyle{empty}
\begin{document}
\begin{center}
  {\large\bfseries Valeurs calculees par \texttt{polytech.fluides}}\\[0.3ex]
  {\footnotesize Toutes proviennent de \texttt{resultats.tex} ---
   aucune n'est saisie a la main}
\end{center}
\vspace{1ex}
\begin{multicols}{3}
\footnotesize
"""
)

_FIN: Final = "\\end{multicols}\n\\end{document}\n"
_NUMERIQUE: Final = re.compile(r"-?[\d.]+")


@dataclass(frozen=True, slots=True)
class ApercuDesMacros:
    """Publie un document qui compose chaque macro avec son nom."""

    chemin: Path

    def publier(self, macros: Mapping[str, str]) -> None:
        """Ecrit l'apercu ; toute erreur d'ecriture se propage."""
        if not macros:
            msg = "aucune macro a composer"
            raise ValueError(msg)
        lignes = [_PREAMBULE]
        lignes.extend(self._ligne(nom) for nom in sorted(macros))
        lignes.append(_FIN)
        self.chemin.write_text("".join(lignes), encoding="utf-8")

    @staticmethod
    def _ligne(nom: str) -> str:
        """Une entree : le nom de la macro, puis ce qu'elle compose."""
        return (
            rf"\noindent\texttt{{\footnotesize {nom}}}\\"
            rf"\hspace*{{1em}}$\{nom}$\par\smallskip" + "\n"
        )
