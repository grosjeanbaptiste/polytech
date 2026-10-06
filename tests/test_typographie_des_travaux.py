"""Aucun travail ne compose un caractere que sa police remplace en silence.

Sous lualatex, une police T1 a 256 positions. Les guillemets francais et
le signe de paragraphe, ecrits tels quels, n'y sont pas perdus : ils
tombent sur la position de meme numero, qui porte une autre lettre. Le
PDF compose alors `n accent aigu ... z point suscrit` autour de chaque
citation, et `g breve` pour un paragraphe -- sans aucun avertissement.

`tests/test_coherence_des_corriges.py` le gardait pour les seuls
corriges d'electromagnetisme. `AGENTS.md` nommait la dette pour les deux
autres matieres : 53 lignes dans l'activite d'integration de mecanique
des fluides, 14 dans le TP 1 de thermique. Cette garde vaut pour tout
`travaux/`, et ne regarde que ce qui se compose : un commentaire peut
porter ce qu'il veut.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

TRAVAUX: Final = Path(__file__).resolve().parent.parent / "travaux"

REMPLACES: Final = {
    "«": r"\og",
    "»": r"\fg{}",
    "§": r"\S",
    "µ": r"\textmu{} ou \micro",
    "°": r"\textdegree{}, ^\circ ou \degree",
}
"""Le caractere que la police remplace, et la commande qui le compose."""

COMMENTAIRE: Final = re.compile(r"(?<!\\)%.*$")


def _sources() -> list[Path]:
    return sorted(TRAVAUX.rglob("*.tex"))


def _composees(source: Path, caractere: str) -> list[int]:
    lignes = source.read_text(encoding="utf-8").splitlines()
    return [
        rang
        for rang, ligne in enumerate(lignes, 1)
        if caractere in COMMENTAIRE.sub("", ligne)
    ]


def test_la_garde_porte_sur_des_sources() -> None:
    """Un dossier vide rendrait la garde verte sans rien garder."""
    assert _sources()


def test_un_commentaire_n_est_pas_compose() -> None:
    """Le `%` ouvre un commentaire, sauf echappe par une contre-oblique."""
    assert COMMENTAIRE.sub("", "texte % « cite »") == "texte "
    assert COMMENTAIRE.sub("", r"50 \% « cite »") == r"50 \% « cite »"


@pytest.mark.parametrize("caractere", sorted(REMPLACES))
def test_aucune_source_ne_compose_un_caractere_remplace(caractere: str) -> None:
    """Chaque occurrence est nommee par son fichier et sa ligne."""
    fautes = {
        str(source.relative_to(TRAVAUX)): rangs
        for source in _sources()
        if (rangs := _composees(source, caractere))
    }
    assert fautes == {}, f"ecrire {REMPLACES[caractere]} a la place de {caractere!r}"


CHEMIN_A_BLANC: Final = re.compile(r"\\chemin\{[^}]*\s[^}]*\}")
r"""Un `\chemin{...}` dont l'argument porte un blanc.

`\chemin` repose sur `\nolinkurl`, qui avale les blancs : la commande
`python -m polytech.electronique` sortait `python-mpolytech.electronique`
dans vingt-trois corriges, sans aucun avertissement. Un chemin n'a pas de
blanc ; une commande se compose en `\texttt`.
"""


def test_un_chemin_avec_blanc_est_reconnu() -> None:
    """La garde voit la faute, et ne s'alarme pas d'un vrai chemin."""
    assert CHEMIN_A_BLANC.search(r"\chemin{python -m polytech.signaux}")
    assert not CHEMIN_A_BLANC.search(r"\chemin{polytech.signaux} et \chemin{a/b.py}")


def test_aucune_commande_n_est_composee_comme_un_chemin() -> None:
    """Sinon ses blancs disparaissent du PDF."""
    fautes = {
        str(source.relative_to(TRAVAUX)): rangs
        for source in _sources()
        if (
            rangs := [
                rang
                for rang, ligne in enumerate(
                    source.read_text(encoding="utf-8").splitlines(), 1
                )
                if CHEMIN_A_BLANC.search(COMMENTAIRE.sub("", ligne))
            ]
        )
    }
    assert fautes == {}, r"ecrire \texttt{...} pour une commande"
