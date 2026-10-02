"""Le README decompose son total ; la decomposition doit tenir aussi.

Le nombre total de documents LaTeX est garde depuis longtemps. La phrase
qui le **decompose** -- « les quatre de l'activite d'integration, celui
du TP 1, les neuf corriges d'electromagnetisme, et les 398 planches » --
ne l'etait pas, et elle a derive deux fois pendant la campagne : chaque
corrige ajoute change un terme de la somme, et le terme est ecrit en
lettres.

Une somme qui ne fait pas son total est pire qu'une somme absente : elle
donne au lecteur un moyen de verifier, et ce moyen mentirait.

Le test porte sur le terme qui bouge -- le compte des corriges
d'electromagnetisme -- et sur la somme entiere. Les trois autres termes
sont stables, mais la somme les attrape si l'un change.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

RACINE = Path(__file__).resolve().parent.parent
README = RACINE / "README.md"
TRAVAUX = RACINE / "travaux" / "Electromagnetisme"

NUMERAUX: Final = {
    "un": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "dix-sept": 17,
    "dix-huit": 18,
    "dix-neuf": 19,
    "vingt": 20,
    "vingt et un": 21,
}
"""De un a vingt et un : la campagne compte vingt et un exercices au plus."""

ANNONCE = re.compile(r"\*\*(\d+) documents LaTeX\*\*")
DECOMPOSITION = re.compile(
    r"les quatre de l'activité\s+d'intégration de mécanique des fluides, "
    r"celui du TP 1 de thermique, les\s+([\w-]+(?: et un)?) corrigés "
    r"d'électromagnétisme déjà écrits, et les (\d+) planches",
)


def _corriges_d_electromagnetisme() -> int:
    """Les dossiers de travail qui portent un document compose."""
    return sum(
        1
        for dossier in TRAVAUX.iterdir()
        if dossier.is_dir()
        and any(f.name != "resultats.tex" for f in dossier.glob("*.tex"))
    )


def test_la_decomposition_est_bien_presente_dans_le_readme() -> None:
    """Sans elle, les deux tests suivants passeraient a vide.

    C'est le piege des tests qui s'appuient sur une recherche : si le
    motif ne trouve rien, l'assertion suivante peut tenir par vacuite.
    """
    assert DECOMPOSITION.search(README.read_text(encoding="utf-8")) is not None


def test_le_nombre_de_corriges_annonce_en_lettres_est_exact() -> None:
    """« neuf corrigés d'électromagnétisme » doit en compter neuf."""
    trouve = DECOMPOSITION.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    assert NUMERAUX[trouve.group(1)] == _corriges_d_electromagnetisme()


def test_la_decomposition_fait_bien_le_total_annonce() -> None:
    """Quatre, plus un, plus les corriges, plus les planches.

    C'est le controle que le lecteur ferait, et il doit aboutir.
    """
    texte = README.read_text(encoding="utf-8")
    total = ANNONCE.search(texte)
    trouve = DECOMPOSITION.search(texte)
    assert total is not None
    assert trouve is not None
    somme = 4 + 1 + NUMERAUX[trouve.group(1)] + int(trouve.group(2))
    assert somme == int(total.group(1))
