"""Seconds chemins pour les modeles construits d'apres une figure.

Cause : le modele de 0.26 (Signaux) venait d'une lecture d'image fausse que
rien ne recoupait. Ici, chaque modele est confronte au TEXTE converti du
document (lu seulement, jamais reecrit), sans passer par la lecture de la figure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sympy as sp

from polytech.signaux.application import resolus_graphes as gr
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.thermique.application.enonce_tp_02 import aires_du_four

RACINE = Path(__file__).resolve().parent.parent / "cours-markdown"
RESOLUS = RACINE / "SignauxSystemes/EXS/Exs_01_02_04_05_Resolus.md"
CAMION = RACINE / "ThermiqueFondements/TP/TP_02_Camion_refrigere_Solution.md"
STEFAN = 5.670374419e-8


def _texte(chemin: Path) -> str:
    if not chemin.exists():
        pytest.skip(f"texte converti absent : {chemin}")
    return chemin.read_text(encoding="utf-8")


def _polynome(coefficients: str) -> sp.Expr:
    valeurs = [int(c) for c in coefficients.split()]
    return sum(c * P ** (len(valeurs) - 1 - k) for k, c in enumerate(valeurs))


@pytest.mark.parametrize("exercice", ["0.25", "0.27", "0.28"])
def test_les_graphes_de_mason_redonnent_l_appel_matlab_du_texte(exercice: str) -> None:
    """Le `pzmap([num], [den])` imprime dit `H(p)` sans lire le graphe."""
    texte = _texte(RESOLUS)
    solutions = texte[texte.index("Solutions") :]
    debut = solutions.index(f"Exercice {exercice}_")
    appel = re.search(r"pzmap\(\[([^\]]+)\],\s*\[([^\]]+)\]\)", solutions[debut:])
    assert appel is not None
    imprimee = _polynome(appel.group(1)) / _polynome(appel.group(2))
    assert sp.simplify(gr.transmittance(exercice) - imprimee) == 0


def test_0_26_le_modele_redonne_la_reponse_imprimee() -> None:
    """La solution imprimee de 0.26 est `H(p) = p/(p+1)`."""
    assert "H(p)=p/(p+1)" in _texte(RESOLUS)
    assert sp.simplify(gr.transmittance("0.26") - P / (P + 1)) == 0


def test_t18_les_nombres_imprimes_du_four_se_contredisent_sans_le_schema() -> None:
    """`J1 - J2 = phi / S_noir` est la definition de l'echange : pas de lecture d'image.

    Avec les trois nombres imprimes (J1, phi, S noir) on obtient `J2 = 42 836`,
    et non les 41 900 imprimes ; le T3 imprime est pourtant celui de ces 41 900.
    """
    texte = _texte(CAMION)
    for imprime in ("101625", "86266", "4.19 x 104", "1074.5"):
        assert imprime in texte
    j1, phi, s_noir = 86266.0, 101625.0, 2.34
    assert j1 - phi / s_noir == pytest.approx(42836.0, abs=1.0)
    _, s13, s23 = aires_du_four()
    moyenne = (s13 * j1 + s23 * 41900.0) / (s13 + s23)
    assert (moyenne / STEFAN) ** 0.25 == pytest.approx(1074.5, abs=0.2)
