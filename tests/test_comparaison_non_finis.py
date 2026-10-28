"""Un nan ou un inf n'est pas une mesure : toute comparaison le refuse en levant."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import (
    egales,
    egales_arrondies,
    eloignees,
    est_negative,
    est_positive,
    proches,
    proches_relativement,
    rapport_au_dessus,
    suite_nulle,
    suites_proches,
)

if TYPE_CHECKING:
    from collections.abc import Callable

NAN, INF = math.nan, math.inf
NON_FINIS = pytest.mark.parametrize("anormal", [NAN, INF, -INF], ids=str)

CAS: dict[str, Callable[[float], object]] = {
    "egales/obtenue": lambda a: egales(a, 1),
    "egales/attendue": lambda a: egales(1, a),
    "egales/deux_inf": lambda a: egales(a, a),
    "egales/matrice": lambda a: egales(sp.Matrix([a]), sp.Matrix([1])),
    "proches/valeur": lambda a: proches(a, 1, 0.1),
    "proches/reference": lambda a: proches(1, a, 0.1),
    "proches/tolerance": lambda a: proches(1, 1, a),
    "eloignees/valeur": lambda a: eloignees(a, 1, 0.1),
    "eloignees/reference": lambda a: eloignees(1, a, 0.1),
    "relativement/valeur": lambda a: proches_relativement(a, 1, 0.1),
    "relativement/reference": lambda a: proches_relativement(1, a, 0.1),
    "suites_proches": lambda a: suites_proches([a], [1.0], 0.1),
    "suite_nulle": lambda a: suite_nulle([0.0, a], 0.1),
    "rapport/valeur": lambda a: rapport_au_dessus(a, 1.0, 2.0),
    "rapport/reference": lambda a: rapport_au_dessus(1.0, a, 2.0),
    "rapport/facteur": lambda a: rapport_au_dessus(1.0, 1.0, a),
    "arrondies/valeur": lambda a: egales_arrondies(a, 1.0, 2),
    "arrondies/attendue": lambda a: egales_arrondies(1.0, a, 2),
    "positive": est_positive,
    "negative": est_negative,
}


@pytest.mark.parametrize("nom", CAS)
@NON_FINIS
def test_un_non_fini_leve(nom: str, anormal: float) -> None:
    with pytest.raises(ValueError, match="non fini"):
        CAS[nom](anormal)


def test_un_nan_symbolique_leve_aussi() -> None:
    with pytest.raises(ValueError, match="non fini"):
        egales(sp.nan, 1)
    with pytest.raises(ValueError, match="non fini"):
        egales(sp.oo, sp.oo)


def test_les_valeurs_finies_passent_toujours() -> None:
    assert egales(1, 1)
    assert proches(1, 1.05, 0.1)
    assert eloignees(1, 2, 0.1)
    assert est_positive(1)
    assert est_negative(-1)


def test_un_nan_apres_un_terme_faux_leve_quand_meme() -> None:
    """`all` court-circuiterait sur le premier terme faux et taurait le nan."""
    with pytest.raises(ValueError, match="non fini"):
        suite_nulle([5.0, NAN], 0.1)
    with pytest.raises(ValueError, match="non fini"):
        suites_proches([9.0, NAN], [1.0, 1.0], 0.1)
