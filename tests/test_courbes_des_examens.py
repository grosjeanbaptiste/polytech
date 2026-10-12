"""Les figures des corriges d'examen : des coordonnees, et une aire recoupee.

Chaque integrale definie d'un examen est dessinee comme l'aire qu'elle
mesure. L'aire est alors recalculee par quadrature numerique (`scipy`), un
chemin qui n'emploie ni primitive ni decomposition : il recoupe la valeur
exacte que le corrige etablit.
"""

from __future__ import annotations

import math
import re

import pytest
import sympy as sp

from polytech.maths.application import courbes_examens as courbes
from polytech.maths.application import examen_2021_analyse as an
from polytech.maths.application import examen_2022, examen_2025
from polytech.maths.application.macros_examen_2021 import macros_de_l_examen_2021
from polytech.maths.application.macros_examen_2022 import macros_de_l_examen_2022
from polytech.maths.application.macros_examen_2025 import macros_de_l_examen_2025
from polytech.maths.domain.symboles import VARIABLE as X

POINT = re.compile(r"\((-?[\d.]+),(-?[\d.]+)\)")


def _points(coordonnees: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in POINT.findall(coordonnees)]


def test_l_aire_de_2022_va_de_un_a_deux() -> None:
    points = _points(courbes.integrande_2022())
    assert (points[0][0], points[-1][0]) == (1.0, 2.0)
    assert points[0][1] == pytest.approx(math.log(2), abs=1e-3)
    assert points[-1][1] == pytest.approx(math.log(3) / 4, abs=1e-3)


def test_l_aire_de_2025_va_de_deux_a_trois() -> None:
    points = _points(courbes.integrande_2025())
    assert (points[0][0], points[-1][0]) == (2.0, 3.0)
    assert points[0][1] == pytest.approx(16 / 15, abs=1e-3)
    assert points[-1][1] == pytest.approx(36 / 80, abs=1e-3)


def test_la_quadrature_recoupe_la_valeur_exacte_de_2022() -> None:
    exacte = float(examen_2022.integrale()[2])
    assert courbes.aire_2022() == pytest.approx(exacte, abs=1e-10)


def test_la_quadrature_recoupe_la_valeur_exacte_de_2025() -> None:
    exacte = float(examen_2025.integrale()[2])
    assert courbes.aire_2025() == pytest.approx(exacte, abs=1e-10)


def test_la_solution_8_de_2021_part_de_six() -> None:
    points = _points(courbes.solution_8_de_2021())
    assert points[0] == (0.0, 6.0)
    assert all(-math.pi / 2 < x < math.pi / 2 for x, _ in points)


def test_la_primitive_k_est_reelle_des_deux_cotes_de_e() -> None:
    """`|1 - ln^2 x|` : la primitive vaut sur `]1/e, e[` comme au-dela."""
    primitive = an.primitive_k()
    assert primitive.atoms(sp.Abs)
    for point in (sp.Integer(2), sp.Integer(10)):
        assert primitive.subs(X, point).is_real
        pente = sp.diff(primitive, X).subs(X, point)
        attendue = sp.log(point) / (point * (1 - sp.log(point) ** 2))
        assert abs(sp.N(pente - attendue, 30)) < sp.Float("1e-25")


def test_les_equations_de_2021_s_ecrivent_en_t_comme_l_enonce() -> None:
    macros = macros_de_l_examen_2021()
    assert "t^{2}" in macros["ExamVingtEtUnSolutionSept"]
    for nom in ("Sept", "Huit", "Neuf"):
        for forme in ("Generale", "Solution"):
            assert "x" not in macros[f"ExamVingtEtUn{forme}{nom}"]


def test_chaque_corrige_publie_l_aire_de_son_integrale() -> None:
    """Une mutation a survecu ici : le corrige de 2025 publiait l'aire de 2022."""
    deux = float(macros_de_l_examen_2022()["ExamVingtDeuxAire"])
    cinq = float(macros_de_l_examen_2025()["ExamVingtCinqAire"])
    assert deux == pytest.approx(float(examen_2022.integrale()[2]), abs=1e-4)
    assert cinq == pytest.approx(float(examen_2025.integrale()[2]), abs=1e-4)
