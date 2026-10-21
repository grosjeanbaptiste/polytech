"""La serie de Laurent d'une fonction dans une couronne.

Oracle : syllabus de Mathematiques III,
`MathematiquesIngenieur3/SYLLABUS/Syllabus_Feuillet_1_Analyse_complexe.pdf`,
pages 23 a 25 : theoreme 14, relation (99) ; l'exemple (106),
`f = 1 / ((z - i)(z - 3))`, developpe sur le disque `|z| < 1` (107), sur la
couronne `1 < |z| < 3` et a l'exterieur `|z| > 3` ; et (108),
`1 / (1 - z) = - somme_(k >= 1) z^(-k)` pour `|z| > 1`. Les examens : janvier
2023, question 2 (`1 / ((z - 1)(z + 4i))` dans `2 < |z| < 3`), janvier 2026,
question 3 a (`e^z / (z + 1)^2` autour de `-1`).
"""

from __future__ import annotations

import cmath

import pytest
import sympy as sp

from polytech.maths.domain.analyse_complexe.couronne import Couronne

z = sp.Symbol("z")
F = 1 / ((z - sp.I) * (z - 3))
A = 1 / (sp.I - 3)


@pytest.mark.parametrize("k", [0, 1, 4])
def test_l_exemple_106_sur_le_disque(k: int) -> None:
    """(107) : `a_k = (i - 3)^-1 (3^-(k+1) - i^-(k+1))`."""
    attendu = A * (sp.Integer(3) ** -(k + 1) - sp.I ** -(k + 1))
    assert sp.simplify(Couronne(0, 0, 1).coefficient(F, z, k) - attendu) == 0


@pytest.mark.parametrize("k", [-3, -1, 0, 2])
def test_l_exemple_106_sur_la_couronne_c1(k: int) -> None:
    """`a_k = (i - 3)^-1 3^-(k+1)` pour `k >= 0`, `(i - 3)^-1 i^-(k+1)` sinon."""
    attendu = A * (sp.Integer(3) ** -(k + 1) if k >= 0 else sp.I ** -(k + 1))
    assert sp.simplify(Couronne(0, 1, 3).coefficient(F, z, k) - attendu) == 0


@pytest.mark.parametrize("k", [-4, -2, -1, 0, 3])
def test_l_exemple_106_a_l_exterieur(k: int) -> None:
    """Aucun terme d'indice positif ; `(i - 3)^-1 (i^-(k+1) - 3^-(k+1))` sinon."""
    attendu = 0 if k >= 0 else A * (sp.I ** -(k + 1) - sp.Integer(3) ** -(k + 1))
    assert sp.simplify(Couronne(0, 3, sp.oo).coefficient(F, z, k) - attendu) == 0


def test_l_un_sur_un_moins_z_hors_du_disque() -> None:
    """(108) : `-1` pour chaque puissance negative, rien d'autre."""
    g = 1 / (1 - z)
    assert [Couronne(0, 1, sp.oo).coefficient(g, z, k) for k in range(-3, 2)] == [
        -1,
        -1,
        -1,
        0,
        0,
    ]


def test_la_serie_de_janvier_2023_vaut_f_dans_toute_la_couronne() -> None:
    """La serie de `2 < |z| < 3` vaut `f` jusqu'a `1 < |z| < 4`."""
    f = 1 / ((z - 1) * (z + 4 * sp.I))
    serie = Couronne(0, 2, 3).somme_partielle(f, z, -60, 60)
    for point in (1.5, 2.5j, -3.0 + 0.2j):
        attendue = complex(f.subs(z, point))
        assert cmath.isclose(complex(serie.subs(z, point)), attendue, rel_tol=1e-6)


def test_le_developpement_de_janvier_2026_autour_de_moins_un() -> None:
    """`e^z / (z + 1)^2 = e^-1 (1/(z+1)^2 + 1/(z+1) + somme (z+1)^n / (n+2)!)`."""
    f = sp.exp(z) / (z + 1) ** 2
    assert Couronne(-1, 0, sp.oo).coefficient(f, z, -2) == sp.exp(-1)
    assert Couronne(-1, 0, sp.oo).coefficient(f, z, -1) == sp.exp(-1)
    for n in (0, 1, 3):
        attendu = sp.exp(-1) / sp.factorial(n + 2)
        assert sp.simplify(Couronne(-1, 0, sp.oo).coefficient(f, z, n) - attendu) == 0


def test_une_couronne_qui_contient_un_pole_est_refusee() -> None:
    with pytest.raises(ValueError, match="singularite dans la couronne"):
        Couronne(0, 2, 4).coefficient(F, z, 0)


def test_une_infinite_de_singularites_est_refusee() -> None:
    """`1 / sin z` : sympy ne les enumere pas, aucun resultat ne passe."""
    with pytest.raises(ValueError, match="nombre infini ou inconnu"):
        Couronne(0, 0, 1).coefficient(1 / sp.sin(z), z, 0)


def test_la_somme_partielle_va_jusqu_a_sa_borne() -> None:
    """La serie geometrique sur le disque : `1 + z + z^2`."""
    assert (
        sp.expand(Couronne(0, 0, 1).somme_partielle(1 / (1 - z), z, 0, 2))
        == 1 + z + z**2
    )
