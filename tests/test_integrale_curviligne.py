"""L'integrale d'une fonction complexe le long d'une courbe.

Oracle : syllabus de Mathematiques III,
`MathematiquesIngenieur3/SYLLABUS/Syllabus_Feuillet_1_Analyse_complexe.pdf`,
page 8 : definition 15, relation (27), `int_gamma f dz = int_a^b f(gamma(t))
gamma'(t) dt` ; remarque 1, l'independance vis-a-vis du parametrage ;
remarque 2, relation (28), `int_gamma F' dz = F(gamma(b)) - F(gamma(a))`.
Et les deux cas des examens : aout 2023, question 3 (`conj(z)` sur le
cercle de centre `2i`), examen blanc du 3 decembre 2025, question 1 (b) (le
cercle parcouru deux fois).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_complexe.integrales import (
    integrale_curviligne,
    integrale_curviligne_numerique,
)
from polytech.maths.domain.analyse_complexe.residus import residu

z, t = sp.symbols("z t")
THETA = sp.Symbol("theta", real=True)


def _cercle(centre: object, rayon: object, tours: int = 1) -> sp.Expr:
    return sp.sympify(centre) + rayon * sp.exp(sp.I * tours * THETA)


def test_un_sur_z_sur_le_cercle_unite_vaut_deux_pi_i() -> None:
    """Relation (27), le calcul de reference."""
    assert (
        integrale_curviligne(1 / z, _cercle(0, 1), THETA, 0, 2 * sp.pi)
        == 2 * sp.pi * sp.I
    )


def test_le_parametrage_ne_change_pas_l_integrale() -> None:
    """Remarque 1 : le meme segment, parcouru a deux vitesses."""
    tau = sp.Symbol("tau", real=True)
    lent = integrale_curviligne(z**2, (1 + sp.I) * tau, tau, 0, 1)
    rapide = integrale_curviligne(z**2, (1 + sp.I) * tau**2, tau, 0, 1)
    assert sp.simplify(lent - rapide) == 0


def test_une_derivee_s_integre_par_ses_extremites() -> None:
    """Relation (28) : `F = z^3` le long d'une parabole, de 0 a 1 + i."""
    tau = sp.Symbol("tau", real=True)
    valeur = integrale_curviligne(3 * z**2, tau + sp.I * tau**2, tau, 0, 1)
    assert sp.simplify(valeur - (1 + sp.I) ** 3) == 0


def test_le_conjugue_sur_le_cercle_d_aout_2023() -> None:
    """`int conj(z) dz = 2 pi i rho^2` sur le cercle de centre `2i`."""
    rho = sp.Symbol("rho", positive=True)
    valeur = integrale_curviligne(
        sp.conjugate(z), _cercle(2 * sp.I, rho), THETA, -sp.pi, sp.pi
    )
    assert sp.simplify(valeur - 2 * sp.pi * sp.I * rho**2) == 0


def test_le_cercle_parcouru_deux_fois_de_l_examen_blanc() -> None:
    """`1 / (z + z^2)` sur `1 + i + 2i e^(2 i theta)`, `theta` dans ]-pi, pi[."""
    courbe = 1 + sp.I + 2 * sp.I * sp.exp(2 * sp.I * THETA)
    valeur = integrale_curviligne_numerique(
        1 / (z + z**2), courbe, THETA, -sp.pi, sp.pi
    )
    assert abs(valeur - 4j * float(sp.pi)) < 1e-9


def test_une_integrale_que_sympy_ne_ferme_pas_leve() -> None:
    """Aucun resultat partiel ne passe pour une valeur."""
    g = sp.Function("g")  # indeterminee : sympy rend l'integrale telle quelle
    with pytest.raises(ValueError, match="pas de forme close"):
        integrale_curviligne(g(z), _cercle(0, 1), THETA, 0, 2 * sp.pi)


def test_le_conjugue_sur_le_cercle_parcouru_deux_fois_de_l_examen_blanc() -> None:
    """Question 1 (b)(ii) : deux tours d'un cercle de rayon 2, `16 pi i`."""
    courbe = 1 + sp.I + 2 * sp.I * sp.exp(2 * sp.I * THETA)
    valeur = integrale_curviligne(sp.conjugate(z), courbe, THETA, -sp.pi, sp.pi)
    assert valeur == 16 * sp.pi * sp.I
    # Second chemin : sur le cercle, `conj(z) = conj(c) + r^2 / (z - c)`.
    centre = 1 + sp.I
    sur_le_cercle = sp.conjugate(centre) + 4 / (z - centre)
    assert 2 * 2 * sp.pi * sp.I * residu(sur_le_cercle, z, centre) == valeur
