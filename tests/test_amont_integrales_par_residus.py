"""Mathematiques III, section 9 partie 2 : une integrale reelle par les residus.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ce test
verifie que le symbole auquel `cursus/notions/MathematiquesIngenieur3.toml`
rattache l'enonce de la page 18 le porte bien.

Source : `MathematiquesIngenieur3/SLIDES/`
`Slides_09_Calcul_des_residus_partie_2_Calcul_d_integrales.pdf`, page 18 :
si `q` n'a aucune racine reelle et `deg q >= deg p + 2`, l'integrale de
`p / q` sur la droite vaut `2 pi i` fois la somme des residus aux
singularites du demi-plan superieur.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.analyse_complexe.residus import integrale_par_les_residus

Z = sp.Symbol("z")
X = sp.Symbol("x", real=True)


def _du_demi_plan_superieur(denominateur: sp.Expr) -> list[sp.Expr]:
    return [racine for racine in sp.roots(denominateur, Z) if sp.im(racine) > 0]


@pytest.mark.parametrize(
    ("numerateur", "denominateur"),
    [(sp.Integer(1), Z**2 + 1), (sp.Integer(1), Z**4 + 1), (Z**2, Z**4 + 5 * Z**2 + 4)],
    ids=["1/(1+x2)", "1/(1+x4)", "x2/((x2+1)(x2+4))"],
)
def test_l_integrale_sur_la_droite_est_la_somme_des_residus_du_demi_plan(
    numerateur: sp.Expr, denominateur: sp.Expr
) -> None:
    """Page 18 : par les residus, puis par une primitive -- deux chemins."""
    fonction = numerateur / denominateur
    poles = _du_demi_plan_superieur(denominateur)
    par_les_residus = integrale_par_les_residus(fonction, Z, poles)
    directe = sp.integrate(fonction.subs(Z, X), (X, -sp.oo, sp.oo))
    assert sp.simplify(par_les_residus - directe) == 0
    assert sp.im(par_les_residus) == 0


def test_le_demi_plan_inferieur_donne_l_oppose() -> None:
    """Le contour se referme dans l'autre sens : le demi-plan choisi compte."""
    par_le_bas = integrale_par_les_residus(1 / (Z**2 + 1), Z, [-sp.I])
    assert par_le_bas == -sp.pi
