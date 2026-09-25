"""Les enonces de lois du cours sont portes par la bibliotheque.

Ce ne sont pas des resultats a calculer, mais le document ne doit pas les
recopier pour autant : ils sont construits ici, une seule fois.
"""

import sympy as sp

from polytech.fluides.domain.cinematique.champ_generique import CHAMP_GENERIQUE
from polytech.fluides.domain.cinematique.lignes import definition_ligne_de_courant
from polytech.fluides.domain.cinematique.symboles import T, X, Y
from polytech.fluides.domain.statique.stevin import (
    loi_de_stevin,
    relation_entre_deux_points,
)

U = sp.Function("u")(X, Y, T)
V = sp.Function("v")(X, Y, T)


def test_la_loi_de_stevin_est_une_egalite_du_cours() -> None:
    """P + rho g z = constante (cours II.2)."""
    loi = loi_de_stevin()
    assert sp.Symbol("rho", positive=True) * sp.Symbol("g", positive=True) in [
        t.as_coeff_Mul()[1] * sp.Symbol("g", positive=True) for t in loi.lhs.args
    ] or loi.lhs.has(sp.Symbol("rho", positive=True))


def test_la_relation_entre_deux_points_ajoute_le_poids_de_la_colonne() -> None:
    relation = relation_entre_deux_points()
    assert relation.lhs == sp.Symbol("p_bas")
    assert relation.rhs.has(sp.Symbol("p_haut"))


def test_la_divergence_generique_est_la_definition_du_cours() -> None:
    """Le meme operateur que pour un champ concret, applique a (u, v)."""
    attendu = sp.Derivative(U, X) + sp.Derivative(V, Y)
    assert sp.simplify(CHAMP_GENERIQUE.divergence() - attendu) == 0


def test_l_acceleration_generique_est_la_derivee_particulaire() -> None:
    ax, _ = CHAMP_GENERIQUE.acceleration()
    attendu = sp.Derivative(U, T) + U * sp.Derivative(U, X) + V * sp.Derivative(U, Y)
    assert sp.simplify(ax - attendu) == 0


def test_la_definition_d_une_ligne_de_courant_egalise_les_deux_rapports() -> None:
    definition = definition_ligne_de_courant()
    assert definition.lhs.has(sp.Symbol("dx"))
    assert definition.rhs.has(sp.Symbol("dy"))
