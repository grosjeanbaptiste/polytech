"""La loi de l'hydrostatique est *etablie*, pas posee.

Le corrige affirmait « dp/dz = -rho g s'integre en Delta p = -rho g Delta z ».
Ces tests exigent que les deux etapes -- le bilan des forces sur une tranche,
puis son integration -- soient reellement menees par la bibliotheque.
"""

import sympy as sp

from polytech.fluides.domain.statique.bosse import (
    HAUTEUR_BOSSE,
    compensation_de_la_bosse,
    contributions_de_la_bosse,
)
from polytech.fluides.domain.statique.equilibre import (
    SECTION,
    bilan_des_forces,
    developpement_au_premier_ordre,
    integration_sur_une_colonne,
    loi_locale,
)
from polytech.fluides.domain.statique.stevin import (
    expression_variation_de_pression,
)
from polytech.fluides.domain.statique.symboles import PROFONDEUR, RHO, G, Z

DENIVELEE = sp.Symbol("h_2", positive=True)


def test_le_bilan_des_forces_compte_les_deux_faces_et_le_poids() -> None:
    """Poussee du dessous, poussee du dessus, poids : trois termes."""
    bilan = bilan_des_forces()
    assert bilan.has(RHO * G * SECTION)
    assert len(bilan.args) == 3


def test_le_premier_ordre_fait_apparaitre_la_derivee_de_la_pression() -> None:
    """Le developpement de p(z + dz) est ce qui produit dp/dz."""
    assert developpement_au_premier_ordre().has(sp.Derivative)


def test_l_equilibre_donne_la_loi_locale_du_cours() -> None:
    """Le bilan nul equivaut a dp/dz + rho g = 0."""
    pression = sp.Function("p")
    attendu = sp.diff(pression(Z), Z) + RHO * G
    assert sp.simplify(loi_locale().lhs - attendu) == 0
    assert loi_locale().rhs == 0


def test_l_integration_de_la_loi_locale_donne_la_loi_de_stevin() -> None:
    """Entre deux altitudes, seule la difference d'altitude intervient."""
    equation = integration_sur_une_colonne()
    z_bas, z_haut = sp.Symbol("z_1"), sp.Symbol("z_2")
    assert sp.simplify(equation.rhs - (-RHO * G * (z_haut - z_bas))) == 0


def test_la_bosse_apporte_une_montee_puis_une_descente() -> None:
    """Un U inverse se parcourt en deux troncons de signes opposes."""
    montee, descente = contributions_de_la_bosse(DENIVELEE)
    assert sp.simplify(montee + RHO * G * HAUTEUR_BOSSE) == 0
    assert sp.simplify(descente - RHO * G * (HAUTEUR_BOSSE + DENIVELEE)) == 0


def test_la_hauteur_de_la_bosse_disparait_de_la_somme() -> None:
    """C'est la seule vraie subtilite du manometre : elle doit etre prouvee."""
    equation = compensation_de_la_bosse(DENIVELEE)
    assert sp.simplify(equation.rhs - RHO * G * DENIVELEE) == 0
    assert not equation.rhs.has(HAUTEUR_BOSSE)


def test_la_forme_litterale_du_poids_de_colonne_est_publiable() -> None:
    """`VariationPression` est citee par le corrige sans test propre.

    Elle doit rendre l'egalite du cours, Delta p = rho g h, sous forme
    non evaluee afin que le document l'affiche telle quelle.
    """
    equation = expression_variation_de_pression()
    assert sp.simplify(equation.rhs - RHO * G * PROFONDEUR) == 0
    assert equation.lhs.name == r"\Delta p"
