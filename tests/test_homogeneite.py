"""Les resultats de la partie II doivent etre homogenes, pas seulement annonces.

Le corrige pose une convention d'unites -- « les coefficients valent 1 mais
portent les dimensions qui rendent les expressions homogenes » -- et n'en
verifiait jamais la consequence. Ces tests la verifient sur chaque resultat
publie, et exhibent le piege du champ B : ses deux composantes ne portent
pas le meme coefficient.
"""

import pytest
import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.dimensions import (
    ACCELERATION,
    TAUX,
    TEMPS,
    VITESSE,
    DimensionIndeterminee,
    ExpressionNonHomogene,
    dimension_de,
    dimension_des_contributions,
)

COEFF_B_U = TEMPS**-2
"""u = xt : pour que ce soit une vitesse, le coefficient est en s^-2."""
COEFF_B_V = TEMPS**-1
"""v = x - y : une longueur, donc un coefficient en s^-1."""
COEFF_C = TEMPS**-1


def test_les_deux_composantes_de_b_sont_bien_des_vitesses() -> None:
    """C'est ce qui impose deux coefficients differents, et non un seul."""
    assert dimension_de(enonce.CHAMP_B.u, COEFF_B_U) == VITESSE
    assert dimension_de(enonce.CHAMP_B.v, COEFF_B_V) == VITESSE


def test_les_deux_composantes_de_c_sont_des_vitesses() -> None:
    """Ici un seul coefficient suffit : les deux composantes sont en x ou y."""
    assert dimension_de(enonce.CHAMP_C.u, COEFF_C) == VITESSE
    assert dimension_de(enonce.CHAMP_C.v, COEFF_C) == VITESSE


def test_la_divergence_de_b_est_un_taux_malgre_les_apparences() -> None:
    """« t - 1 » n'est homogene que contribution par contribution.

    du/dx porte le coefficient de u, dv/dy celui de v : les deux retombent
    sur des secondes a la puissance -1. Les additionner sous un coefficient
    unique donnerait une somme absurde.
    """
    derivees = enonce.CHAMP_B.derivees_partielles()
    obtenu = dimension_des_contributions(
        [(derivees.du_dx, COEFF_B_U), (derivees.dv_dy, COEFF_B_V)],
    )
    assert obtenu == TAUX


def test_une_divergence_dimensionnee_d_un_bloc_est_refusee() -> None:
    """Le piege, exhibe : c'est bien une erreur si l'on ignore l'origine."""
    with pytest.raises(ExpressionNonHomogene, match="non homogene"):
        dimension_de(enonce.CHAMP_B.divergence(), COEFF_B_U)


def test_le_taux_de_rotation_de_b_est_un_taux() -> None:
    derivees = enonce.CHAMP_B.derivees_partielles()
    obtenu = dimension_des_contributions([(derivees.dv_dx, COEFF_B_V)])
    assert obtenu == TAUX


def test_le_champ_d_acceleration_de_b_est_une_acceleration() -> None:
    """a_x = x(t^2 + 1) : terme local et terme convectif doivent s'accorder."""
    derivees = enonce.CHAMP_B.derivees_partielles()
    locale = (derivees.du_dt, COEFF_B_U)
    convective = (enonce.CHAMP_B.u * derivees.du_dx, COEFF_B_U**2)
    assert dimension_des_contributions([locale, convective]) == ACCELERATION


def test_le_champ_d_acceleration_de_c_est_une_acceleration() -> None:
    ax, ay = enonce.CHAMP_C.acceleration()
    assert dimension_de(ax, COEFF_C**2) == ACCELERATION
    assert dimension_de(ay, COEFF_C**2) == ACCELERATION


def test_le_tenseur_de_c_est_homogene_en_taux() -> None:
    """Les coefficients non nuls du tenseur sont tous des taux."""
    tenseur = enonce.CHAMP_C.tenseur_taux_deformation()
    non_nuls = [(c, COEFF_C) for c in tenseur if c != 0]
    assert dimension_des_contributions(non_nuls) == TAUX


def test_une_somme_inhomogene_est_refusee() -> None:
    """X + t ne veut rien dire : une longueur plus un temps."""
    with pytest.raises(ExpressionNonHomogene, match="non homogene"):
        dimension_de(
            sp.Symbol("x", real=True) + sp.Symbol("t", real=True), sp.Integer(1)
        )


def test_une_expression_nulle_n_a_pas_de_dimension() -> None:
    """omega_z = 0 pour l'ecoulement C : rien a verifier, et on le dit."""
    with pytest.raises(DimensionIndeterminee, match="indeterminee"):
        dimension_de(enonce.CHAMP_C.taux_de_rotation(), COEFF_C)
