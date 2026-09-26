"""Les definitions du cours III sont *deduites*, pas recopiees.

Le corrige posait omega_z, le tenseur des taux de deformation et la derivee
materielle comme des formules a retenir. Ces tests exigent qu'elles tombent
d'un raisonnement : decomposition du gradient de vitesse, derivation en
chaine le long d'une trajectoire, bilan sur les aretes d'une particule.
"""

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.champ_generique import CHAMP_GENERIQUE
from polytech.fluides.domain.cinematique.decomposition import (
    gradient_de_vitesse,
    partie_antisymetrique,
    partie_symetrique,
)
from polytech.fluides.domain.cinematique.evolution import image_du_cercle
from polytech.fluides.domain.cinematique.existence import (
    divergence_d_un_champ_derivant_de_psi,
    gradients_orthogonaux,
    rotation_d_un_champ_derivant_de_phi,
)
from polytech.fluides.domain.cinematique.operateurs import (
    derivee_le_long_de_la_trajectoire,
    derivee_materielle,
    divergence,
    taux_de_rotation,
    tenseur_taux_deformation,
)
from polytech.fluides.domain.cinematique.particule import (
    ECART_MATERIEL,
    allongement_d_une_arete,
    dilatation_relative,
)
from polytech.fluides.domain.cinematique.symboles import T, X, Y

U, V = CHAMP_GENERIQUE.u, CHAMP_GENERIQUE.v


def _nul(expression: sp.Expr) -> bool:
    return bool(sp.simplify(expression) == 0)


def test_le_gradient_de_vitesse_se_scinde_en_deformation_et_rotation() -> None:
    """Toute matrice est somme de sa part symetrique et de sa part antisymetrique."""
    gradient = gradient_de_vitesse(U, V)
    reconstruit = partie_symetrique(gradient) + partie_antisymetrique(gradient)
    assert _nul(sp.Matrix(gradient - reconstruit).norm())


def test_la_part_symetrique_est_le_tenseur_des_taux_de_deformation() -> None:
    """Le cours pose epsilon ; la decomposition le produit."""
    obtenu = partie_symetrique(gradient_de_vitesse(U, V))
    assert _nul(sp.Matrix(obtenu - tenseur_taux_deformation(U, V)).norm())


def test_la_part_antisymetrique_porte_le_taux_de_rotation() -> None:
    """omega_z n'est pas une formule a retenir : c'est un coefficient de Omega."""
    antisymetrique = partie_antisymetrique(gradient_de_vitesse(U, V))
    assert _nul(antisymetrique[1, 0] - taux_de_rotation(U, V))
    assert _nul(antisymetrique[0, 1] + taux_de_rotation(U, V))


def test_la_derivee_materielle_est_la_derivee_totale_le_long_du_mouvement() -> None:
    """D/Dt n'est pas une convention : c'est la derivation en chaine."""
    temperature = sp.Function("f")(X, Y, T)
    assert _nul(
        derivee_le_long_de_la_trajectoire(temperature, U, V)
        - derivee_materielle(temperature, U, V),
    )


def test_la_derivee_materielle_se_verifie_sur_un_champ_concret() -> None:
    """Meme identite, sur l'ecoulement C de l'activite."""
    champ = enonce.CHAMP_C
    assert _nul(
        derivee_le_long_de_la_trajectoire(champ.u, champ.u, champ.v)
        - derivee_materielle(champ.u, champ.u, champ.v),
    )


def test_une_arete_materielle_s_allonge_au_taux_de_dilatation() -> None:
    """La vitesse relative des deux bouts vaut (du/dx) delta_x au premier ordre."""
    allongement = allongement_d_une_arete(U)
    assert _nul(allongement - ECART_MATERIEL * sp.diff(U, X))


def test_la_divergence_est_le_taux_de_dilatation_de_l_aire() -> None:
    """(1/A) DA/Dt = div V, obtenu en derivant le produit des deux aretes."""
    assert _nul(dilatation_relative(U, V) - divergence(U, V))


def test_un_champ_derivant_d_une_fonction_de_courant_est_a_divergence_nulle() -> None:
    """Theoreme de Schwarz : c'est pourquoi psi n'existe pas si div V != 0."""
    assert divergence_d_un_champ_derivant_de_psi() == 0


def test_un_champ_derivant_d_un_potentiel_est_irrotationnel() -> None:
    """Meme argument, cote potentiel : phi n'existe pas si omega_z != 0."""
    assert rotation_d_un_champ_derivant_de_phi() == 0


def test_les_equipotentielles_coupent_les_lignes_de_courant_a_angle_droit() -> None:
    """L'orthogonalite affirmee par le corrige doit etre calculee."""
    champ = enonce.CHAMP_C
    assert gradients_orthogonaux(champ.u, champ.v) == 0


def test_le_disque_materiel_devient_une_ellipse_d_aire_constante() -> None:
    """L'equation de l'image est obtenue en eliminant l'angle polaire."""
    equation = image_du_cercle(enonce.CHAMP_C.tenseur_taux_deformation())
    rayon = sp.Symbol("r_0", positive=True)
    assert equation.rhs == 1
    # Un point de l'axe x, a l'instant t, est bien sur l'image.
    sur_l_axe = equation.lhs.subs({X: rayon * sp.exp(2 * T), Y: 0})
    assert sp.simplify(sur_l_axe) == 1
