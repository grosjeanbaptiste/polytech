"""L'homogeneite est verifiee *avant* publication, non seulement dans les tests.

Le module `dimensions.py` annoncait dans sa docstring qu'il verifie « la
consequence sur chaque resultat publie ». C'etait faux au sens strict :
aucun chemin de production ne l'appelait, seuls deux fichiers de tests. Le
concept existait sans cas d'usage.

Les coefficients dimensionnels des champs de l'enonce vivaient eux aussi
dans un fichier de test -- du savoir de production loge chez son verificateur.
"""

import pytest
import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.application.generation import GenererMacros
from polytech.fluides.application.homogeneite import (
    COEFFICIENT_B_U,
    COEFFICIENT_B_V,
    COEFFICIENT_C,
    verifier_l_homogeneite,
    verifier_les_champs_publies,
)
from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.dimensions import (
    TEMPS,
    DimensionIndeterminee,
    ExpressionNonHomogene,
)
from polytech.fluides.domain.cinematique.symboles import T, X, Y


def test_les_champs_de_l_enonce_passent_la_verification() -> None:
    """Le cas reel : rien ne doit lever sur les deux champs publies."""
    verifier_les_champs_publies()


def test_les_coefficients_sont_ceux_que_le_corrige_enonce() -> None:
    """U = xt est une vitesse si son coefficient porte des s^-2."""
    assert COEFFICIENT_B_U == TEMPS**-2
    assert COEFFICIENT_B_V == TEMPS**-1
    assert COEFFICIENT_C == TEMPS**-1


def test_un_champ_dont_une_composante_melange_les_dimensions_est_refuse() -> None:
    """`xt + x` : un produit et une longueur nue ne s'additionnent pas."""
    boiteux = ChampDeVitesse(u=X * T + X, v=X - Y)
    with pytest.raises(ExpressionNonHomogene, match="non homogene"):
        verifier_l_homogeneite(boiteux, COEFFICIENT_B_U, COEFFICIENT_B_V)


def test_un_champ_homogene_mais_qui_n_est_pas_une_vitesse_est_refuse() -> None:
    """Homogene ne suffit pas : encore faut-il que ce soit une vitesse.

    `u = x` avec un coefficient en s^-2 donne une acceleration. Aucune
    contradiction interne, donc `dimension_de` ne leve pas ; c'est la
    comparaison a la dimension attendue qui l'attrape.
    """
    accelere = ChampDeVitesse(u=X, v=X - Y)
    with pytest.raises(ExpressionNonHomogene, match="n'est pas une vitesse"):
        verifier_l_homogeneite(accelere, COEFFICIENT_B_U, COEFFICIENT_B_V)


def test_la_publication_des_macros_echoue_si_un_champ_n_est_pas_homogene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La verification est branchee sur la production, non seulement testee.

    Sans ce branchement, une formule inhomogene serait publiee dans le
    corrige sans que rien ne bronche -- l'echec silencieux exact que la
    doctrine du depot interdit.
    """
    monkeypatch.setattr(enonce, "CHAMP_B", ChampDeVitesse(u=X * T + X, v=X - Y))
    with pytest.raises(ExpressionNonHomogene):
        GenererMacros.macros_publiees()


def test_la_verification_precede_tout_calcul_de_macro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elle doit lever *avant* qu'un resultat faux ne soit calcule.

    Un champ a composante nulle ferait echouer bien des calculs en aval,
    avec des erreurs obscures. C'est la verification qui parle la premiere,
    et elle distingue deux cas que le premier jet de ce test confondait :
    une expression nulle n'est pas *inhomogene*, sa dimension est
    *indeterminee*. C'est la distinction juste, et le message le dit.
    """
    monkeypatch.setattr(enonce, "CHAMP_C", ChampDeVitesse(u=sp.Integer(0), v=X))
    with pytest.raises(DimensionIndeterminee, match="expression nulle"):
        GenererMacros.macros_publiees()
