"""Domaine de validite des formules de la partie II.

Le corrige resout `dy/dx = v/u` en divisant par u, prend `ln x` sans valeur
absolue et ecrit `exp(ln x / t)` sans dire que t doit etre non nul. Une seule
de ces conditions etait enoncee (pour l'ecoulement C). Ces tests exigent que
le domaine soit calcule, et que la solution close y soit verifiee.
"""

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.symboles import T, X, Y
from polytech.fluides.domain.cinematique.validite import (
    lieu_singulier,
    residu_des_lignes_de_courant,
)


def test_la_pente_de_c_n_est_pas_definie_sur_l_axe_des_ordonnees() -> None:
    """U = 2x s'annule en x = 0 : dy/dx = v/u y perd son sens."""
    assert lieu_singulier(enonce.CHAMP_C.u) == sp.FiniteSet(sp.Eq(X, 0))


def test_la_pente_de_b_degenere_aussi_a_l_instant_initial() -> None:
    """U = xt s'annule sur x = 0 *et* a t = 0 : deux causes, pas une."""
    assert lieu_singulier(enonce.CHAMP_B.u) == sp.FiniteSet(sp.Eq(X, 0), sp.Eq(T, 0))


def test_la_solution_close_de_c_verifie_son_equation() -> None:
    """Report dans dy/dx - v/u : verification symbolique, pas numerique."""
    assert residu_des_lignes_de_courant(enonce.CHAMP_C) == 0


def test_la_solution_close_de_b_verifie_son_equation() -> None:
    """Le calcul par facteur integrant, verifie en y reportant son resultat."""
    assert residu_des_lignes_de_courant(enonce.CHAMP_B) == 0


def test_un_champ_a_composante_u_constante_n_a_aucun_point_singulier() -> None:
    """Controle du module : un ecoulement uniforme ne degenere nulle part."""
    uniforme = ChampDeVitesse(u=sp.Integer(1), v=Y * 0)
    assert lieu_singulier(uniforme.u) == sp.EmptySet
