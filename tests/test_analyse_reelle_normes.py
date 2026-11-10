"""Les normes du syllabus de mathematiques appliquees.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
theoreme 6.1 (norme d'un vecteur, d'un vecteur de fonctions : le supremum des
valeurs absolues des coordonnees) et definition 10.1 (`L_p` : `int |f|^p`
fini). Les oracles sont des cas classiques : le syllabus ecrit ses formules
en images.
"""

import pytest
import sympy as sp

from polytech.maths.domain.analyse_reelle.normes import (
    est_dans_lp,
    norme_lp,
    norme_sup_de_fonctions,
    norme_sup_du_vecteur,
)

T = sp.Symbol("t", real=True)


def test_la_norme_sup_d_un_vecteur_est_la_plus_grande_valeur_absolue() -> None:
    assert norme_sup_du_vecteur((1, -3, 2)) == 3


def test_un_vecteur_vide_est_refuse() -> None:
    with pytest.raises(ValueError, match="vecteur vide"):
        norme_sup_du_vecteur(())


def test_la_norme_sup_d_un_vecteur_de_fonctions_prend_le_sup_de_chaque_coordonnee() -> (
    None
):
    fonctions = (sp.cos(T), 2 * sp.sin(T))
    assert norme_sup_de_fonctions(fonctions, T, sp.Interval(0, 2 * sp.pi)) == 2


def test_la_norme_l1_d_une_exponentielle_symetrique() -> None:
    assert norme_lp(sp.exp(-sp.Abs(T)), 1, T, sp.S.Reals) == 2


def test_la_norme_l2_d_une_exponentielle_symetrique() -> None:
    assert norme_lp(sp.exp(-sp.Abs(T)), 2, T, sp.S.Reals) == 1


def test_un_sur_t_est_de_carre_sommable_mais_pas_sommable_sur_un_infini() -> None:
    domaine = sp.Interval(1, sp.oo)
    assert est_dans_lp(1 / T, 2, T, domaine)
    assert not est_dans_lp(1 / T, 1, T, domaine)


def test_une_fonction_hors_de_lp_n_a_pas_de_norme() -> None:
    with pytest.raises(ValueError, match="n'est pas de puissance 1-sommable"):
        norme_lp(1 / T, 1, T, sp.Interval(1, sp.oo))


def test_un_exposant_inferieur_a_un_est_refuse() -> None:
    with pytest.raises(ValueError, match="exposant p >= 1"):
        norme_lp(T, sp.Rational(1, 2), T, sp.Interval(0, 1))


def test_la_norme_l2_prend_la_racine_de_l_integrale() -> None:
    """`int e^(-4|t|) dt = 1/2`, donc `||e^(-2|t|)||_2 = racine(2) / 2`."""
    norme = norme_lp(sp.exp(-2 * sp.Abs(T)), 2, T, sp.S.Reals)
    assert sp.simplify(norme - sp.sqrt(2) / 2) == 0
