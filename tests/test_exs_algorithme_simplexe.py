"""Optimisation : les exercices I et II sur l'algorithme simplexe.

Source : `Optimisation/EXS/Exs_Algorithme_simplexe.pdf`, planches 3 a 19.
Les planches portent les tableaux et les resultats ; ils sont recopies
comme oracles et recalcules par la bibliotheque -- les six bases de
l'exercice I une a une, puis tous les sommets optimaux de l'exercice II.
Toutes les valeurs des planches sont retrouvees.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.application import simplexe_bornes as bo
from polytech.optimisation.application import simplexe_decompte as dc
from polytech.optimisation.application import simplexe_optimums as op
from polytech.optimisation.application import simplexe_unicite as un
from polytech.optimisation.domain.contraintes import Contrainte, Sens

M = bo.M
INF, SUP, EGAL = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
DEMI = sp.Rational(1, 2)

# --- Exercice I : une variable bornee des deux cotes ------------------------


def test_i_l_optimum_des_planches() -> None:
    """Planche 11 : `x = -1`, `y = 5`, `z = -19`."""
    assert bo.optimum() == (-19, (-1, 5))


def test_i_les_six_tableaux() -> None:
    """Planches 6 a 11 : la solution et la valeur de chaque tableau."""
    attendus = [
        ((0, 0), 6 * M),
        ((0, DEMI), sp.Rational(7, 2) * M - sp.Rational(3, 2)),
        ((2, sp.Rational(3, 2)), M / 2 + sp.Rational(7, 2)),
        ((2, 2), 2),
        ((0, 4), -12),
        ((-1, 5), -19),
    ]
    obtenus = [(etape.point, sp.expand(etape.valeur)) for etape in bo.parcours()]
    assert obtenus == [(p, sp.expand(v)) for p, v in attendus]


def test_i_trois_tableaux_non_admissibles_puis_trois_admissibles() -> None:
    """Planches 6 a 11 : « non admissible » tant qu'une artificielle est en base."""
    assert [etape.admissible for etape in bo.parcours()] == [False] * 3 + [True] * 3


def test_i_la_valeur_decroit_a_chaque_tableau() -> None:
    """Second chemin : `M` etant grand, chaque pivot ameliore l'objectif."""
    valeurs = [float(etape.valeur.subs(M, 1000)) for etape in bo.parcours()]
    assert valeurs == sorted(valeurs, reverse=True)


def test_i_le_point_zero_quatre_n_est_pas_un_sommet_du_domaine() -> None:
    """Le domaine est le segment de (-1, 5) a (2, 2) : (0, 4) est en son milieu."""
    assert bo.sommets_du_domaine() == ((-1, 5), (2, 2))
    assert bo.parcours()[4].point == (0, 4)


def test_i_non_unique_dans_les_variables_dedoublees_seulement() -> None:
    """Planche 11 : « non unique et non degeneree ». Le cout reduit de `x+` est nul."""
    assert un.cout_reduit_de_x_plus() == 0
    assert un.x_plus_ouvre_un_rayon()
    assert un.deplacement_de_x_le_long_du_rayon() == 0
    assert not un.est_degeneree()


def test_i_la_variante_translatee() -> None:
    """Planches 13 et 14 : `x = x' - 3` ; trois tableaux, `x' = 2`, `y = 5`."""
    attendus = [((0, 0), 7 * M - 12), ((0, 3), 4 * M - 21), ((2, 5), -19)]
    obtenus = [(e.point, sp.expand(e.valeur)) for e in bo.parcours_translate()]
    assert obtenus == [(p, sp.expand(v)) for p, v in attendus]
    assert bo.optimum_translate() == (-19, (2, 5))


def test_i_le_decompte_de_la_planche_16() -> None:
    """Planche 16 : 3, 5, 0, 5 pour `x+ - x-` ; 2, 4, 2, 4 pour `x' - 3`."""
    assert dc.decompte(translate=False) == (3, 5, 0, 5)
    assert dc.decompte(translate=True) == (2, 4, 2, 4)


def test_i_quand_une_contrainte_demande_une_artificielle() -> None:
    """Une egalite toujours ; une inegalite si son ecart ne peut pas ouvrir la base."""
    cas = {
        (EGAL, 1): True,
        (INF, 1): False,
        (INF, -1): True,
        (SUP, 1): True,
        (SUP, -1): False,
    }
    for (sens, second_membre), attendu in cas.items():
        assert (
            dc.demande_une_artificielle(Contrainte((1,), sens, second_membre))
            is attendu
        )


def test_i_les_deux_formulations_de_la_planche_16_ont_le_meme_optimum() -> None:
    """Second chemin : changer de variable ne change pas le probleme."""
    assert dc.optimum(translate=False) == dc.optimum(translate=True)
    assert dc.optimum(translate=False) == (-25, (sp.Rational(-5, 2), 0))


# --- Exercice II : toutes les solutions optimales ---------------------------


def test_ii_les_quatre_sommets_optimaux() -> None:
    """Planches 18 et 19 : (1,0,1), (0,1,1), (2,0,0) et (0,2,0), de valeur 2."""
    attendus = {(1, 0, 1), (0, 1, 1), (2, 0, 0), (0, 2, 0)}
    assert set(op.sommets_optimaux()) == attendus
    assert op.valeur_optimale() == 2


@pytest.mark.parametrize(
    "poids", [(1, 1, 0, 0), (2, -1, 0, 0), (sp.Rational(1, 2), sp.Rational(1, 2))]
)
def test_ii_des_poids_qui_ne_sont_pas_convexes_sont_refuses(
    poids: tuple[sp.Expr, ...],
) -> None:
    """Leur somme vaut 1, aucun n'est negatif, et il y en a un par sommet."""
    with pytest.raises(ValueError, match="poids non convexes pour "):
        op.combinaison(poids)


def test_ii_toute_combinaison_convexe_est_optimale() -> None:
    """Planche 19 : l'ensemble convexe des quatre sommets."""
    poids = (
        sp.Rational(1, 10),
        sp.Rational(2, 10),
        sp.Rational(3, 10),
        sp.Rational(4, 10),
    )
    point = op.combinaison(poids)
    assert op.est_admissible(point)
    assert op.valeur(point) == 2


def test_ii_aucun_autre_sommet_n_est_optimal() -> None:
    """Second chemin : les sommets du domaine, enumeres par leurs bases."""
    optimaux = {s for s in op.sommets_du_domaine() if op.valeur(s) == 2}
    assert optimaux == set(op.sommets_optimaux())
