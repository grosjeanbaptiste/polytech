"""Optimisation : les exercices III et IV, le dual et la sensibilite.

Source : `Optimisation/EXS/Exs_Algorithme_simplexe.pdf`, planches 21 a 45.
Le probleme dual, ses variables et le theoreme des ecarts
complementaires ; puis les intervalles dans lesquels un cout ou un second
membre peut varier sans changer la base optimale. Chaque valeur des
planches est recopiee comme oracle et recalculee.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.application import simplexe_dual as du
from polytech.optimisation.application import simplexe_sensibilite as se
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

DEMI = sp.Rational(1, 2)

# --- Exercice III : le dual -------------------------------------------------


def test_iii_le_dual_de_la_planche_27() -> None:
    """Planche 27 : `max -6u1 + 4u2 + 2u3 - 2u4 - 3u5`, deux contraintes."""
    objectif, contraintes, libres = du.dual()
    assert objectif == (-6, 4, 2, -2, -3)
    assert contraintes == (((1, 1, -2, -1, 1), "=", 4), ((-1, 1, 4, 0, 0), "<=", -3))
    assert libres == {2}


def test_iii_les_variables_duales_des_planches() -> None:
    """Planches 28 et 29 : `u = (7/2, 1/2, 0, 0, 0)`, et `w = z = -19`."""
    attendues = (sp.Rational(7, 2), DEMI, 0, 0, 0)
    assert du.variables_duales() == attendues
    assert du.valeur_du_dual() == -19


def test_iii_les_ecarts_complementaires() -> None:
    """Planche 29 : les sept produits sont nuls."""
    assert du.produits_des_ecarts() == (0,) * 7


def test_iii_les_duales_par_les_seuls_ecarts() -> None:
    """Second chemin : les ecarts complementaires suffisent a trouver `u`."""
    assert du.duales_par_les_ecarts() == du.variables_duales()


def test_iii_les_duales_se_lisent_dans_le_tableau_optimal() -> None:
    """Planche 28 : `u1 = 7/2` et `u2 = 1/2`, les couts reduits de `x1e` et `x1a`."""
    assert du.couts_reduits_des_ecarts() == (sp.Rational(7, 2), DEMI)


# --- Exercice IV : la sensibilite -------------------------------------------


def test_iv_l_optimum_et_les_duales() -> None:
    """Planche 36 : `(2, 3, 0)`, `z = -4` ; `u = (2, 3, 0)`."""
    assert se.optimum() == (-4, (2, 3, 0))
    assert se.variables_duales() == (2, 3, 0)


def test_iv_les_intervalles_des_couts() -> None:
    """Planches 38 a 40 : `1 + dx >= -2`, `-2 + dy <= 0`, `6 + dz >= -4`."""
    assert se.intervalles_des_couts() == ((-2, sp.oo), (-sp.oo, 0), (-4, sp.oo))


def test_iv_les_intervalles_des_seconds_membres() -> None:
    """Planches 42 a 44 : `2 <= d1 <= 6`, `1 <= d2 <= 5`, `d3 >= 3`."""
    assert se.intervalles_des_seconds_membres() == ((2, 6), (1, 5), (3, sp.oo))


@pytest.mark.parametrize("rang", [0, 1, 2])
def test_iv_la_base_tient_dans_l_intervalle_et_cede_au_dela(rang: int) -> None:
    """Second chemin : le probleme resolu de nouveau, de part et d'autre des bornes."""
    assert se.la_base_tient(rang, dedans=True)
    assert not se.la_base_tient(rang, dedans=False)


def test_un_probleme_impossible_n_a_pas_d_optimum() -> None:
    """Deux contraintes contradictoires : la fonction le dit au lieu de rendre rien."""
    contradictoire = ProgrammeGeneral(
        Optimisation.MIN,
        (1,),
        (Contrainte((1,), Sens.INFERIEUR, 1), Contrainte((1,), Sens.SUPERIEUR, 2)),
    )
    with pytest.raises(ValueError, match="pas d'optimum fini"):
        optimum_fini(contradictoire)
