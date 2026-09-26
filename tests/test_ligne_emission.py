"""Le corrige affirme que les trois familles coincident : il faut la troisieme.

Lignes de courant et trajectoires etaient calculees ; la ligne d'emission
etait seulement annoncee comme confondue avec elles.
"""

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.emission import (
    INSTANT_EMISSION,
    invariant_de_la_ligne_emission,
    ligne_emission,
)

DEPART = (sp.Symbol("x_0", positive=True), sp.Symbol("y_0", positive=True))


def test_la_ligne_d_emission_depend_de_l_instant_d_emission() -> None:
    """Chaque particule du filet a ete lachee a un instant tau different."""
    abscisse, _ = ligne_emission(enonce.CHAMP_C.u, enonce.CHAMP_C.v, *DEPART)
    assert abscisse.has(INSTANT_EMISSION)


def test_son_invariant_ne_depend_plus_de_l_instant_d_emission() -> None:
    """C'est precisement ce qui la fait coincider avec la ligne de courant."""
    invariant = invariant_de_la_ligne_emission(
        enonce.CHAMP_C.u,
        enonce.CHAMP_C.v,
        *DEPART,
    )
    assert not invariant.has(INSTANT_EMISSION)
    assert sp.simplify(invariant - DEPART[0] * DEPART[1]) == 0
