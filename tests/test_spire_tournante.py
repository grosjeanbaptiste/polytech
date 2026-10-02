"""La seance 2.1, exercice 5 : une spire qui tourne dans un champ uniforme.

Une spire rectangulaire de cotes `l` et `L` tourne a la vitesse angulaire
`omega` dans un champ `B` uniforme. A `t = 0` elle est perpendiculaire au
champ. Le manuscrit derive le flux et encadre `e = omega B l L sin(omega t)`.

Ce fichier ajoute une seconde route, que le manuscrit ne suit pas : le
champ electromoteur `v ^ B` sur les deux cotes qui coupent les lignes de
champ. Elle n'emploie aucun flux, et son accord verifie a la fois
l'amplitude et la phase.

C'est aussi le **contrepoint de l'exercice 12** : pour le disque, le flux
enlace ne variait pas et la regle du flux donnait zero. Ici le circuit
est un fil bien defini, et la regle du flux est juste.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.spire_tournante import (
    fem_par_le_flux,
    fem_par_les_cotes_actifs,
    flux_a_travers_la_spire,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS

INDUCTION = sp.Symbol("B", positive=True)
PETIT_COTE = sp.Symbol("ell", positive=True)
GRAND_COTE = sp.Symbol("L", positive=True)
AMPLITUDE = PULSATION * INDUCTION * PETIT_COTE * GRAND_COTE


def test_le_flux_est_maximal_a_l_instant_initial() -> None:
    """A `t = 0` la spire est perpendiculaire au champ : `phi = B l L`."""
    flux = flux_a_travers_la_spire(INDUCTION, PETIT_COTE, GRAND_COTE)
    assert sp.simplify(flux.subs(TEMPS, 0) - INDUCTION * PETIT_COTE * GRAND_COTE) == 0


def test_la_fem_est_celle_qu_encadre_le_manuscrit() -> None:
    """`e = omega B l L sin(omega t)`."""
    fem = fem_par_le_flux(INDUCTION, PETIT_COTE, GRAND_COTE)
    assert sp.simplify(fem - AMPLITUDE * sp.sin(PULSATION * TEMPS)) == 0


def test_les_cotes_actifs_donnent_la_meme_fem() -> None:
    """Route independante : `v ^ B` sur les deux cotes paralleles a l'axe."""
    par_le_flux = fem_par_le_flux(INDUCTION, PETIT_COTE, GRAND_COTE)
    par_les_cotes = fem_par_les_cotes_actifs(INDUCTION, PETIT_COTE, GRAND_COTE)
    assert sp.simplify(par_le_flux - par_les_cotes) == 0


def test_la_fem_est_nulle_quand_le_flux_est_maximal() -> None:
    """Et maximale quand le flux est nul : c'est la **variation** qui compte."""
    fem = fem_par_le_flux(INDUCTION, PETIT_COTE, GRAND_COTE)
    flux = flux_a_travers_la_spire(INDUCTION, PETIT_COTE, GRAND_COTE)
    quart_de_tour = sp.pi / (2 * PULSATION)
    assert fem.subs(TEMPS, 0) == 0
    assert sp.simplify(flux.subs(TEMPS, quart_de_tour)) == 0
    assert sp.simplify(fem.subs(TEMPS, quart_de_tour) - AMPLITUDE) == 0


def test_la_fem_moyenne_sur_un_tour_est_nulle() -> None:
    """Une spire tournante est un alternateur : aucune composante continue."""
    fem = fem_par_le_flux(INDUCTION, PETIT_COTE, GRAND_COTE)
    periode = 2 * sp.pi / PULSATION
    assert sp.simplify(sp.integrate(fem, (TEMPS, 0, periode))) == 0
