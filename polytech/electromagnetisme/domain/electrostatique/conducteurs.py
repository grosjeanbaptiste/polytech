"""Le champ dans, le long de et sur un conducteur homogene en equilibre.

Le syllabus d'electricite generale (3.1 a 3.6) :

* a l'interieur, le champ est nul et le potentiel constant (3.2.1) ; le
  conducteur ne se charge qu'en surface (3.2.5) ;
* le long de la surface, a l'exterieur, le champ est normal et vaut
  `E_ext = sigma / eps_0` (3.3.3) ;
* **sur** la surface -- le champ qui agit sur la charge `sigma dS`, celle-ci
  retiree -- il n'en vaut que la moitie, `E_sur = sigma / (2 eps_0)`
  (3.4.2), d'ou une force `sigma^2 / (2 eps_0)` par unite de surface ;
* une enveloppe creuse porte `Q_0i = -somme(Q_k)` sur sa face interieure
  et `Q_0e = Q_0 + somme(Q_k)` sur sa face exterieure (3.6.1, 3.6.2) :
  c'est l'influence totale.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE

if TYPE_CHECKING:
    from collections.abc import Sequence

Reel = Fraction | float
"""Une charge : un reel, ou une fraction quand le compte est exact."""


def champ_le_long_de_la_surface(surfacique: object) -> sp.Expr:
    """`E_ext = sigma / eps_0`, normal a la surface, cote exterieur."""
    return sp.sympify(surfacique) / PERMITTIVITE


def champ_sur_la_surface(surfacique: object) -> sp.Expr:
    """`E_sur = sigma / (2 eps_0)` : le champ qui agit sur la charge de surface."""
    return champ_le_long_de_la_surface(surfacique) / 2


def force_surfacique(surfacique: object) -> sp.Expr:
    """`sigma E_sur = sigma^2 / (2 eps_0)` : toujours dirigee vers l'exterieur."""
    return sp.sympify(surfacique) * champ_sur_la_surface(surfacique)


def charge_surfacique_depuis_le_champ(champ: object) -> sp.Expr:
    """`sigma = eps_0 E` : la charge que porte un conducteur ou regne ce champ."""
    return PERMITTIVITE * sp.sympify(champ)


def charge_surfacique_maximale(champ_disruptif: object) -> sp.Expr:
    """`sigma = eps_0 E` : la charge au-dela de laquelle le milieu s'ionise."""
    return charge_surfacique_depuis_le_champ(champ_disruptif)


def charge_lineique_pour_un_potentiel(
    potentiel: object, rayon: object, reference: object
) -> sp.Expr:
    """`lambda = 2 pi eps_0 V(R) / ln(rho_0 / R)`, pour un cylindre seul (3.8.2)."""
    rho_0, r = sp.sympify(reference), sp.sympify(rayon)
    if rho_0 == r:
        msg = "rayon de reference egal au rayon du conducteur : potentiel nul impose"
        raise ValueError(msg)
    return 2 * sp.pi * PERMITTIVITE * sp.sympify(potentiel) / sp.log(rho_0 / r)


def charges_des_faces_de_l_enveloppe(
    charge: Reel, interieures: Sequence[Reel]
) -> tuple[Reel, Reel]:
    """`(Q_0i, Q_0e)` : les charges des faces interieure et exterieure."""
    entourees = sum(interieures)
    return (-entourees, charge + entourees)
