"""Les composants ideaux, et ce qu'un composant reel leur ajoute.

Planches du chapitre 1, partie II. Trois lois definissent les composants
ideaux : `V = R I`, `I = C dV/dt`, `V = L dI/dt`. Un composant reel n'en
est jamais un seul.

* Une **ligne** ne transmet pas instantanement : `V_out(t) = V_in(t - l /
  v)`, avec une vitesse de l'ordre du tiers de celle de la lumiere.
* Une **resistance** est aussi un condensateur. Ses deux armatures
  metalliques se chargent, et la continuite appliquee a l'une d'elles
  donne `I = V / R + C dV/dt`, avec `C = eps A / l` : un condensateur
  ideal en parallele, qu'aucune sonde ne peut isoler.
* Un **fil** porte une inductance interne, due au champ magnetique dans
  le metal lui-meme : 50 nH/m pour une section ronde, a basse frequence.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.symboles import TEMPS

INDUCTANCE_INTERNE_LINEIQUE: Final = PERMEABILITE_DU_VIDE / (8 * sp.pi.evalf())
"""`mu_0 / 8 pi`, en H/m : les « 50 nH/m » de la planche 24 (Rosa, 1908)."""


def tension_de_la_resistance(resistance: object, courant: object) -> sp.Expr:
    """`V = R I`."""
    return sp.sympify(resistance) * sp.sympify(courant)


def courant_du_condensateur(capacite: object, tension: object) -> sp.Expr:
    """`I = C dV/dt`."""
    return sp.sympify(capacite) * sp.diff(sp.sympify(tension), TEMPS)


def tension_de_l_inductance(inductance: object, courant: object) -> sp.Expr:
    """`V = L dI/dt`."""
    return sp.sympify(inductance) * sp.diff(sp.sympify(courant), TEMPS)


def sortie_de_la_ligne(entree: sp.Expr, longueur: float, vitesse: float) -> sp.Expr:
    """`V_out(t) = V_in(t - l / v)` : la ligne retarde, sans deformer."""
    if vitesse <= 0:
        msg = f"vitesse de propagation nulle ou negative : {vitesse} m/s"
        raise ValueError(msg)
    return entree.subs(TEMPS, TEMPS - longueur / vitesse)


def courant_de_la_resistance_reelle(
    resistance: object, capacite: object, tension: object
) -> sp.Expr:
    """`I = V / R + C dV/dt` : la resistance ideale et sa capacite cachee."""
    conduit = sp.sympify(tension) / sp.sympify(resistance)
    return conduit + courant_du_condensateur(capacite, tension)
