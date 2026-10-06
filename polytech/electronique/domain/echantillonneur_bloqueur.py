"""L'echantillonneur-bloqueur : charger vite, garder longtemps.

Syllabus, section 5.11.3. Un convertisseur regarde le signal pendant un
temps d'ouverture ; pour qu'il y voie une tension constante, un
condensateur la memorise. Deux suiveurs l'encadrent :

* le premier le **charge** a travers sa faible impedance de sortie, avec
  la constante de temps `C R_o` -- `R_o` peut descendre sous l'ohm ;
* le second le laisse se **decharger** dans sa grande impedance d'entree,
  avec la constante `C R_in` -- `R_in` depasse volontiers le gigaohm.

Le rapport des deux constantes, `R_in / R_o`, dit combien de temps la
tension se garde pour le temps qu'elle a mis a s'etablir.
"""

from __future__ import annotations

import math


def constante_de_temps(capacite: float, resistance: float) -> float:
    """`tau = R C`, en secondes."""
    if capacite <= 0 or resistance <= 0:
        msg = f"capacite ou resistance nulle ou negative : {capacite}, {resistance}"
        raise ValueError(msg)
    return capacite * resistance


def fraction_perdue(duree: float, capacite: float, resistance: float) -> float:
    """`1 - e^(-t / R C)` : la part de l'ecart qui s'est resorbee en `t`.

    Au maintien, c'est la tension perdue ; a la charge, la tension gagnee.
    """
    if duree < 0:
        msg = f"duree negative : {duree} s"
        raise ValueError(msg)
    return -math.expm1(-duree / constante_de_temps(capacite, resistance))
