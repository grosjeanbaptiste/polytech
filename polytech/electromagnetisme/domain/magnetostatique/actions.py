"""Les actions du champ sur une charge et sur un circuit.

Le syllabus d'electricite generale (8.3, 8.5 et 8.7) :

* une charge en mouvement dans un champ electrique et un champ magnetique
  subit `F = q (E + v ^ B)` (8.3.4), dite force de Lorentz, de
  Laplace-Lorentz ou de Coulomb-Lorentz ;
* quand un conducteur filiforme se deplace ou se deforme, le travail des
  forces du champ vaut le courant par le flux coupe, `dT = I d(phi)`
  (8.5.1) ; pour un circuit entier, a courant maintenu constant, c'est le
  courant par l'accroissement du flux qui le traverse (8.7).
"""

from __future__ import annotations

from polytech.electromagnetisme.domain.magnetostatique.lois import force_de_lorentz

Triplet = tuple[float, float, float]


def force_electromagnetique(
    charge: float, champ: Triplet, vitesse: Triplet, induction: Triplet
) -> Triplet:
    """`F = q (E + v ^ B)`, en newtons."""
    m_x, m_y, m_z = force_de_lorentz(charge, vitesse, induction)
    e_x, e_y, e_z = champ
    return (charge * e_x + m_x, charge * e_y + m_y, charge * e_z + m_z)


def travail_a_courant_constant(
    courant: float, flux_initial: float, flux_final: float
) -> float:
    """`T = I (phi_2 - phi_1)` : positif si le flux a travers le circuit augmente."""
    return courant * (flux_final - flux_initial)
