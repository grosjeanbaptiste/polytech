"""Le diamagnetisme d'un electron lie, et le couple qui oriente un moment.

Les planches du chapitre III.5 (4 a 6) demontrent ce que le syllabus
affirme a son paragraphe 10.1. Un electron tourne autour de son noyau a la
pulsation `omega`, retenu par une force centripete `m_e omega^2 r`. Une
induction `B` normale a l'orbite y ajoute la force magnetique `e v B` :
la pulsation monte a `omega_B`, telle que
`m_e omega^2 r + e omega_B r B = m_e omega_B^2 r`. Au premier ordre,
l'ecart vaut `e B / (2 m_e)` -- la pulsation de Larmor -- et le moment
magnetique de l'orbite varie de `-e^2 B r^2 / (4 m_e)` : il s'oppose a
l'induction, quel que soit le corps.

Un moment magnetique `m` plonge dans une induction subit le couple
`m ^ B`, qui tend a l'aligner : c'est l'origine du paramagnetisme.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def ecart_de_larmor(charge: Grandeur, induction: Grandeur, masse: Grandeur) -> Grandeur:
    """`Delta omega = e B / (2 m_e)`, en radians par seconde."""
    if (sp.sympify(masse) <= 0) == sp.true:
        msg = f"masse nulle ou negative : {masse} kg"
        raise ValueError(msg)
    return charge * induction / (2 * masse)


def pulsation_sous_induction(
    pulsation: Grandeur, charge: Grandeur, induction: Grandeur, masse: Grandeur
) -> sp.Expr:
    """`omega_B`, racine positive de `omega_B^2 - (e B / m_e) omega_B - omega^2`."""
    larmor = ecart_de_larmor(charge, induction, masse)
    return larmor + sp.sqrt(pulsation**2 + larmor**2)


def moment_induit(
    charge: Grandeur, induction: Grandeur, rayon: Grandeur, masse: Grandeur
) -> Grandeur:
    """`Delta m = -e^2 B r^2 / (4 m_e)`, selon la normale : oppose a l'induction."""
    return -charge * ecart_de_larmor(charge, induction, masse) * rayon**2 / 2


def couple_sur_un_moment(
    moment: tuple[Any, Any, Any], induction: tuple[Any, Any, Any]
) -> tuple[Any, Any, Any]:
    """`C = m ^ B`, en newtons-metres."""
    x, y, z = sp.Matrix(moment).cross(sp.Matrix(induction))
    return (x, y, z)
