"""Le spectre de raies d'un signal periodique.

Syllabus de signaux et systemes, section 2.1.4. Rendre un signal
periodique, c'est le convoluer par un peigne de Dirac (2.17) ; sa
transformee est alors multipliee par un peigne (2.18), et il n'en reste
que des raies, aux multiples de `f_0 = 1 / T_0` :

`F_k = F(k f_0) / T_0`,

`F` etant la transformee d'une periode. Les exponentielles imaginaires
etant orthogonales et de puissance unite, la valeur efficace du signal
est la racine de la somme des carres des raies (2.14), et le taux de
distorsion harmonique rapporte celle des harmoniques d'ordre superieur a
un a celle du signal entier.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.frequence import FREQUENCE
from polytech.signaux.domain.symboles import IMAGE

if TYPE_CHECKING:
    from collections.abc import Sequence


def raie(image_d_une_periode: sp.Expr, periode: sp.Expr, rang: int) -> sp.Expr:
    """`F_k = F(k / T_0) / T_0`, prolongee par continuite ou `F` s'ecrit 0/0."""
    if not sp.sympify(periode).is_positive:
        msg = f"periode nulle ou negative : {periode}"
        raise ValueError(msg)
    image, frequence = sp.sympify(image_d_une_periode), rang / sp.sympify(periode)
    valeur = image.subs(FREQUENCE, frequence)
    if valeur.has(sp.nan, sp.zoo):
        valeur = sp.limit(image, FREQUENCE, frequence)
    return sp.simplify(valeur / periode)


def valeur_efficace(raies: Sequence[sp.Expr]) -> sp.Expr:
    """(2.14) : la racine de la somme des carres de toutes les raies."""
    return sp.sqrt(sum((sp.Abs(r) ** 2 for r in raies), sp.S.Zero))


def valeur_efficace_d_un_signal_reel(positives: Sequence[sp.Expr]) -> sp.Expr:
    """(2.15) : moyenne nulle, raies conjuguees deux a deux."""
    return sp.sqrt(2 * sum((sp.Abs(r) ** 2 for r in positives), sp.S.Zero))


def taux_de_distorsion_harmonique(amplitudes: Sequence[float]) -> float:
    """En pour cent, les amplitudes etant celles des rangs 1, 2, 3...

    Le rapport de la valeur efficace des harmoniques de rang superieur a
    un a celle du signal entier, composante continue exclue.
    """
    if not amplitudes or amplitudes[0] == 0:
        msg = "fondamental nul ou absent : le taux de distorsion n'est pas defini"
        raise ValueError(msg)
    total = sum(a**2 for a in amplitudes)
    return 100 * math.sqrt(sum(a**2 for a in amplitudes[1:]) / total)


def frequence_fondamentale(periode_en_echantillons: int, cadence: float) -> float:
    """`F_e / N_0`, en hertz, pour une periode lue en nombre d'echantillons."""
    if periode_en_echantillons <= 0:
        msg = f"periode nulle ou negative : {periode_en_echantillons} echantillon(s)"
        raise ValueError(msg)
    return cadence / periode_en_echantillons


def raie_de_sortie(
    raie_d_entree: sp.Expr, transmittance: sp.Expr, rang: int, fondamentale: sp.Expr
) -> sp.Expr:
    """`Y_k = X_k H(j k omega_0)` : en regime, chaque raie garde sa frequence."""
    reponse = sp.sympify(transmittance).subs(IMAGE, sp.I * rang * fondamentale)
    if reponse.has(sp.zoo, sp.nan):
        msg = f"pole sur la raie de rang {rang} : le regime n'existe pas"
        raise ValueError(msg)
    return sp.simplify(raie_d_entree * reponse)
