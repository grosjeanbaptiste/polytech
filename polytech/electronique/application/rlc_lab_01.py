"""Laboratoire 1 d'electronique, section 2.4 : le circuit RLC passe-bas.

`R` et `L` en serie, la sortie sur `C` : un diviseur d'impedances dont le
bras s'annule presque a la resonance. `L = 8,2 mH`, `C = 1 nF` ; `R` vaut
0, 1 kohm ou 10 kohm, et le generateur ajoute ses 600 ohm quand le
transfert est rapporte a sa force electromotrice.

Le transfert est le biquad passe-bas de `reponse_du_biquad`, avec
`Q = racine(L / C) / R` : tout ce que le TP 7 a etabli sur lui vaut ici.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.domain.abcd import gain_en_tension, serie
from polytech.electronique.domain.reponse_du_biquad import module_carre

INDUCTANCE: Final = 8.2e-3
CAPACITE: Final = 1e-9
GENERATEUR: Final = 600.0


def passe_bas(
    resistance: sp.Expr, inductance: sp.Expr, capacite: sp.Expr, p: sp.Expr
) -> sp.Expr:
    """`Z_C / (Z_R + Z_L + Z_C)`."""
    return gain_en_tension(serie(resistance + p * inductance), 1 / (p * capacite))


def asymptotes(p: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Le transfert quand `Z_C` domine la somme, puis quand `Z_L` la domine."""
    produit = sp.nsimplify(INDUCTANCE * CAPACITE)
    return sp.Integer(1), 1 / (p**2 * produit)


def pulsation_de_resonance() -> float:
    """`w_R = 1 / racine(L C)` : `Z_L + Z_C` s'y annule."""
    return 1 / math.sqrt(INDUCTANCE * CAPACITE)


def frequence_de_resonance() -> float:
    """La meme en hertz."""
    return pulsation_de_resonance() / (2 * math.pi)


def impedance_caracteristique() -> float:
    """`racine(L / C)` : le module commun de `Z_L` et de `Z_C` a la resonance."""
    return math.sqrt(INDUCTANCE / CAPACITE)


def reponse_a_la_resonance(resistance: float) -> complex:
    """`H(j w_R) = -j racine(L / C) / R`, evalue sur le diviseur lui-meme."""
    p = sp.Symbol("p")
    r, ind, c = (sp.nsimplify(v) for v in (resistance, INDUCTANCE, CAPACITE))
    transfert = passe_bas(r, ind, c, p)
    return complex(
        transfert.subs(p, sp.I / sp.sqrt(sp.nsimplify(INDUCTANCE * CAPACITE)))
    )


def module_a_la_resonance(resistance: float, *, generateur: bool) -> float:
    """`racine(L / C) / R_tot` : le facteur de qualite du circuit."""
    totale = resistance + (GENERATEUR if generateur else 0.0)
    return math.sqrt(module_carre(1.0, impedance_caracteristique() / totale))


def resistance_critique() -> float:
    """`2 racine(L / C)` : au-dela, les deux poles sont reels (`Q < 1/2`)."""
    return 2 * impedance_caracteristique()


def poles_reels(resistance: float) -> bool:
    """Vrai si `R^2 C^2 - 4 L C >= 0` : pas de resonance possible."""
    return resistance >= resistance_critique()


def poles(resistance: float) -> tuple[float, float]:
    """Les deux poles reels, le plus lent d'abord."""
    if not poles_reels(resistance):
        msg = f"poles complexes pour {resistance} ohm : pas de valeurs reelles"
        raise ValueError(msg)
    demi = resistance / (2 * INDUCTANCE)
    ecart = math.sqrt(demi**2 - pulsation_de_resonance() ** 2)
    return -demi + ecart, -demi - ecart


def tensions_a_la_resonance(resistance: float) -> tuple[complex, complex, complex]:
    """`V_C`, `V_L` et `V_L + V_C` rapportees a la tension d'entree du circuit."""
    sur_c = reponse_a_la_resonance(resistance)
    sur_l = -sur_c
    return sur_c, sur_l, sur_c + sur_l


def resistance_parasite(gain_releve: float, resistance: float) -> float:
    """La resistance serie qu'il faut ajouter pour expliquer un gain releve.

    A la resonance le module vaut `racine(L / C) / R_serie` : un gain
    releve, en decibels, donne donc la resistance serie reelle.
    """
    if resistance < 0:
        msg = f"resistance negative : {resistance} ohm"
        raise ValueError(msg)
    return impedance_caracteristique() / 10 ** (gain_releve / 20) - resistance


def tension_mesuree_a_la_resonance(
    resistance: float, *, parasite: float = 0.0
) -> float:
    """Question 6 : `V_1 / V_in = R / (R + R_G)`, la branche LC etant un fil."""
    serie_reelle = resistance + parasite
    return serie_reelle / (serie_reelle + GENERATEUR)


def tension_mesuree(resistance: float, frequence: float) -> float:
    """`|V_1 / V_in|` a cette frequence : le pont entre `R_G` et le circuit."""
    w = 2 * math.pi * frequence
    circuit = complex(resistance, w * INDUCTANCE - 1 / (w * CAPACITE))
    return abs(circuit / (circuit + GENERATEUR))
