"""Les performances d'une boucle fermee, telles que le cours les mesure.

Ancrage direct sur `Exs_08_Performances_Reponses` : la feuille publie
`|M(0)| = -3 dB` et `y(infini) = 0,707`, puis `D = 87 %` et
`y_max = 1,32`. Les deux couples se recoupent exactement -- `10^(-3/20)`
vaut `0,7079`, et `0,707 x 1,87` vaut `1,322` -- et ce sont ces deux
relations que ce module porte.

Ce qu'il ne porte **pas**, et c'est delibere : le passage du pic de
resonance au depassement. Le cours le lit sur les abaques de
Black-Nichols, qui sont un abaque et non une formule ; le seul lien en
forme close est celui du second ordre, et il est expose comme tel. Deduire
`D` d'un pic par la formule du second ordre sur un systeme d'ordre trois
serait une approximation non declaree.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.signaux.domain.harmonique import depuis_les_decibels

CENT = 100


def gain_statique_depuis_decibels(decibels: sp.Expr) -> sp.Expr:
    """`|M(0)|` en dB donne la valeur finale de la reponse indicielle."""
    return depuis_les_decibels(decibels)


def maximum_de_la_reponse(valeur_finale: sp.Expr, depassement: sp.Expr) -> sp.Expr:
    """`y_max = y(infini) (1 + D)`, avec `D` en fraction et non en pourcent."""
    return sp.simplify(valeur_finale * (1 + depassement))


def depassement_du_second_ordre(amortissement: sp.Expr) -> sp.Expr:
    """`D = exp(-pi zeta / racine(1 - zeta^2))`, formule du second ordre."""
    zeta = sp.sympify(amortissement)
    if not sp.simplify(zeta > 0) or not sp.simplify(zeta < 1):
        msg = (
            f"amortissement hors de ]0, 1[ : {zeta} -- le depassement n'est "
            f"defini que pour un second ordre oscillant"
        )
        raise ValueError(msg)
    return sp.exp(-sp.pi * zeta / sp.sqrt(1 - zeta**2))


def amortissement_du_second_ordre(depassement: sp.Expr) -> sp.Expr:
    """L'inverse de la relation precedente, pour remonter de `D` a `zeta`."""
    logarithme = sp.log(sp.sympify(depassement))
    return sp.simplify(-logarithme / sp.sqrt(sp.pi**2 + logarithme**2))


def en_pourcent(fraction: sp.Expr) -> sp.Expr:
    """Le depassement tel que la feuille de reponses l'ecrit."""
    return sp.simplify(CENT * fraction)


AMORTISSEMENT_FAIBLE = 0.3
"""En dessous, la reponse oscille visiblement : « faiblement amorti »."""

AMORTISSEMENT_CRITIQUE = 1.0
"""Au-dela, plus aucune oscillation : « amorti »."""


def erreur_statique_de_consigne(gain_de_boucle: float) -> float:
    """`1 / (1 + K)` : ce qui reste de l'ecart en regime permanent.

    Un gain infini -- c'est-a-dire un integrateur dans la boucle --
    l'annule exactement. C'est le resultat que le fascicule met en avant,
    et c'est la raison d'etre du terme integral d'un correcteur.
    """
    if gain_de_boucle < 0:
        msg = f"gain de boucle negatif : {gain_de_boucle}"
        raise ValueError(msg)
    if math.isinf(gain_de_boucle):
        return 0.0
    return 1 / (1 + gain_de_boucle)


def erreur_de_perturbation(
    amplitude: float,
    gain_aval: float,
    gain_de_boucle: float,
) -> float:
    """L'effet d'une perturbation, attenue par la boucle mais pas annule.

    Le fascicule insiste : une perturbation qui entre **en aval** du
    correcteur n'est pas rejetee comme une erreur de consigne. Son effet
    vaut `A G_aval / (1 + K)`, et le gain aval ne se laisse pas oublier.
    """
    return amplitude * gain_aval * erreur_statique_de_consigne(gain_de_boucle)


def module_en_continu(gain_de_boucle: float) -> float:
    """`K / (1 + K)` : le gain de la boucle fermee en continu, note `|M(0)|`."""
    if gain_de_boucle < 0:
        msg = f"gain de boucle negatif : {gain_de_boucle}"
        raise ValueError(msg)
    return gain_de_boucle / (1 + gain_de_boucle)


def bande_passante_du_second_ordre(
    pulsation_propre: float,
    amortissement: float,
) -> float:
    """La pulsation a -3 dB d'un second ordre, en rad/s.

    `omega_0 racine(1 - 2 zeta^2 + racine(2 - 4 zeta^2 + 4 zeta^4))`. Elle
    est proportionnelle a `omega_0` a amortissement fixe -- c'est ce qui
    permet de regler la rapidite sans toucher a l'allure.
    """
    if amortissement <= 0:
        msg = f"amortissement nul ou negatif : {amortissement}"
        raise ValueError(msg)
    if pulsation_propre <= 0:
        msg = f"pulsation propre nulle ou negative : {pulsation_propre}"
        raise ValueError(msg)
    carre = amortissement**2
    sous_racine = 1 - 2 * carre + math.sqrt(2 - 4 * carre + 4 * carre**2)
    return pulsation_propre * math.sqrt(sous_racine)


def classe_d_amortissement(amortissement: float) -> str:
    """La classe que le fascicule illustre en trois planches.

    Les seuils sont dits : `0,3` et `1`. Le premier est conventionnel --
    au-dessous, le depassement passe les trente pour cent et s'impose a
    l'oeil ; le second est exact, c'est l'amortissement critique.
    """
    if amortissement <= 0:
        msg = f"amortissement nul ou negatif : {amortissement}"
        raise ValueError(msg)
    if amortissement < AMORTISSEMENT_FAIBLE:
        return "faiblement amorti"
    if amortissement < AMORTISSEMENT_CRITIQUE:
        return "moderement amorti"
    return "amorti"
