"""Les montages de base de l'AOP, tels que le fascicule les enumere.

Additionneur, soustracteur, convertisseur courant-tension, derivateur,
integrateur. Chacun se ramene a un transfert, et deux d'entre eux portent
un piege que le fascicule signale.

Le **derivateur** a un zero a l'origine : il amplifie les hautes
frequences, donc le bruit, et le derivateur pur est inutilisable -- le
fascicule ne le montre jamais sans une resistance en serie.

L'**integrateur** a un pole a l'origine : il diverge en continu, et le
moindre decalage de tension le sature. D'ou la resistance de fuite en
parallele sur la capacite, que tout montage reel porte.

Le **soustracteur** ne rejette le mode commun qu'a une condition :
`R_1/R_2 = R_3/R_4`. Un desequilibre de un pour cent le fait remonter, et
c'est ce qui limite le taux de rejection reel d'un amplificateur
differentiel a base d'AOP -- bien avant le composant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp


def somme_ponderee(
    contre_reaction: float,
    entrees: tuple[float, ...],
) -> tuple[float, ...]:
    """Les poids `-R_f / R_i` de l'additionneur, un par voie."""
    if not entrees:
        msg = "aucune entree : l'additionneur n'a rien a sommer"
        raise ValueError(msg)
    if any(resistance <= 0 for resistance in entrees):
        msg = f"resistance d'entree nulle ou negative : {entrees}"
        raise ValueError(msg)
    return tuple(-contre_reaction / resistance for resistance in entrees)


def soustracteur(
    entree_inverseuse: float,
    contre_reaction: float,
    entree_directe: float,
    vers_la_masse: float,
) -> tuple[float, float]:
    """Les deux gains du soustracteur : differentiel, puis mode commun.

    Le second doit etre nul, et il ne l'est qu'a l'equilibre des ponts.
    Le rendre explicitement, au lieu de le supposer nul, est ce qui permet
    de chiffrer l'effet d'un desequilibre.
    """
    for valeur in (entree_inverseuse, contre_reaction, entree_directe, vers_la_masse):
        if valeur <= 0:
            msg = f"resistance nulle ou negative dans le pont : {valeur} ohm"
            raise ValueError(msg)
    direct = vers_la_masse / (entree_directe + vers_la_masse)
    gain_direct = direct * (1 + contre_reaction / entree_inverseuse)
    gain_inverse = contre_reaction / entree_inverseuse
    differentiel = (gain_direct + gain_inverse) / 2
    return differentiel, gain_direct - gain_inverse


def convertisseur_courant_tension(transimpedance: float) -> float:
    """`-R` : le gain d'un convertisseur courant-tension, en ohms."""
    if transimpedance <= 0:
        msg = f"transimpedance nulle ou negative : {transimpedance} ohm"
        raise ValueError(msg)
    return -transimpedance


def derivateur(resistance: float, capacite: float, variable: sp.Expr) -> sp.Expr:
    """`-R C p` : un zero a l'origine, donc du bruit amplifie."""
    if resistance <= 0:
        msg = f"resistance nulle ou negative : {resistance} ohm"
        raise ValueError(msg)
    if capacite <= 0:
        msg = f"capacite nulle ou negative : {capacite} F"
        raise ValueError(msg)
    return -resistance * capacite * variable


def integrateur(resistance: float, capacite: float, variable: sp.Expr) -> sp.Expr:
    """`-1 / (R C p)` : un pole a l'origine, donc une derive en continu."""
    if resistance <= 0:
        msg = f"resistance nulle ou negative : {resistance} ohm"
        raise ValueError(msg)
    if capacite <= 0:
        msg = f"capacite nulle ou negative : {capacite} F"
        raise ValueError(msg)
    return -1 / (resistance * capacite * variable)
