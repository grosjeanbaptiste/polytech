"""Les signaux elementaires, et la classe a laquelle chacun appartient.

Le cours ouvre sur cette classification, et elle n'est pas decorative :
elle decide de ce que l'on peut calculer. Un signal d'**energie finie** a
une energie et une puissance moyenne nulle ; un signal de **puissance
finie** a une puissance et une energie infinie. Confondre les deux fait
calculer une energie qui diverge sans que rien le signale -- sympy rend
`oo`, qui se compose parfaitement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import sympy as sp

from polytech.signaux.domain.symboles import TEMPS


class Classe(Enum):
    """Les deux classes du cours, et celle qui n'en est aucune."""

    ENERGIE_FINIE = "energie finie"
    PUISSANCE_FINIE = "puissance finie"
    NI_L_UN_NI_L_AUTRE = "ni energie ni puissance finie"


def echelon() -> sp.Expr:
    """L'echelon unite, defini sur `t >= 0` comme le cours le pose."""
    return sp.Heaviside(TEMPS)


def rampe() -> sp.Expr:
    """La rampe unite `t u(t)`."""
    return TEMPS * sp.Heaviside(TEMPS)


@dataclass(frozen=True, slots=True)
class Signal:
    """Un signal causal, donne par son expression en `t`."""

    expression: sp.Expr

    def sur_le_domaine(self) -> sp.Expr:
        """L'expression reduite sur `t >= 0`, ou l'echelon unite vaut un.

        Les deux integrales de ce module courent de zero : le facteur
        `Heaviside(t)` y vaut donc un partout, et le garder empeche
        seulement sympy de conclure -- il ne sait pas integrer
        `Heaviside(t)^2`. C'est la meme decision que pour l'inversion de
        Laplace dans `polytech.maths`, et pour la meme raison : le signal
        est causal, le domaine est la demi-droite positive.

        Un echelon **retarde** `Heaviside(t - tau)` n'est pas reduit : il
        dit a quel instant le signal commence, ce qui est une information
        de l'enonce et non une convention de domaine.
        """
        return self.expression.subs(sp.Heaviside(TEMPS), 1)

    def energie(self) -> sp.Expr:
        """L'integrale de `|x|^2` sur `[0, oo)` ; peut diverger."""
        return sp.integrate(sp.Abs(self.sur_le_domaine()) ** 2, (TEMPS, 0, sp.oo))

    def puissance_moyenne(self) -> sp.Expr:
        """La limite de l'energie sur la duree, quand la duree croit."""
        duree = sp.Symbol("T", positive=True)
        reduite = sp.Abs(self.sur_le_domaine()) ** 2
        partielle = sp.integrate(reduite, (TEMPS, 0, duree))
        return sp.limit(partielle / duree, duree, sp.oo)

    def classe(self) -> Classe:
        """La classe du cours a laquelle ce signal appartient."""
        if self.energie().is_finite:
            return Classe.ENERGIE_FINIE
        if self.puissance_moyenne().is_finite:
            return Classe.PUISSANCE_FINIE
        return Classe.NI_L_UN_NI_L_AUTRE

    def energie_verifiee(self) -> sp.Expr:
        """L'energie ; leve si elle diverge, au lieu de rendre `oo`."""
        energie = self.energie()
        if not energie.is_finite:
            msg = (
                f"energie infinie : {self.expression} n'est pas un signal "
                f"d'energie finie (classe {self.classe().value})"
            )
            raise ValueError(msg)
        return energie


DECIBELS_DE_PUISSANCE = 10
"""Le rapport signal sur bruit compare des puissances : 10 log, pas 20."""

GAIN_PAR_BIT = 6.0206
"""`20 log10(2)` : ce qu'un bit de plus apporte au rapport signal sur bruit."""

CONSTANTE_DU_QUANTIFICATEUR = 1.7609
"""`10 log10(3/2)` : le terme constant de la regle des six decibels par bit."""


def produit_scalaire(
    premier: sp.Expr,
    second: sp.Expr,
    duree: sp.Expr,
) -> sp.Expr:
    """`int_0^T x y dt` : l'operation dont l'orthogonalite decoule.

    C'est elle qui fonde les series de Fourier -- les harmoniques sont
    orthogonales deux a deux, et c'est pourquoi leurs coefficients se
    calculent independamment.
    """
    if sp.simplify(duree) == 0:
        msg = "duree nulle : le produit scalaire n'est pas defini"
        raise ValueError(msg)
    return sp.simplify(sp.integrate(premier * second, (TEMPS, 0, duree)))


def rapport_signal_sur_bruit_en_decibels(
    puissance_du_signal: float,
    puissance_du_bruit: float,
) -> float:
    """`10 log10(P_s / P_n)`, en decibels."""
    if puissance_du_bruit <= 0:
        msg = f"puissance de bruit nulle ou negative : {puissance_du_bruit}"
        raise ValueError(msg)
    if puissance_du_signal <= 0:
        msg = f"puissance de signal nulle ou negative : {puissance_du_signal}"
        raise ValueError(msg)
    return DECIBELS_DE_PUISSANCE * math.log10(puissance_du_signal / puissance_du_bruit)


def pas_de_quantification(etendue: float, bits: int) -> float:
    """`etendue / 2^n` : il se divise par deux a chaque bit ajoute."""
    if bits <= 0:
        msg = f"nombre de bits nul ou negatif : {bits}"
        raise ValueError(msg)
    if etendue <= 0:
        msg = f"etendue nulle ou negative : {etendue}"
        raise ValueError(msg)
    return etendue / float(2**bits)


def rapport_signal_sur_bruit_d_un_quantificateur(bits: int) -> float:
    """`6,02 n + 1,76` dB : la regle que le fascicule emploie sans la demontrer.

    Elle suppose un signal sinusoidal pleine echelle et une erreur de
    quantification uniforme. Douze bits donnent 74 dB, et c'est le nombre
    qu'un dimensionnement de chaine d'acquisition emploie directement.
    """
    if bits <= 0:
        msg = f"nombre de bits nul ou negatif : {bits}"
        raise ValueError(msg)
    return GAIN_PAR_BIT * bits + CONSTANTE_DU_QUANTIFICATEUR
