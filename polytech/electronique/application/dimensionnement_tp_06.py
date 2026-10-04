"""TP 6 d'electronique, exercices 4, 7, 8 et 9 : choisir et chiffrer un AOP.

La bande d'un montage se lit sur son pole en boucle fermee : l'AOP y a
le gain `A(s) = 2 pi GBW / s` (au-dela de son pole dominant), le montage
est resolu, et le pole se lit sur le denominateur. Les donnees des AOP
sont celles de l'enonce et de ses extraits de fiches techniques.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.domain.amplification import nombre_d_etages
from polytech.electronique.domain.aop import CONDUCTANCE, GAIN, aop
from polytech.electronique.domain.cascade_d_etages import (
    bande_de_la_cascade,
    nombre_d_etages_en_cascade,
)
from polytech.electronique.domain.limites import limite_a_l_infini
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
    resoudre,
)

S: Final = sp.Symbol("s")
VIN: Final = sp.Symbol("V_in")
_I: Final = sp.Symbol("I_S")
GBW_LM741: Final = 1.5e6
PENTE_LM741: Final = 0.5
"""`V/us`, la valeur typique de la fiche du LM741 que l'enonce reproduit."""
BANDE_AUDIO: Final = 20e3
PRODUITS_A_DEUX_MEGAHERTZ: Final = {"AD8129": 200e6, "OPA549": 0.9e6, "OPA132": 8e6}
"""Le tableau de l'exercice 9 : produit gain-bande de chaque AOP."""


def bande_d_un_etage(gain: float) -> float:
    """Exercice 4.1 : `GBW / G`, la regle du cours."""
    return GBW_LM741 / gain


def etages_selon_le_cours() -> int:
    """Exercice 4.2, chaque etage tenant `100 kHz` : la reponse officielle."""
    return nombre_d_etages(150, GBW_LM741, 100e3)


def etages_selon_la_chaine() -> int:
    """Exercice 4.2, la *chaine* tenant `100 kHz` : le retrecissement compte."""
    return nombre_d_etages_en_cascade(150, GBW_LM741, 100e3)


def bande_de_la_chaine(etages: int) -> float:
    """La bande de `etages` etages identiques qui font ensemble un gain 150."""
    return bande_de_la_cascade(etages, bande_d_un_etage(150 ** (1 / etages)))


def _transfert(circuit: list[Element], noeuds: int, sortie: int, gbw: float) -> sp.Expr:
    """`V_s / V_in`, AOP de gain `2 pi GBW / s`, resistance de sortie nulle."""
    tension = resoudre(circuit, noeuds)[sortie - 1] / VIN
    sans_sortie = limite_a_l_infini(tension, CONDUCTANCE)
    return sp.cancel(sans_sortie.subs(GAIN, 2 * sp.pi * sp.nsimplify(gbw) / S))


def bande_de_l_inverseur(r1: float, r2: float, gbw: float) -> float:
    """Exercice 7.2 : la frequence du pole de l'inverseur en boucle fermee."""
    circuit: list[Element] = [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, 1 / sp.nsimplify(r1)),
        Admittance(2, 3, 1 / sp.nsimplify(r2)),
        *aop(0, 2, 3),
    ]
    return pole_en_boucle_fermee(circuit, 3, 3, gbw)


def pole_en_boucle_fermee(
    circuit: list[Element],
    noeuds: int,
    sortie: int,
    gbw: float,
) -> float:
    """La frequence du pole unique du montage, s'il est stable.

    Une contre-reaction branchee sur l'entree `+` met ce pole dans le
    demi-plan droit : son module serait le meme, et le montage, instable.
    """
    _, denominateur = sp.fraction(_transfert(circuit, noeuds, sortie, gbw))
    (pole,) = sp.solve(denominateur, S)
    if not pole.is_negative:
        msg = f"pole instable en boucle fermee : {pole}"
        raise ValueError(msg)
    return float(-pole / (2 * sp.pi))


def aop_de_l_exercice_8() -> str:
    """Le moins cher des deux AOP qui tient le gain 10 jusqu'a `20 kHz`.

    Le LM741 (0,54 EUR) coute moins que le CA3140A (1,96 EUR) : il suffit
    de verifier qu'il tient la bande, et la pente qu'exige `1 V` a `20 kHz`.
    """
    tient_la_bande = bande_d_un_etage(10) >= BANDE_AUDIO
    tient_la_pente = 2 * math.pi * BANDE_AUDIO * 1.0 / 1e6 <= PENTE_LM741
    return "LM741" if tient_la_bande and tient_la_pente else "CA3140A"


def gains_maximaux_a_deux_megahertz() -> dict[str, float]:
    """Exercice 9 : le gain que chaque AOP tient encore a `2 MHz`."""
    return {nom: gbw / 2e6 for nom, gbw in PRODUITS_A_DEUX_MEGAHERTZ.items()}


def pente_requise() -> float:
    """`2 pi f A` en `V/us` : `4 V` de crete a `2 MHz`."""
    return 2 * math.pi * 2e6 * 4 / 1e6


def suiveur_opa549_a_deux_megahertz() -> tuple[float, float]:
    """Le module et la phase (degres) du suiveur OPA549 a `2 MHz`."""
    circuit: list[Element] = [SourceDeTension(1, 0, VIN, _I), *aop(1, 2, 2)]
    reponse = complex(
        _transfert(circuit, 2, 2, 0.9e6).subs(S, sp.I * 2 * sp.pi * 2e6).evalf()
    )
    return abs(reponse), math.degrees(math.atan2(reponse.imag, reponse.real))
