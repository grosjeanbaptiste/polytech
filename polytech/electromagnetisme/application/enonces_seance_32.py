"""Les circuits de la seance 3.2 en regime sinusoidal : exercices 1, 5, 6.

Chaque circuit est un reseau operationnel dont les sources sont des
**phaseurs efficaces** constants ; sa restriction a `p = j omega` donne le
regime etabli. Le resolveur ecrit la loi des noeuds, et ne choisit
aucune maille : c'est ce qui le rend independant du manuscrit.

Les conventions sont lues sur les figures du recueil :

- 3.3.13 : `i` descend dans `R` ;
- 3.3.17 : `i_1` monte dans la branche de `e_1`, `i_3` entre par le haut
  dans la branche de `e_3` et traverse la source du `+` vers le `-` ;
- 3.3.18 : `i_1` entre dans `L_1` par son point, `i_2` **sort** de `L_2`
  par le sien. La secondaire est donc declaree de son point vers l'autre
  borne. Et `i_2` est lu la ou la figure le dessine, dans `R_2` : le lire
  sur la bobine lierait le sens de lecture au point, et une erreur sur
  le point s'annulerait avec une erreur sur le sens.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import symboles_seance_32 as d
from polytech.electromagnetisme.domain.circuits.elements_operationnels import (
    Bobine,
    Capacite,
    Couplage,
)
from polytech.electromagnetisme.domain.circuits.operationnel import (
    ReseauOperationnel,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.circuits.sinusoidaux import (
    en_parallele,
    impedance_capacitive,
)
from polytech.electromagnetisme.domain.symboles import PULSATION


def circuit_du_un(retard_en_degres: sp.Expr = d.UN_RETARD) -> ReseauOperationnel:
    """Figure 3.3.13 : deux alternateurs et leurs inductances, sur `R`.

    Le retard de `e_2` est un parametre pour une seule raison : le
    document compare le partage de la puissance avec et sans lui.
    """
    retard = sp.exp(-sp.I * sp.pi * retard_en_degres / 180)
    return ReseauOperationnel(
        elements=(
            SourceDeTension("s1", "masse", d.UN_E1),
            Bobine("s1", "h", d.UN_L1),
            SourceDeTension("s2", "masse", d.UN_E2 * retard),
            Bobine("s2", "h", d.UN_L2),
            Resistance("h", "masse", d.UN_R),
        ),
        reference="masse",
    )


BOBINE_DU_CINQ = Bobine("r", "h", d.L1)
SOURCE_DU_CINQ = SourceDeTension("k", "masse", d.E3)


def circuit_du_cinq() -> ReseauOperationnel:
    """Figure 3.3.17, litterale : le systeme que l'exercice demande."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("s", "masse", d.E1),
            Resistance("s", "r", d.R1),
            BOBINE_DU_CINQ,
            Resistance("h", "masse", d.R2),
            Capacite("h", "k", d.C3),
            Resistance("h", "k", d.R3),
            SOURCE_DU_CINQ,
        ),
        reference="masse",
    )


def impedance_du_cinq() -> sp.Expr:
    """`Z_3 = R_3 // C_3` a 50 Hz."""
    parallele = en_parallele(d.CINQ_R3, impedance_capacitive(d.CINQ_C3))
    return parallele.subs(PULSATION, d.OMEGA_DU_CINQ)


def branche_du_cinq() -> tuple[sp.Expr, sp.Expr]:
    """La tension de la branche `e_3`-`R_3`-`C_3`, et son courant `i_3`."""
    courant = d.COURANT_TROIS_DU_CINQ
    return d.CINQ_E3 + impedance_du_cinq() * courant, courant


PRIMAIRE_DU_SIX = Bobine("h1", "masse", d.SIX_L1)
SECONDAIRE_DU_SIX = Bobine("masse", "h2", d.SIX_L2)


def circuit_du_six() -> ReseauOperationnel:
    """Figure 3.3.18 : le primaire alimente, le secondaire `R_2`-`C_2`."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("h1", "m", d.SIX_E1),
            Resistance("m", "masse", d.SIX_R1),
            PRIMAIRE_DU_SIX,
            SECONDAIRE_DU_SIX,
            Resistance("h2", "r", d.SIX_R2),
            Capacite("r", "masse", d.SIX_C2),
        ),
        couplages=(Couplage(PRIMAIRE_DU_SIX, SECONDAIRE_DU_SIX, d.SIX_L12),),
        reference="masse",
    )


def courant_secondaire_du_six() -> sp.Expr:
    """`i_2`, lu dans `R_2` ou la figure le dessine : de `C_2` vers `L_2`."""
    potentiels = circuit_du_six().potentiels()
    return (potentiels["r"] - potentiels["h2"]) / d.SIX_R2
