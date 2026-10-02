"""Les quatre circuits de la seance 3.1, tels que le recueil les dessine.

Chaque fonction decrit une figure de la section 3.2.3 du recueil
`Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`, dans l'etat qui suit
la commutation. Les conditions initiales sont portees par les elements
eux-memes : c'est la que le recueil les fait entrer, et c'est ce qui les
empeche d'etre oubliees.

L'exercice 7 a deux circuits. Le second repart de la fin de l'impulsion,
source eteinte, avec pour seule memoire la tension du condensateur a cet
instant : il donne la reponse sans passer par la superposition.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.symboles_seance_31 import (
    DUREE,
    L1,
    L2,
    L12,
    R1,
    R2,
    TENSION,
    C,
    R,
)
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
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE, original

ECHELON = TENSION / VARIABLE
"""Une source continue enclenchee a l'instant zero."""

BOBINE_DU_DEUX = Bobine("m", "masse", L2)
PRIMAIRE = Bobine("a", "masse", L1, courant_initial=TENSION / R)
"""Avant l'ouverture, l'interrupteur shunte `R_1` : le courant vaut `E / R`."""

SECONDAIRE = Bobine("c", "masse", L2)

MATRICE_DU_QUATRE = sp.Matrix(
    [
        [R + R1 + VARIABLE * L1, VARIABLE * L12, 0],
        [VARIABLE * L12, R2 + VARIABLE * L2, -1],
        [0, 1, VARIABLE * C],
    ],
)
"""La matrice que le recueil imprime en reponse a l'exercice 4."""

SECOND_MEMBRE_DU_QUATRE = sp.Matrix(
    [TENSION / VARIABLE + L1 * TENSION / R, L12 * TENSION / R, 0],
)
"""Son second membre : la source, et les deux flux initiaux."""


def circuit_du_deux() -> ReseauOperationnel:
    """Figure 3.2.11 : `R_1`, puis `C` en parallele avec `R_2`-`L_2`."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("e", "masse", ECHELON),
            Resistance("e", "n", R1),
            Capacite("n", "masse", C),
            Resistance("n", "m", R2),
            BOBINE_DU_DEUX,
        ),
        reference="masse",
    )


def circuit_du_quatre() -> ReseauOperationnel:
    """Figure 3.2.13, interrupteur ouvert : deux mailles couplees par `L_12`."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("e", "masse", ECHELON),
            Resistance("e", "a", R + R1),
            PRIMAIRE,
            Capacite("k", "masse", C),
            Resistance("k", "c", R2),
            SECONDAIRE,
        ),
        reference="masse",
        couplages=(Couplage(PRIMAIRE, SECONDAIRE, L12),),
    )


def circuit_du_sept() -> ReseauOperationnel:
    """Figure 3.2.16 : `C` en parallele avec `R_1`, puis `R_2` ou se lit `u`."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("g", "masse", ECHELON),
            Capacite("g", "n", C),
            Resistance("g", "n", R1),
            Resistance("n", "masse", R2),
        ),
        reference="masse",
    )


def circuit_du_sept_apres_l_impulsion() -> ReseauOperationnel:
    """Le meme circuit a partir de `t = T`, la source retombee a zero."""
    sortie = original(circuit_du_sept().potentiels()["n"]).subs(TEMPS, DUREE)
    return ReseauOperationnel(
        elements=(
            SourceDeTension("g", "masse", sp.Integer(0)),
            Capacite("g", "n", C, tension_initiale=TENSION - sortie),
            Resistance("g", "n", R1),
            Resistance("n", "masse", R2),
        ),
        reference="masse",
    )
