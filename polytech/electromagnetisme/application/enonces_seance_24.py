"""Les cinq reseaux de la seance 2.4, tels que le recueil les dessine.

Chaque fonction decrit une figure de la section 2.3.3 du recueil
`Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`. Les noeuds `A` et `B`
sont ses bornes ; les autres portent un nom de circonstance.

Le sens des sources est celui des figures : la borne « plus » d'une pile
est son grand trait, et une source de courant debite dans le sens de sa
fleche. Les deux sont **verifies** contre les reponses imprimees -- un
sens inverse changerait un signe du resultat, et le test le verrait.

Ce fichier porte des donnees, non de la logique : une figure par
fonction, un element par ligne.
"""

from __future__ import annotations

from polytech.electromagnetisme.application.symboles_seance_24 import (
    CHARGE,
    CHARGE_PRIME,
    COURANT,
    COURANT_3,
    COURANT_PRIME,
    R1,
    R2,
    R3,
    R4,
    R5,
    TENSION,
    TENSION_1,
    TENSION_2,
    TENSION_5,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Condensateur,
    Element,
    Inductance,
    Reseau,
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)

SOURCE_CINQ = SourceDeTension("B", "h", TENSION_5)
"""La source `E_5` de l'exercice 6, nommee pour qu'on puisse lire son courant."""


def reseau_du_deux(*, charge: bool = False) -> Reseau:
    """Figure 2.3.16 : deux sources reelles en parallele, et la charge `R`."""
    elements: tuple[Element, ...] = (
        SourceDeTension("a", "B", TENSION_1),
        Resistance("a", "A", R1),
        SourceDeTension("b", "B", TENSION_2),
        Resistance("b", "A", R2),
    )
    if charge:
        elements = (*elements, Resistance("A", "B", CHARGE))
    return Reseau(elements=elements, reference="B")


def reseau_du_trois() -> Reseau:
    """Figure 2.3.17 : `R_2` puis `R_3` doublee d'une source de courant."""
    return Reseau(
        elements=(
            Resistance("A", "m", R2),
            Resistance("m", "B", R3),
            SourceDeCourant("B", "m", COURANT_3),
            SourceDeTension("A", "n", TENSION_1),
            Resistance("n", "B", R1),
        ),
        reference="B",
    )


def reseau_du_quatre() -> Reseau:
    """Figure 2.3.18 : le circuit entier, inductance et condensateur compris."""
    return Reseau(
        elements=(
            SourceDeTension("e", "B", TENSION_1),
            Resistance("e", "N", R1),
            Resistance("N", "c", R2),
            Condensateur("c", "B"),
            Resistance("N", "f", R3),
            SourceDeTension("f", "B", TENSION_2),
            Resistance("N", "A", R4),
            Inductance("N", "A"),
            SourceDeCourant("A", "B", COURANT),
        ),
        reference="B",
    )


def reseau_du_six() -> Reseau:
    """Figure 2.3.20 : `E_1` entre `B` et le rail du bas, `E_5`-`R_5` a cote."""
    return Reseau(
        elements=(
            SourceDeTension("B", "g", TENSION_1),
            SOURCE_CINQ,
            Resistance("h", "g", R5),
            Resistance("B", "t", R1),
            Resistance("t", "A", R3),
            Resistance("A", "g", R4),
        ),
        reference="B",
    )


def reseau_charge_du_six() -> Reseau:
    """Figures 2.3.20 et 2.3.21 reunies : la charge `R'` et sa source `I'`."""
    nu = reseau_du_six()
    branche = (
        Resistance("A", "B", CHARGE_PRIME),
        SourceDeCourant("A", "B", COURANT_PRIME),
    )
    return Reseau(elements=(*nu.elements, *branche), reference=nu.reference)


def reseau_du_neuf() -> Reseau:
    """Figure 2.3.24 (a) : l'inductance double la branche `R_2`-`R_1`."""
    return Reseau(
        elements=(
            SourceDeTension("P", "q", TENSION),
            Resistance("q", "B", R3),
            SourceDeCourant("B", "q", COURANT),
            Resistance("P", "m", R2),
            Resistance("m", "A", R1),
            Inductance("P", "A"),
            Resistance("A", "B", R4),
        ),
        reference="B",
    )
