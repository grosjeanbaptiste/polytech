"""Le circuit du relais -- seance 3.1, exercice 1.

Le seul exercice numerique de la seance, et le seul du premier ordre :
une source, trois resistances et la bobine du relais, en serie. Il vit a
part des trois autres circuits parce qu'il porte, en plus de sa figure,
le calcul de l'instant d'enclenchement.

Cet instant se lit sur la transformee sans revenir dans le temps : la
valeur finale par le theoreme du meme nom, et la constante de temps par
le rapport de l'inductance a la resistance totale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.electromagnetisme.application.symboles_seance_31 import (
    FEM_DU_RELAIS,
    INDUCTANCE_DE_LA_BOBINE,
    RESISTANCE_DE_LA_BOBINE,
    RESISTANCE_DE_LIGNE,
    RESISTANCE_INTERNE,
    SEUIL_DU_RELAIS,
)
from polytech.electromagnetisme.domain.circuits.elements_operationnels import Bobine
from polytech.electromagnetisme.domain.circuits.operationnel import (
    ReseauOperationnel,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.circuits.transitoires import instant_du_seuil
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    valeur_finale,
)

if TYPE_CHECKING:
    import sympy as sp

BOBINE_DU_RELAIS = Bobine("b", "masse", INDUCTANCE_DE_LA_BOBINE)

RESISTANCE_TOTALE = RESISTANCE_INTERNE + RESISTANCE_DE_LIGNE + RESISTANCE_DE_LA_BOBINE
"""Les trois resistances sont en serie : c'est leur somme qui freine."""

CONSTANTE_DE_TEMPS = INDUCTANCE_DE_LA_BOBINE / RESISTANCE_TOTALE
"""`tau = L / R` : la duree caracteristique de l'etablissement."""


def circuit_du_relais() -> ReseauOperationnel:
    """Figure 3.2.10 : la source, la ligne et la bobine du relais, en serie."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("s", "masse", FEM_DU_RELAIS / VARIABLE),
            Resistance("s", "u", RESISTANCE_INTERNE),
            Resistance("u", "v", RESISTANCE_DE_LIGNE),
            Resistance("v", "b", RESISTANCE_DE_LA_BOBINE),
            BOBINE_DU_RELAIS,
        ),
        reference="masse",
    )


def courant_final_du_relais() -> sp.Expr:
    """La valeur vers laquelle le courant s'etablit, lue sur la transformee."""
    return valeur_finale(circuit_du_relais().courant(BOBINE_DU_RELAIS))


def instant_d_enclenchement() -> sp.Expr:
    """L'instant ou le courant du relais atteint son seuil."""
    return instant_du_seuil(
        courant_final_du_relais(),
        CONSTANTE_DE_TEMPS,
        SEUIL_DU_RELAIS,
    )
