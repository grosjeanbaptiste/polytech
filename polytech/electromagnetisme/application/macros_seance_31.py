"""Macros de la seance 3.1, exercices 2, 4 et 7 : les litteraux.

Chaque transformee vient du resolveur de reseau en domaine de Laplace,
qui ecrit la loi des noeuds et ne choisit aucune maille. Le manuscrit,
lui, ecrit ses mailles puis substitue : que les deux aboutissent aux
memes expressions est ce que le document montre.

Pour l'exercice 7, la reponse temporelle est publiee **par morceaux** --
le palier, l'excedent, l'amortissement -- et non d'un bloc : c'est sous
cette forme qu'elle se lit, et l'inversion symbolique la rend autrement.

Les prefixes sont `TroisUnDeux`, `TroisUnQuatre` et `TroisUnSept`.
L'exercice 1, numerique, est dans `macros_seance_31_relais`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.enonces_seance_31 import (
    BOBINE_DU_DEUX,
    MATRICE_DU_QUATRE,
    PRIMAIRE,
    SECOND_MEMBRE_DU_QUATRE,
    SECONDAIRE,
    circuit_du_deux,
    circuit_du_quatre,
    circuit_du_sept,
    circuit_du_sept_apres_l_impulsion,
)
from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.application.macros_seance_31_relais import (
    macros_du_relais,
)
from polytech.electromagnetisme.application.symboles_seance_31 import DUREE
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    original,
    valeur_finale,
    valeur_initiale,
)


def _macros_du_deux() -> dict[str, str]:
    """La transformee de la tension, et ses deux bouts."""
    reseau = circuit_du_deux()
    tension = reseau.potentiels()["n"]
    courant = reseau.courant(BOBINE_DU_DEUX)
    return {
        "TroisUnDeuxTransformee": expression_latex(sp.factor(tension)),
        "TroisUnDeuxRegime": expression_latex(sp.simplify(valeur_finale(tension))),
        "TroisUnDeuxTensionInitiale": expression_latex(valeur_initiale(tension)),
        "TroisUnDeuxCourantDeRegime": expression_latex(
            sp.simplify(valeur_finale(courant)),
        ),
    }


def _macros_du_quatre() -> dict[str, str]:
    """Le systeme du recueil, puis ce que valent ses inconnues aux deux bouts."""
    reseau = circuit_du_quatre()
    primaire = reseau.courant(PRIMAIRE)
    secondaire = reseau.courant(SECONDAIRE)
    tension = reseau.potentiels()["k"]
    ordre = sp.degree(sp.denom(sp.together(secondaire)), VARIABLE)
    return {
        "TroisUnQuatreMatrice": expression_latex(MATRICE_DU_QUATRE),
        "TroisUnQuatreSecondMembre": expression_latex(SECOND_MEMBRE_DU_QUATRE),
        "TroisUnQuatrePrimaireInitial": expression_latex(
            sp.simplify(valeur_initiale(primaire)),
        ),
        "TroisUnQuatrePrimaireFinal": expression_latex(
            sp.simplify(valeur_finale(primaire)),
        ),
        "TroisUnQuatreSecondaireFinal": expression_latex(
            sp.simplify(valeur_finale(secondaire)),
        ),
        "TroisUnQuatreTensionFinale": expression_latex(
            sp.simplify(valeur_finale(tension)),
        ),
        "TroisUnQuatreOrdre": str(ordre),
    }


def _macros_du_sept() -> dict[str, str]:
    """La transformee, puis la reponse en ses trois morceaux."""
    transformee = circuit_du_sept().potentiels()["n"]
    reponse = original(transformee)
    palier = sp.limit(reponse, TEMPS, sp.oo)
    apres = original(circuit_du_sept_apres_l_impulsion().potentiels()["n"])
    return {
        "TroisUnSeptTransformee": expression_latex(sp.factor(transformee)),
        "TroisUnSeptDepart": expression_latex(valeur_initiale(transformee)),
        "TroisUnSeptPalier": expression_latex(sp.simplify(palier)),
        "TroisUnSeptExcedent": expression_latex(
            sp.simplify(reponse.subs(TEMPS, 0) - palier),
        ),
        "TroisUnSeptAmortissement": expression_latex(
            sp.simplify(-sp.diff(sp.log(apres), TEMPS)),
        ),
        "TroisUnSeptSaut": expression_latex(
            sp.simplify(reponse.subs(TEMPS, DUREE) - apres.subs(TEMPS, 0)),
        ),
    }


def macros_de_la_seance_trois_un() -> dict[str, str]:
    """Les quatre exercices de la seance."""
    return {
        **macros_du_relais(),
        **_macros_du_deux(),
        **_macros_du_quatre(),
        **_macros_du_sept(),
    }
