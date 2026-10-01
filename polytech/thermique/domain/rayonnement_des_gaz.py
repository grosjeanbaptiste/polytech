"""Le rayonnement des gaz de combustion : loi de Beer, opacite, trajet moyen.

Chapitre 4 du tome 1 du syllabus, et il n'etait pas dans la bibliotheque
parce que la mesure de couverture ne regardait que les planches. Il sert
les chambres de combustion et les ecrans de tubes -- la ou le gaz n'est
plus un milieu transparent entre deux parois, mais un emetteur et un
absorbeur a part entiere.

Ce que le syllabus etablit en clair, et qui est implemente ici :

* la **loi de Beer** -- un rayon monochromatique traversant un gaz
  absorbant perd `a_lambda f_lambda dx` sur un trajet elementaire, donc
  decroit exponentiellement avec la longueur ;
* le coefficient lineaire est **proportionnel a la pression partielle**,
  « c'est-a-dire a la concentration en molecules » ;
* le produit `p l` est l'**opacite** du milieu diathermane.

La consequence est ce qui rend les abaques lisibles : l'exposant de Beer
valant `k p l`, l'absorption ne depend que de l'opacite, et non du couple
`(p, l)` separement. Deux melanges de meme opacite absorbent pareil.

Ce que le syllabus ne donne qu'en **abaque** -- le facteur d'emission
`epsilon_g` d'un volume gazeux en fonction de l'opacite et de la
temperature (4.3.2), et le rayon moyen equivalent (4.4) -- n'est pas
interpole ici. Le trajet moyen `4 V / S` est la forme classique, et elle
n'est pas prise sur parole : deux geometries dont le resultat est connu
la verifient.
"""

from __future__ import annotations

import math


def _exiger_le_trajet(coefficient: float, longueur: float) -> None:
    """Les deux gardes de la loi de Beer, ecrites une fois."""
    if coefficient < 0:
        msg = f"coefficient d'absorption negatif : {coefficient} m^-1"
        raise ValueError(msg)
    if longueur < 0:
        msg = f"longueur de trajet negative : {longueur} m"
        raise ValueError(msg)


def transmittance(coefficient: float, longueur: float) -> float:
    """`exp(-a l)` : la fraction du rayon qui ressort du gaz.

    Elle decroit **exponentiellement** et non lineairement, si bien que
    doubler le trajet eleve la transmittance au carre. C'est l'erreur
    qu'un raisonnement proportionnel ferait, et elle est du cote
    optimiste : elle sous-estime l'absorption des longs trajets.
    """
    _exiger_le_trajet(coefficient, longueur)
    return math.exp(-coefficient * longueur)


def absorptivite_du_trajet(coefficient: float, longueur: float) -> float:
    """`1 - exp(-a l)` : la fraction que le gaz absorbe.

    Le modele du syllabus ne connait que deux issues -- transmis ou
    absorbe -- parce qu'il traite un milieu absorbant et non diffusant.
    La somme des deux vaut donc un exactement.
    """
    return 1.0 - transmittance(coefficient, longueur)


def opacite(pression_partielle: float, longueur: float) -> float:
    """`p l` : « l'opacite du milieu diathermane », dit le syllabus (4.2.1).

    Deux facteurs pour une grandeur : un gaz peu concentre sur un long
    trajet absorbe autant qu'un gaz concentre sur un trajet court. C'est
    ce qui permet aux abaques d'experience de n'avoir qu'une variable.
    """
    if pression_partielle < 0:
        msg = f"pression partielle negative : {pression_partielle} bar"
        raise ValueError(msg)
    if longueur < 0:
        msg = f"longueur de trajet negative : {longueur} m"
        raise ValueError(msg)
    return pression_partielle * longueur


def coefficient_lineaire(constante: float, pression_partielle: float) -> float:
    """`a = k p` : le coefficient d'absorption croit avec la concentration.

    Le syllabus l'ecrit ainsi : « le coefficient d'absorption lineaire
    pour un gaz etant proportionnel a la concentration en molecules,
    c'est-a-dire a la pression partielle p ».
    """
    if pression_partielle < 0:
        msg = f"pression partielle negative : {pression_partielle} bar"
        raise ValueError(msg)
    if constante < 0:
        msg = f"constante d'absorption negative : {constante}"
        raise ValueError(msg)
    return constante * pression_partielle


def longueur_moyenne_de_rayonnement(volume: float, surface: float) -> float:
    """`4 V / S` : le trajet moyen d'un rayon dans le volume gazeux.

    Le syllabus donne le rayon moyen equivalent en abaque et traite a
    part le volume hemispherique. L'expression `4 V / S` est la forme
    classique de ce trajet moyen, et elle est **verifiee** plutot que
    citee : sur une sphere elle rend `4 R / 3`, sur un cube `2 a / 3`.
    """
    if surface <= 0:
        msg = f"surface d'echange nulle ou negative : {surface} m^2"
        raise ValueError(msg)
    if volume <= 0:
        msg = f"volume gazeux nul ou negatif : {volume} m^3"
        raise ValueError(msg)
    return 4.0 * volume / surface
