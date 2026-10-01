"""Les circuits a courants periodiques en general -- chapitre 17 du syllabus.

Il n'etait pas porte : la bibliotheque traitait le sinusoidal, et le
periodique **non** sinusoidal est autre chose. C'est la que la puissance
deformante apparait, et elle n'existe pas en sinusoidal pur -- ce qui en
fait le marqueur du chapitre.

Le piege du chapitre est que **les valeurs efficaces ne s'additionnent
pas** : elles se composent en carres, par orthogonalite des harmoniques.
Un courant continu de 3 A portant un fondamental de 4 A efficaces vaut
5 A efficaces, et non 7.

Le syllabus avertit lui-meme que la puissance reactive « fait l'objet
dans la litterature de definitions differentes, qui semblent convenir,
l'une mieux que l'autre, selon le type de probleme rencontre ». Il en
donne deux :

* la premiere complete simplement l'active -- `Q^2 = S^2 - P^2` ;
* la seconde distingue la reactive du **fondamental** de ce que les
  harmoniques apportent, et nomme le reste puissance **deformante** --
  `Q_d^2 = Q^2 - Q_1^2`, pour laquelle « on a propose l'unite : vad ».

Les deux se reunissent dans une identite qui enferme tout le chapitre :
`S^2 = P^2 + Q_1^2 + D^2`. Elle est la verification la plus forte de ce
module, parce qu'une faute sur n'importe laquelle des trois puissances
s'y verrait.

La decomposition du courant va avec : le syllabus pose `i_p` en phase et
`i_q` en quadrature, avec `P = U I_p` et `Q = U I_q`, et note que `i_p`
est « l'homologue du courant actif (ou watte) » du sinusoidal.
"""

from __future__ import annotations

import math

FACTEUR_MAXIMAL = 1.0
"""`|lambda| <= 1` : le syllabus le demontre, ce n'est pas une precaution."""


def valeur_efficace_des_harmoniques(
    continue_: float,
    harmoniques: tuple[float, ...],
) -> float:
    """`racine(I_0^2 + somme I_n^2)` : les contributions en carres."""
    return math.sqrt(continue_**2 + sum(h**2 for h in harmoniques))


def puissance_apparente(tension_efficace: float, courant_efficace: float) -> float:
    """`S = U I`, qui « reste definie par le produit des valeurs efficaces »."""
    return tension_efficace * courant_efficace


def facteur_de_puissance(active: float, apparente: float) -> float:
    """`lambda = P / S`, qui ne depasse jamais l'unite.

    La garde porte un resultat du cours : une active superieure a
    l'apparente serait une faute de mesure ou de calcul, jamais un
    circuit.
    """
    if apparente <= 0:
        msg = f"puissance apparente nulle ou negative : {apparente} VA"
        raise ValueError(msg)
    facteur = active / apparente
    if abs(facteur) > FACTEUR_MAXIMAL:
        msg = f"facteur de puissance au-dela de l'unite : {facteur:.4g}"
        raise ValueError(msg)
    return facteur


def puissance_reactive_globale(apparente: float, active: float) -> float:
    """`Q = racine(S^2 - P^2)`, la premiere definition du syllabus."""
    if abs(active) > abs(apparente):
        msg = (
            f"active superieure a l'apparente : {active} W pour "
            f"{apparente} VA -- la reactive serait imaginaire"
        )
        raise ValueError(msg)
    return math.sqrt(apparente**2 - active**2)


def puissance_deformante(globale: float, fondamentale: float) -> float:
    """`D = racine(Q^2 - Q_1^2)` : ce que le fondamental n'explique pas.

    Nulle en sinusoidal pur, ou la reactive globale **est** celle du
    fondamental. Toute valeur non nulle signale donc un courant non
    sinusoidal, et c'est ce qui en fait le marqueur du chapitre.
    """
    if abs(fondamentale) > abs(globale):
        msg = (
            f"reactive du fondamental superieure a la globale : "
            f"{fondamentale} contre {globale} -- la part ne depasse pas le tout"
        )
        raise ValueError(msg)
    return math.sqrt(globale**2 - fondamentale**2)


def courant_actif(active: float, tension_efficace: float) -> float:
    """`I_p = P / U` : la composante en phase, dite wattee."""
    if tension_efficace <= 0:
        msg = f"tension efficace nulle ou negative : {tension_efficace} V"
        raise ValueError(msg)
    return active / tension_efficace


def courant_reactif(reactive: float, tension_efficace: float) -> float:
    """`I_q = Q / U` : la composante en quadrature, dite dewattee."""
    if tension_efficace <= 0:
        msg = f"tension efficace nulle ou negative : {tension_efficace} V"
        raise ValueError(msg)
    return reactive / tension_efficace


def taux_de_distorsion(fondamental: float, harmoniques: tuple[float, ...]) -> float:
    """Les harmoniques de rang deux et plus, rapportees au fondamental.

    Grandeur des normes de qualite de reseau, et elle se lit
    directement : 30 % sur un fondamental de 10 A veut dire 3 A repartis
    dans les harmoniques.
    """
    if fondamental <= 0:
        msg = f"fondamental nul ou negatif : {fondamental}"
        raise ValueError(msg)
    return math.sqrt(sum(h**2 for h in harmoniques)) / fondamental
