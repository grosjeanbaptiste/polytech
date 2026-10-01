"""La reponse indicielle d'une boucle fermee, lue sur ses caracteristiques.

Separe de `performances`, qui porte les **erreurs de regime**, parce que ce
module porte l'**allure** : depassement, amortissement, pic de resonance,
bande passante, temps de montee. Les deux repondent a des questions
differentes -- l'une « ou finit-on », l'autre « comment y va-t-on » -- et
le fichier unique depassait la limite de taille du depot.

Ancrage direct sur `Exs_08_Performances_Reponses` : la feuille publie
`|M(0)| = -3 dB` et `y(infini) = 0,707`, puis `D = 87 %` et
`y_max = 1,32`. Les deux couples se recoupent exactement.

Ce que ce module ne porte **pas**, et c'est delibere : le passage du pic de
resonance au depassement par les abaques de Black-Nichols. Le cours les lit
sur un abaque ; le seul lien en forme close est celui du second ordre, et il
est expose comme tel.
"""

from __future__ import annotations

import math

import sympy as sp

CENT = 100

AMORTISSEMENT_FAIBLE = 0.3
"""En dessous, la reponse oscille visiblement : « faiblement amorti »."""

AMORTISSEMENT_CRITIQUE = 1.0
"""Au-dela, plus aucune oscillation : « amorti »."""

PIC_IMPOSSIBLE = 0.7071067811865476
"""`1/racine(2)` : au-dela de cet amortissement, il n'y a plus de pic.

La reponse frequentielle devient monotone, et chercher un maximum de
resonance n'a plus de sens. Le fascicule le montre sur sa troisieme
illustration -- le systeme amorti.
"""

MONTEE_BASSE = 1.9
"""`t_m omega_BP` minimal observe sur les systemes du fascicule."""

MONTEE_HAUTE = 2.8
"""Et maximal. La relation encadre, elle ne donne pas une valeur unique."""


def depassement_du_second_ordre(amortissement: sp.Expr) -> sp.Expr:
    """`D = exp(-pi zeta / racine(1 - zeta^2))`, formule du second ordre."""
    zeta = sp.sympify(amortissement)
    if not sp.simplify(zeta > 0) or not sp.simplify(zeta < 1):
        msg = (
            f"amortissement hors de ]0, 1[ : {zeta} -- le depassement n'est "
            f"defini que pour un second ordre oscillant"
        )
        raise ValueError(msg)
    return sp.exp(-sp.pi * zeta / sp.sqrt(1 - zeta**2))


def amortissement_du_second_ordre(depassement: sp.Expr) -> sp.Expr:
    """L'inverse de la relation precedente, pour remonter de `D` a `zeta`."""
    logarithme = sp.log(sp.sympify(depassement))
    return sp.simplify(-logarithme / sp.sqrt(sp.pi**2 + logarithme**2))


def maximum_de_la_reponse(valeur_finale: sp.Expr, depassement: sp.Expr) -> sp.Expr:
    """`y_max = y(infini) (1 + D)`, avec `D` en fraction et non en pourcent."""
    return sp.simplify(valeur_finale * (1 + depassement))


def en_pourcent(fraction: sp.Expr) -> sp.Expr:
    """Le depassement tel que la feuille de reponses l'ecrit."""
    return sp.simplify(CENT * fraction)


def classe_d_amortissement(amortissement: float) -> str:
    """La classe que le fascicule illustre en trois planches.

    Les seuils sont dits : `0,3` et `1`. Le premier est conventionnel --
    au-dessous, le depassement passe les trente pour cent et s'impose a
    l'oeil ; le second est exact, c'est l'amortissement critique.
    """
    if amortissement <= 0:
        msg = f"amortissement nul ou negatif : {amortissement}"
        raise ValueError(msg)
    if amortissement < AMORTISSEMENT_FAIBLE:
        return "faiblement amorti"
    if amortissement < AMORTISSEMENT_CRITIQUE:
        return "moderement amorti"
    return "amorti"


def bande_passante_du_second_ordre(
    pulsation_propre: float,
    amortissement: float,
) -> float:
    """La pulsation a -3 dB d'un second ordre, en rad/s.

    `omega_0 racine(1 - 2 zeta^2 + racine(2 - 4 zeta^2 + 4 zeta^4))`. Elle
    est proportionnelle a `omega_0` a amortissement fixe -- c'est ce qui
    permet de regler la rapidite sans toucher a l'allure.
    """
    if amortissement <= 0:
        msg = f"amortissement nul ou negatif : {amortissement}"
        raise ValueError(msg)
    if pulsation_propre <= 0:
        msg = f"pulsation propre nulle ou negative : {pulsation_propre}"
        raise ValueError(msg)
    carre = amortissement**2
    sous_racine = 1 - 2 * carre + math.sqrt(2 - 4 * carre + 4 * carre**2)
    return pulsation_propre * math.sqrt(sous_racine)


def pic_de_resonance_du_second_ordre(amortissement: float) -> float:
    """`1 / (2 zeta racine(1 - zeta^2))` ; leve s'il n'y a pas de pic.

    Au-dela de `1/racine(2)` la reponse frequentielle est monotone : il
    n'existe aucun maximum, et rendre un nombre serait rendre un nombre qui
    ne correspond a rien.
    """
    if amortissement <= 0:
        msg = f"amortissement nul ou negatif : {amortissement}"
        raise ValueError(msg)
    if amortissement >= PIC_IMPOSSIBLE:
        msg = (
            f"pas de pic de resonance : zeta = {amortissement} depasse "
            f"1/racine(2), la reponse frequentielle est monotone"
        )
        raise ValueError(msg)
    return 1 / (2 * amortissement * math.sqrt(1 - amortissement**2))


def estimation_du_depassement_depuis_le_pic(pic: float) -> float:
    """Du pic de resonance au depassement indiciel, via l'amortissement.

    « Et si on pouvait estimer la reponse indicielle BF ?! », demande la
    planche. Le chemin est celui-ci : le pic donne `zeta`, et `zeta` donne
    le depassement. Les deux relations sont **exactes** pour un second
    ordre et approchees au-dela -- ce qui est dit, et ce qui en fait une
    estimation plutot qu'un calcul.
    """
    if pic < 1:
        msg = f"pic sous l'unite : {pic} -- un pic de resonance amplifie"
        raise ValueError(msg)
    if pic == 1:
        return 0.0
    carre = (1 - math.sqrt(1 - 1 / pic**2)) / 2
    amortissement = math.sqrt(carre)
    return float(depassement_du_second_ordre(sp.Float(amortissement)))


def temps_de_montee_depuis_la_bande(bande: float) -> tuple[float, float]:
    """Les deux bornes du temps de montee, en secondes.

    `t_m omega_BP` n'a pas de valeur unique : il depend de l'allure, donc
    de l'amortissement. Le fascicule l'encadre, et c'est pourquoi cette
    fonction rend un couple -- rendre une valeur unique serait inventer
    une precision que la relation n'a pas.
    """
    if bande <= 0:
        msg = f"bande passante nulle ou negative : {bande} rad/s"
        raise ValueError(msg)
    return MONTEE_BASSE / bande, MONTEE_HAUTE / bande


def minimum_de_resonance(modules: tuple[float, ...]) -> float:
    """Le `|M|min` que le fascicule prevoit « en cas de presence ».

    Il n'existe pas toujours : un second ordre n'en a pas, et une courbe
    monotone non plus. Il faut un creux **interieur** -- un module plus bas
    que ses deux voisins -- et le dire vaut mieux que de rendre le plus
    petit element d'une suite decroissante.
    """
    minimum = 3
    if len(modules) < minimum:
        msg = f"moins de trois points : {len(modules)}, aucun creux ne se voit"
        raise ValueError(msg)
    creux = [
        valeur
        for rang, valeur in enumerate(modules[1:-1], start=1)
        if valeur < modules[rang - 1] and valeur < modules[rang + 1]
    ]
    if not creux:
        msg = "aucun minimum interieur : la courbe n'a pas de creux"
        raise ValueError(msg)
    return min(creux)
