"""L'aimant permanent optimal -- section 12.2 du syllabus.

« Pour realiser une induction donnee `B_e` dans un entrefer donne de
volume `V_e`, on aura donc a utiliser le plus petit volume `V` d'un
materiau donne si le point de fonctionnement est tel que le produit
`-H B` passe par un maximum (`H` est negatif). » Le syllabus en donne
aussi la lecture graphique : « le sommet de celui des rectangles inscrits
dans la courbe de desaimantation qui a l'aire maximale ».

Sur la droite de desaimantation -- le modele le plus simple qui respecte
les deux specifications, joignant `B_r` a champ nul et zero au champ
coercitif -- le maximum se calcule : il est a **la moitie** de chacune,
et le produit y vaut `B_r H_c / 4`.

Le resultat est elegant et il n'est pas evident : un aimant bien
dimensionne travaille a la moitie de son induction remanente, et non
pres de celle-ci. Les tests ne le prennent pas sur parole -- ils
comparent la forme fermee a une maximisation numerique du produit le
long de la droite.

Le syllabus nomme enfin les trois specifications d'un materiau pour
aimant permanent, et le fait que `(HB)max` en soit une est la
consequence directe de tout ce qui precede : il resume a lui seul la
capacite du materiau.
"""

from __future__ import annotations

from typing import Final

SPECIFICATIONS: Final = (
    "le produit d'energie maximal (HB)max",
    "l'induction remanente B_r, a champ nul",
    "le champ coercitif H_c, ou l'induction s'annule",
)
"""Les trois valeurs qu'une fiche technique donne, selon le syllabus."""

FRACTION_OPTIMALE: Final = 0.5
"""La moitie de chaque specification : le sommet du plus grand rectangle."""


def induction_sur_la_droite_de_desaimantation(
    champ: float,
    remanente: float,
    coercitif: float,
) -> float:
    """`B = B_r (1 + H / H_c)` pour `H` dans `[-H_c, 0]`.

    Le modele le plus simple qui respecte les deux specifications. Le
    champ y est **negatif**, comme le syllabus le precise entre
    parentheses : la courbe de desaimantation est le deuxieme quadrant du
    cycle.
    """
    if champ > 0:
        msg = f"champ positif : {champ} A/m -- la desaimantation est en H < 0"
        raise ValueError(msg)
    if coercitif <= 0:
        msg = f"champ coercitif nul ou negatif : {coercitif} A/m"
        raise ValueError(msg)
    if champ < -coercitif:
        msg = (
            f"au-dela du champ coercitif : {champ} A/m pour {coercitif} -- "
            f"l'aimant y est desaimante"
        )
        raise ValueError(msg)
    return remanente * (1 + champ / coercitif)


def produit_d_energie(induction: float, champ: float) -> float:
    """`-H B`, en joules par metre cube. Positif, le champ etant negatif."""
    return -champ * induction


def point_de_fonctionnement_optimal(
    remanente: float,
    coercitif: float,
) -> tuple[float, float]:
    """`(-H_c/2, B_r/2)` : le sommet du rectangle inscrit d'aire maximale."""
    if coercitif <= 0 or remanente <= 0:
        msg = f"specifications nulles ou negatives : {remanente} T, {coercitif} A/m"
        raise ValueError(msg)
    return -FRACTION_OPTIMALE * coercitif, FRACTION_OPTIMALE * remanente


def produit_maximal(remanente: float, coercitif: float) -> float:
    """`(HB)max = B_r H_c / 4`, en J/m^3.

    Il croit comme le **produit** des deux specifications : doubler les
    deux le quadruple, et c'est ce qui separe une ferrite d'un aimant aux
    terres rares.
    """
    champ, induction = point_de_fonctionnement_optimal(remanente, coercitif)
    return produit_d_energie(induction, champ)


def volume_minimal_de_l_aimant(
    induction_entrefer: float,
    volume_entrefer: float,
    produit: float,
) -> float:
    """Le volume d'aimant qu'un entrefer donne reclame, en metres cubes.

    Inversement proportionnel au produit d'energie du materiau : un
    aimant deux fois meilleur est deux fois plus petit, et c'est l'enjeu
    industriel du chapitre 12.3.
    """
    if produit <= 0:
        msg = f"produit d'energie nul ou negatif : {produit} J/m^3 -- pas un aimant"
        raise ValueError(msg)
    if volume_entrefer <= 0:
        msg = f"volume d'entrefer nul ou negatif : {volume_entrefer} m^3"
        raise ValueError(msg)
    return induction_entrefer**2 * volume_entrefer / produit
