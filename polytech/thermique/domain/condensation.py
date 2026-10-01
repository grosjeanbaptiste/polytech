"""La condensation d'une vapeur pure sur une paroi froide.

Chapitre 5 du tome 2 du syllabus, absent de la bibliotheque jusqu'ici
parce que la mesure de couverture ne regardait que les planches.

Le chapitre decompose la transition de la phase gazeuse au liquide en
contact avec la paroi en **trois actions successives**, et ce decoupage
est ce qui porte le raisonnement : il faut les enumerer pour comparer
« les contributions respectives de ces trois phenomenes a la resistance
de chaleur ». La conclusion du chapitre est que, pour une vapeur
**pure**, seule la troisieme compte.

L'argument est chiffre, et c'est ce qui le rend verifiable sans voir les
formules que la conversion n'a pas rendues. La vapeur doit acquerir une
vitesse normale a la paroi de l'ordre de `0,1 m/s` ; cette vitesse
« correspond a une energie cinetique tres faible, et en appliquant le
theoreme de Bernoulli, a une chute de pression (donc de temperature de
saturation) negligeable ». Trois millipascals contre un bar : sept ordres
de grandeur.

Le syllabus pose aussitot la limite de son propre raisonnement : « ce
raisonnement n'est correct que si la vapeur est pure ». Avec des
incondensables -- de l'air -- ceux-ci s'accumulent contre la paroi, la
vapeur doit l'atteindre par diffusion moleculaire, et le potentiel
thermique chute pour de bon. D'ou les accessoires d'elimination dont tout
condenseur est pourvu.
"""

from __future__ import annotations

from typing import Final

ACTIONS_DE_LA_CONDENSATION: Final = (
    "deplacement de la vapeur en phase gazeuse jusqu'a la surface du condensat",
    "condensation a l'interface liquide-vapeur",
    "transmission de la chaleur latente a travers le film de condensat",
)
"""Les trois actions elementaires, dans l'ordre du syllabus.

L'ordre n'est pas decoratif : la resistance totale est leur somme, et le
chapitre montre que la troisieme l'emporte des que la vapeur est pure.
"""


def vitesse_normale_de_la_vapeur(
    flux: float,
    masse_volumique: float,
    chaleur_latente: float,
) -> float:
    """`u_v = phi / (rho_v l_v)`, en m/s.

    Le bilan de masse la donne : le flux `phi` condense `phi / l_v`
    kilogrammes par seconde et par metre carre, que la vapeur doit
    amener a cette vitesse. La chaleur latente au denominateur est ce qui
    rend la condensation si efficace -- beaucoup d'energie par unite de
    masse, donc peu de vitesse a fournir.
    """
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)
    if chaleur_latente <= 0:
        msg = f"chaleur latente nulle ou negative : {chaleur_latente} J/kg"
        raise ValueError(msg)
    return flux / (masse_volumique * chaleur_latente)


def chute_de_pression_dynamique(masse_volumique: float, vitesse: float) -> float:
    """`rho u^2 / 2`, en pascals : la pression dynamique de la vapeur.

    C'est le terme que le syllabus compare a la pression du condenseur
    pour conclure que la chute de temperature de saturation est
    negligeable. L'argument est ecrasant sur une vapeur pure, et c'est ce
    qui autorise a ne retenir que la resistance du film.
    """
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)
    return masse_volumique * vitesse**2 / 2
