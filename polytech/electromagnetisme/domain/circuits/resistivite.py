"""La resistivite, et sa variation avec la temperature.

Le syllabus d'electricite generale (13.10). `rho = R S / l` (13.10.1), en
ohm-metre, est avec la permittivite et la permeabilite la troisieme
grandeur qui caracterise la matiere. Elle depend de la temperature :
`rho(t_2) = rho(t_1) [1 + alpha (t_2 - t_1) + beta (t_2 - t_1)^2]`
(13.10.2), lineaire sur quelques dizaines de degres (13.10.3). Le
coefficient `alpha` est positif pour les metaux purs, negatif pour les
electrolytes et le carbone : une resistance metallique est plus grande a
chaud qu'a froid.

Pour une couche mince, la resistivite superficielle `rho_S` est le
quotient de `rho` par l'epaisseur : en ohms, c'est la resistance d'un
carre. Un ruban de longueur `l` et de largeur `b` a la resistance
`rho_S l / b` -- le syllabus imprime `dS / db` en (13.10.5), ce que les
unites refusent.
"""

from __future__ import annotations

from typing import Final

METAUX: Final = {
    "cuivre": (17e-9, 0.004),
    "aluminium": (27e-9, 0.004),
    "fer": (100e-9, 0.006),
    "manganine": (470e-9, 0.00001),
    "graphite": (10e-6, -0.001),
}
"""Tableau 13.10.1 : `(rho, alpha)` a 20 degres, en ohm.m et par degre.

La manganine est un alliage a 84 % de cuivre, 12 % de manganese et 4 % de
nickel.
"""

RESISTIVITE_DU_VERRE: Final = 1e12
"""En ohm.m, a la temperature ambiante : un bon isolant (page 13-16)."""


def resistivite_a_chaud(
    reference: float, ecart: float, alpha: float, *, beta: float = 0.0
) -> float:
    """`rho(t_1) [1 + alpha dt + beta dt^2]` (13.10.2), en ohm.m."""
    return reference * (1 + alpha * ecart + beta * ecart**2)


def resistivite_superficielle(resistivite: float, epaisseur: float) -> float:
    """`rho_S = rho / e`, en ohms : la resistance d'un carre de la couche."""
    if epaisseur <= 0:
        msg = f"epaisseur nulle ou negative : {epaisseur} m"
        raise ValueError(msg)
    return resistivite / epaisseur


def resistance_d_un_ruban(
    superficielle: float, longueur: float, largeur: float
) -> float:
    """`R = rho_S l / b` (13.10.5), en ohms."""
    if largeur <= 0:
        msg = f"largeur nulle ou negative : {largeur} m"
        raise ValueError(msg)
    return superficielle * longueur / largeur


def resistivite_de_matthiessen(*, impuretes: float, phonons: float) -> float:
    """`rho = rho_i + rho_ph` : les defauts du reseau, et ses vibrations.

    Regle de Matthiessen (electronique, chapitre 1, partie I, planche 28).
    La premiere part ne depend pas de la temperature ; la seconde croit
    avec elle.
    """
    if impuretes < 0 or phonons < 0:
        msg = f"resistivite negative : {impuretes} ou {phonons} ohm.m"
        raise ValueError(msg)
    return impuretes + phonons
