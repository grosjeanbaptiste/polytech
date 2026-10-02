"""Dix spires emboitees, de largeurs doublees -- exercice 8.

L'enonce du recueil : dix spires rectangulaires infiniment longues, de
largeurs 2, 4, 8, ... 1024 cm, dans le plan `Oxy` avec `Oy` pour axe de
symetrie, toutes parcourues dans le sens anti-horlogique par 3 A. Le
champ au centre est la somme de vingt contributions, deux par spire.

**La distance n'est pas la largeur.** L'axe partage chaque spire : ses
deux longs cotes sont a la **demi-largeur**, soit 1, 2, 4, ... 512 cm.
Prendre la largeur pour la distance divise le champ par deux, exactement
-- doubler toutes les distances d'une progression de raison deux revient
a decaler l'indice d'un cran. `demi_largeurs` rend ce pas explicite.

**Les deux cotes s'ajoutent, et il faut dire pourquoi.** Dans une spire
parcourue en sens anti-horlogique, le courant monte d'un cote et descend
de l'autre : les courants sont opposes **et** les positions symetriques.
Les deux inversions de signe se detruisent, et les contributions
s'additionnent. `champ_d_un_seul_cote` rend le facteur deux verifiable.

**Tout est en rationnels exacts.** Les distances sont des puissances de
deux sur cent, et la somme geometrique a une forme fermee : aucune etape
n'a besoin de flottants.

Ce module a porte une fonction `champ_par_le_developpement_du_manuscrit`,
qui reproduisait une « erreur » imputee au manuscrit. Le manuscrit n'en
commet pas : c'est la transcription de l'agent qui avait perdu son
facteur un demi. La fonction est retiree avec la these qu'elle servait.
"""

from __future__ import annotations

import sympy as sp

CENTIMETRE = sp.Rational(1, 100)
"""Un centimetre en metres, exact -- les distances de l'enonce y sont."""

RAISON = 2
"""La raison de la progression : chaque conducteur est deux fois plus loin."""


def distances_geometriques(
    premier_exposant: int,
    nombre: int,
) -> tuple[sp.Expr, ...]:
    """Les `nombre` distances `2^k x 1 cm`, en metres et en exact.

    `premier_exposant` vaut zero pour les distances a l'axe -- le
    premier conducteur est a un centimetre -- et un pour les largeurs de
    l'enonce, qui commencent a deux.
    """
    return tuple(
        sp.Integer(RAISON) ** exposant * CENTIMETRE
        for exposant in range(premier_exposant, premier_exposant + nombre)
    )


def largeurs_des_spires(nombre: int) -> tuple[sp.Expr, ...]:
    """Les largeurs de l'enonce : 2, 4, 8... cm, en metres et en exact."""
    return distances_geometriques(1, nombre)


def demi_largeurs(largeurs: tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    """La distance de chaque long cote a l'axe : la moitie de la largeur.

    C'est le facteur un demi que le manuscrit ecrit sous son trait de
    fraction, annote « distance entre le conducteur et l'axe y ».
    """
    return tuple(largeur / 2 for largeur in largeurs)


def somme_des_inverses(distances: tuple[sp.Expr, ...]) -> sp.Expr:
    """La somme des `1/d`, terme a terme -- le calcul explicite."""
    return sum((1 / distance for distance in distances), start=sp.Integer(0))


def somme_fermee(premier_exposant: int, nombre: int) -> sp.Expr:
    """`100 . 2^(1 - premier) . (1 - 2^-nombre)` : la serie geometrique.

    Forme fermee, independante du calcul explicite : les deux doivent
    coincider exactement.
    """
    premier_terme = 1 / (sp.Integer(RAISON) ** premier_exposant * CENTIMETRE)
    return premier_terme * RAISON * (1 - sp.Rational(1, RAISON**nombre))


def champ_d_un_seul_cote(
    courant: sp.Expr,
    distances: tuple[sp.Expr, ...],
) -> sp.Expr:
    """`sum I / (2 pi d)` : la contribution des conducteurs d'un seul cote.

    Rendue a part pour que le facteur deux de la symetrie se verifie, au
    lieu d'etre pose d'emblee.
    """
    return courant / (2 * sp.pi) * somme_des_inverses(distances)


def champ_de_la_nappe(
    courant: sp.Expr,
    distances: tuple[sp.Expr, ...],
) -> sp.Expr:
    """`sum 2 I / (2 pi d) = (I/pi) sum 1/d` : les deux cotes.

    C'est la formule du manuscrit, les distances etant les
    demi-largeurs.
    """
    return 2 * champ_d_un_seul_cote(courant, distances)


def champ_de_la_nappe_infinie(
    courant: sp.Expr,
    premier_exposant: int,
) -> sp.Expr:
    """Le champ d'une nappe prolongee a l'infini.

    La serie converge, et dix paires en donnent deja plus de 99,9 pour
    cent : une onzieme paire changerait le resultat de moins d'un
    millieme. C'est pourquoi l'enonce peut s'arreter a dix.
    """
    premier_terme = 1 / (sp.Integer(RAISON) ** premier_exposant * CENTIMETRE)
    return courant / sp.pi * premier_terme * RAISON
