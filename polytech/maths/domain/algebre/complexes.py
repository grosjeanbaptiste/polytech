"""Les nombres complexes -- chapitre I du syllabus d'algebre.

Un chapitre entier que rien ne portait. Le syllabus dit lui-meme pourquoi
il ouvre le cours : « malgre le role essentiel que joue l'ensemble des
reels, celui-ci presente certains inconvenients. Par exemple, le polynome
`x^2 + 1` n'a pas de racine reelle. Dans C, le polynome `x^2 + 1` possede
deux racines. »

La motivation est donc exactement le tuple vide que
`racines.racines_reelles_du_second_degre` rend sur ce polynome -- et
`racines_du_second_degre_dans_c` est la reponse que le chapitre apporte :
deux racines toujours, conjuguees quand le discriminant est negatif.

Deux refus plutot que deux conventions.

**L'argument de zero** n'existe pas : toutes les directions le
representent. Rendre zero par convention ferait croire a une direction
choisie, et c'est le genre de silence que ce depot refuse. Son module,
lui, vaut zero sans ambiguite.

**L'ordre nul** d'une racine n'a pas de sens : l'equation `w^0 = z` n'a
pas de solution pour `z` different de un.

Les racines n-iemes sont les sommets d'un polygone regulier centre sur
l'origine. Leur somme est donc nulle des que `n >= 2`, et c'est une
propriete bien plus forte qu'un controle de module : elle contraint les
phases toutes ensemble, si bien qu'un ecart angulaire errone -- `pi/n` au
lieu de `2 pi/n`, l'erreur classique -- la romprait.
"""

from __future__ import annotations

import cmath
import math


def conjugue(z: complex) -> complex:
    """`z_barre` : la partie imaginaire change de signe.

    Involution, et `z z_barre = |z|^2` : c'est l'identite qui le definit,
    et celle par laquelle les tests le verifient.
    """
    return z.conjugate()


def module(z: complex) -> float:
    """`|z| = racine(x^2 + y^2)`. Nul, et defini, en zero."""
    return abs(z)


def argument(z: complex) -> float:
    """L'angle polaire, en radians, dans `]-pi, pi]`.

    Refuse en zero : toutes les directions y conviennent, et en choisir
    une par convention serait inventer une information.
    """
    if z == 0:
        msg = "argument de zero : toutes les directions le representent"
        raise ValueError(msg)
    return math.atan2(z.imag, z.real)


def forme_exponentielle(z: complex) -> tuple[float, float]:
    """`(r, theta)` tels que `z = r e^{i theta}`.

    La notation du syllabus, et son interet : elle rend le produit et la
    puissance triviaux. L'aller-retour doit etre fidele, ce qu'un test
    exige.
    """
    return module(z), argument(z)


def racines_niemes(z: complex, ordre: int) -> tuple[complex, ...]:
    """Les `n` racines n-iemes de `z`, dans l'ordre des arguments croissants.

    Meme module `|z|^(1/n)` pour toutes, et arguments espaces de
    `2 pi / n` : les sommets d'un polygone regulier.
    """
    if ordre <= 0:
        msg = f"ordre de racine nul ou negatif : {ordre}"
        raise ValueError(msg)
    if z == 0:
        return (0j,) * ordre
    rayon, angle = forme_exponentielle(z)
    pas = 2 * math.pi / ordre
    return tuple(
        rayon ** (1 / ordre) * cmath.exp(1j * (angle / ordre + k * pas))
        for k in range(ordre)
    )


def racines_du_second_degre_dans_c(
    a: complex,
    b: complex,
    c: complex,
) -> tuple[complex, complex]:
    """Les deux racines de `a z^2 + b z + c` dans C, toujours deux.

    C'est la reponse du chapitre a l'inconvenient qu'il pointe dans les
    reels : il y a deux racines meme quand le discriminant est negatif,
    et elles sont alors conjuguees.
    """
    if a == 0:
        msg = "coefficient dominant nul : ce n'est pas un second degre"
        raise ValueError(msg)
    delta = b * b - 4 * a * c
    racine = cmath.sqrt(delta)
    return (-b - racine) / (2 * a), (-b + racine) / (2 * a)
