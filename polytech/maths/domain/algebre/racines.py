"""Les racines d'un polynome -- chapitre IV.2 du syllabus d'algebre.

Trois methodes, que le syllabus donne en clair et que rien ne portait.

**Le second degre.** `Delta = b^2 - 4ac`, racines `(-b +- racine)/(2a)`,
et « si on travaille dans R, il y a des racines si et seulement si
Delta >= 0 ». L'absence de racine reelle est donc rendue comme un tuple
vide, non comme une erreur : c'est un resultat, et c'est meme celui qui
motive le chapitre I du meme syllabus -- `x^2 + 1` n'a pas de racine
reelle, d'ou la construction des complexes.

**Les bicarrees.** « Poser `x^2 = y`, resoudre `ay^2 + by + c = 0`,
chaque `y` trouve donne deux valeurs de `x` (sauf `y = 0` : `x = 0`). Si
on travaille dans R, seuls les `y` positifs sont a retenir. » Les deux
parentheses portent le sens : un `y` nul ne rend qu'une racine, un `y`
negatif n'en rend aucune. Un bicarre peut donc avoir quatre, trois, deux,
une ou zero racines reelles.

**Les polynomes a coefficients entiers.** « S'il existe une racine
rationnelle `p/q`, alors `p` divise `a_0` et `q` divise `a_n` ». Ce n'est
pas une formule de resolution mais une **borne** : elle donne la liste
finie ou chercher, et c'est tout son interet. L'evaluation tranche
ensuite.
"""

from __future__ import annotations

import math


def discriminant(a: float, b: float, c: float) -> float:
    """`Delta = b^2 - 4ac`."""
    if a == 0:
        msg = "coefficient dominant nul : ce n'est pas un second degre"
        raise ValueError(msg)
    return b * b - 4 * a * c


def racines_reelles_du_second_degre(
    a: float,
    b: float,
    c: float,
) -> tuple[float, ...]:
    """Les racines reelles, sans repetition de la racine double.

    Rendre deux fois une racine double ferait croire a deux racines
    distinctes, ce qui change la forme de la solution d'une equation
    differentielle -- la ou ce resultat sert ensuite.
    """
    delta = discriminant(a, b, c)
    if delta < 0:
        return ()
    if delta == 0:
        return (-b / (2 * a),)
    racine = math.sqrt(delta)
    return ((-b - racine) / (2 * a), (-b + racine) / (2 * a))


def racines_du_bicarre(a: float, b: float, c: float) -> tuple[float, ...]:
    """Les racines reelles de `a x^4 + b x^2 + c`, par la substitution.

    Un `y` nul ne rend qu'une racine, un `y` negatif n'en rend aucune :
    un bicarre a donc quatre, trois, deux, une ou zero racines reelles.
    """
    racines: list[float] = []
    for y in racines_reelles_du_second_degre(a, b, c):
        if y == 0:
            racines.append(0.0)
        elif y > 0:
            racine = math.sqrt(y)
            racines.extend((-racine, racine))
    return tuple(racines)


def _diviseurs(entier: int) -> tuple[int, ...]:
    """Les diviseurs d'un entier, des deux signes.

    Le syllabus le precise entre parentheses : « lorsqu'on parle de
    diviseurs », il les prend positifs et negatifs.
    """
    valeur = abs(int(entier))
    if valeur == 0:
        msg = "terme constant nul : zero est racine, et tout entier le divise"
        raise ValueError(msg)
    positifs = [d for d in range(1, valeur + 1) if valeur % d == 0]
    return tuple(sorted([*positifs, *(-d for d in positifs)]))


def candidats_entiers(coefficients: tuple[int, ...]) -> tuple[int, ...]:
    """Les diviseurs du terme constant : ou chercher une racine entiere."""
    return _diviseurs(coefficients[-1])


def candidats_rationnels(coefficients: tuple[int, ...]) -> tuple[float, ...]:
    """Les `p/q` ou `p` divise `a_0` et `q` divise `a_n`.

    Une **borne** sur la recherche, et non une resolution : la liste est
    finie, et l'evaluation tranche ensuite.
    """
    return tuple(
        sorted(
            {
                p / q
                for p in _diviseurs(coefficients[-1])
                for q in _diviseurs(coefficients[0])
            }
        ),
    )


def est_racine(coefficients: tuple[int, ...], valeur: float) -> bool:
    """Vrai si cette valeur annule le polynome, par evaluation de Horner."""
    accumule = 0.0
    for coefficient in coefficients:
        accumule = accumule * valeur + coefficient
    return math.isclose(accumule, 0.0, abs_tol=1e-9)
