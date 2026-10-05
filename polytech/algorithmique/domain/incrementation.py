"""Les operateurs d'incrementation et de decrementation, prefixes ou postfixes.

Le syllabus (2.9.4, exemple 7) : `++` ajoute 1 a son operande, `--` lui
retranche 1. Places **devant** la variable, ils la mettent a jour avant son
emploi dans l'expression ; places **derriere**, apres.

* `X = I++` passe d'abord la valeur de `I` a `X`, et incremente `I` apres ;
* `X = ++I` incremente d'abord `I`, et passe la valeur incrementee a `X`.

Chaque fonction rend donc deux choses que le programmeur confond : ce que
**vaut l'expression**, et ce que **devient la variable**.
"""

from __future__ import annotations


def prefixe(valeur: int, pas: int = 1) -> tuple[int, int]:
    """`++i` (ou `--i` avec un pas de -1) : l'expression vaut la valeur mise a jour."""
    return valeur + pas, valeur + pas


def postfixe(valeur: int, pas: int = 1) -> tuple[int, int]:
    """`i++` (ou `i--`) : l'expression vaut l'ancienne valeur, la variable change."""
    return valeur, valeur + pas
