"""Ce que `cout` affiche : la mise en forme d'un flux de sortie.

Le cours (partie 1, planches 15 a 19) : `cout << expression` ecrit a
l'ecran, et `<iomanip>` en regle la presentation.

* un reel s'affiche avec six chiffres significatifs, sans zeros inutiles,
  et passe en notation scientifique quand son exposant sort de la plage ;
  `setprecision(n)` change ce nombre de chiffres ;
* `setw(n)` impose une largeur **minimale** a la sortie suivante, et a
  elle seule ;
* `setfill(c)` remplace les espaces de remplissage par `c`.

`setprecision` et `setfill` restent en vigueur jusqu'a nouvel ordre ;
`setw` s'oublie des la sortie suivante. C'est la source d'erreur la plus
courante dans un affichage en colonnes.
"""

from __future__ import annotations

from typing import Final

ECHAPPEMENTS: Final = {
    "\\n": 0x0A,
    "\\t": 0x09,
    "\\v": 0x0B,
    "\\b": 0x08,
    "\\r": 0x0D,
    "\\f": 0x0C,
}
"""Les caracteres d'echappement de la planche 17, et leur code ASCII."""

PRECISION_PAR_DEFAUT: Final = 6


def ecrire_un_reel(valeur: float, precision: int = PRECISION_PAR_DEFAUT) -> str:
    """Le reel tel que `cout` l'ecrit, avec ce nombre de chiffres significatifs."""
    if precision <= 0:
        msg = f"precision nulle ou negative : {precision}"
        raise ValueError(msg)
    return format(valeur, f".{precision}g")


class Ecran:
    """Un flux de sortie : il retient sa precision, son remplissage, et une largeur."""

    def __init__(self) -> None:
        """Un flux vide, regle comme `cout` au debut d'un programme."""
        self.texte = ""
        self._precision = PRECISION_PAR_DEFAUT
        self._remplissage = " "
        self._largeur = 0

    def setprecision(self, chiffres: int) -> Ecran:
        """Le nombre de chiffres significatifs, jusqu'a nouvel ordre."""
        self._precision = chiffres
        return self

    def setfill(self, caractere: str) -> Ecran:
        """Le caractere de remplissage, jusqu'a nouvel ordre."""
        self._remplissage = caractere
        return self

    def setw(self, largeur: int) -> Ecran:
        """La largeur minimale de la sortie suivante, et d'elle seule."""
        self._largeur = largeur
        return self

    def ecrire(self, valeur: object) -> Ecran:
        """`cout << valeur`."""
        if isinstance(valeur, bool):
            brut = str(int(valeur))
        elif isinstance(valeur, float):
            brut = ecrire_un_reel(valeur, self._precision)
        else:
            brut = str(valeur)
        self.texte += brut.rjust(self._largeur, self._remplissage)
        self._largeur = 0
        return self

    def endl(self) -> Ecran:
        """Un saut de ligne."""
        self.texte += "\n"
        return self

    def lignes(self) -> list[str]:
        """Le texte affiche, ligne par ligne ; un saut final n'ouvre pas de ligne."""
        return self.texte.removesuffix("\n").split("\n")
