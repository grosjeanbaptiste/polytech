"""La memoire d'un programme : des cases, leurs adresses, et les noms qui les designent.

Le cours (partie 1, planches 83 a 85 et 110 a 126) :

* une variable occupe une case ; `&i` est son adresse, et deux `int`
  declares a la suite sont a quatre octets l'un de l'autre ;
* un **pointeur** contient une adresse, et `*p` est ce qui s'y trouve ;
* une **reference** est un autre nom de la meme case : elle doit etre
  initialisee, et la modifier modifie la variable.

D'ou les trois passages de parametres. Par **valeur**, la fonction recoit
une copie, et la variable de l'appelant ne change pas. Par **pointeur** ou
par **reference**, elle recoit la case elle-meme. Un tableau est toujours
passe par son adresse.

Une variable declaree sans valeur contient ce que la memoire contenait :
la lire n'a pas de sens, et ce module le refuse.
"""

from __future__ import annotations

from typing import Any, Final

TAILLE_D_UN_INT: Final = 4
_VIDE: Final = object()


class Memoire:
    """Des cases de quatre octets, rangees a la suite depuis une adresse de depart."""

    def __init__(self, depart: int = 0x1000) -> None:
        """Une memoire vide, dont la premiere variable ira a `depart`."""
        self._cases: dict[int, Any] = {}
        self._adresses: dict[str, int] = {}
        self._libre = depart

    def declarer(self, nom: str, valeur: Any = _VIDE) -> int:  # noqa: ANN401 -- tout type
        """Reserve une case pour `nom` ; sans valeur, elle n'est pas initialisee."""
        self._adresses[nom] = self._libre
        self._cases[self._libre] = valeur
        self._libre += TAILLE_D_UN_INT
        return self._adresses[nom]

    def declarer_un_tableau(self, nom: str, valeurs: list[Any]) -> int:
        """Des cases contigues ; le nom du tableau designe la premiere."""
        base = self.declarer(nom, valeurs[0])
        for valeur in valeurs[1:]:
            self._cases[self._libre] = valeur
            self._libre += TAILLE_D_UN_INT
        return base

    def referencer(self, alias: str, nom: str) -> None:
        """`type &alias = nom;` : un second nom pour la meme case."""
        if nom not in self._adresses:
            msg = f"une reference doit designer une variable existante : {nom}"
            raise KeyError(msg)
        self._adresses[alias] = self._adresses[nom]

    def adresse_de(self, nom: str) -> int:
        """`&nom`."""
        if nom not in self._adresses:
            msg = f"variable non declaree : {nom}"
            raise KeyError(msg)
        return self._adresses[nom]

    def contenu(self, adresse: int) -> Any:  # noqa: ANN401 -- ce que la case contient
        """`*adresse`."""
        if adresse not in self._cases:
            msg = f"adresse hors des variables declarees : {adresse:#x}"
            raise KeyError(msg)
        if self._cases[adresse] is _VIDE:
            msg = "variable non initialisee : sa valeur est celle que la memoire avait"
            raise ValueError(msg)
        return self._cases[adresse]

    def ecrire(self, adresse: int, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """`*adresse = valeur;`."""
        if adresse not in self._cases:
            msg = f"adresse hors des variables declarees : {adresse:#x}"
            raise KeyError(msg)
        self._cases[adresse] = valeur

    def valeur(self, nom: str) -> Any:  # noqa: ANN401 -- ce que la variable contient
        """La valeur de la variable, quel que soit le nom qui la designe."""
        return self.contenu(self.adresse_de(nom))

    def affecter(self, nom: str, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """`nom = valeur;`."""
        self.ecrire(self.adresse_de(nom), valeur)

    def tableau(self, nom: str, taille: int) -> list[Any]:
        """Les `taille` premieres cases du tableau."""
        base = self.adresse_de(nom)
        return [self.contenu(base + TAILLE_D_UN_INT * rang) for rang in range(taille)]
