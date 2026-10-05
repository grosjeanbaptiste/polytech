"""La liste simplement chainee.

Le syllabus (chapitre 9) : « une liste chainee est une sequence d'objets
dynamiques dont chacun contient une indication explicite vers l'element
suivant ». Chaque cellule porte son contenu et un pointeur vers la cellule
suivante ; la derniere pointe sur `NULL`. On ne connait de la liste que
l'adresse de sa premiere cellule, `debut`.

Ce que la liste perd sur le tableau, c'est l'acces direct : on ne la
parcourt qu'en partant du debut. Ce qu'elle gagne, c'est d'inserer et de
supprimer sans rien decaler : il suffit de refaire un chainage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Cellule:
    """Un element de la liste : son contenu, et la cellule qui le suit."""

    contenu: Any
    suivante: Cellule | None = None


class ListeChainee:
    """Une liste connue par sa premiere cellule."""

    def __init__(self) -> None:
        """`debut = NULL;` : la liste vide."""
        self.debut: Cellule | None = None

    def _tete(self) -> Cellule:
        if self.debut is None:
            msg = "liste vide : aucune cellule de debut"
            raise LookupError(msg)
        return self.debut

    def inserer_en_tete(self, contenu: Any) -> None:  # noqa: ANN401 -- tout type
        """La nouvelle cellule pointe sur l'ancien debut, et devient le debut."""
        self.debut = Cellule(contenu, self.debut)

    def inserer_apres_la_tete(self, contenu: Any) -> None:  # noqa: ANN401 -- tout type
        """`p->next = debut->next; debut->next = p;`."""
        tete = self._tete()
        tete.suivante = Cellule(contenu, tete.suivante)

    def supprimer_la_tete(self) -> Any:  # noqa: ANN401 -- le contenu retire
        """`q = debut; debut = debut->next; delete q;` : rend le contenu retire."""
        tete = self._tete()
        self.debut = tete.suivante
        return tete.contenu

    def parcours(self) -> tuple[Any, ...]:
        """Les contenus, du debut jusqu'a la cellule qui pointe sur `NULL`."""
        contenus, cellule = [], self.debut
        while cellule is not None:
            contenus.append(cellule.contenu)
            cellule = cellule.suivante
        return tuple(contenus)

    def liberer(self) -> int:
        """Detruit toutes les cellules ; rend leur nombre, un `delete` par `new`."""
        liberees = 0
        while self.debut is not None:
            self.supprimer_la_tete()
            liberees += 1
        return liberees
