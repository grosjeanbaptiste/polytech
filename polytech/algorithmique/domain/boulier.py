"""Le boulier a quatre : un tableau borne dont les ajouts chassent le bout oppose.

Le syllabus (exercice 8.7) : « un systeme de billes suspendues par des
fils », qui « accepte au maximum quatre billes ». Au-dela de quatre, « si
une bille est ajoutee a droite, la bille de gauche est retiree. De meme,
si une bille est ajoutee a gauche, celle de droite est retiree ».

Le programme demande aussi d'effacer une bille choisie, et de signaler
celle qui n'existe pas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

CAPACITE: Final = 4
"""Le nombre de billes que le boulier accepte."""


class Boulier:
    """Au plus quatre billes, de la plus a gauche a la plus a droite."""

    def __init__(self, billes: Iterable[Any] = ()) -> None:
        """La saisie initiale : `NI <= 4` billes."""
        self._billes = list(billes)
        if len(self._billes) > CAPACITE:
            msg = f"boulier trop charge : {len(self._billes)} billes pour {CAPACITE}"
            raise OverflowError(msg)

    @property
    def billes(self) -> tuple[Any, ...]:
        """Les billes, de gauche a droite."""
        return tuple(self._billes)

    def _chassee(self, rang: int) -> Any:  # noqa: ANN401 -- une bille, ou rien
        return self._billes.pop(rang) if len(self._billes) > CAPACITE else None

    def ajouter_a_gauche(self, bille: Any) -> Any:  # noqa: ANN401 -- la bille chassee
        """Ajoute a gauche ; rend la bille de droite si elle est chassee."""
        self._billes.insert(0, bille)
        return self._chassee(-1)

    def ajouter_a_droite(self, bille: Any) -> Any:  # noqa: ANN401 -- la bille chassee
        """Ajoute a droite ; rend la bille de gauche si elle est chassee."""
        self._billes.append(bille)
        return self._chassee(0)

    def effacer(self, bille: Any) -> None:  # noqa: ANN401 -- tout type
        """Retire la bille choisie ; leve si elle n'est pas sur le boulier."""
        if bille not in self._billes:
            msg = f"bille absente du boulier : {bille}"
            raise LookupError(msg)
        self._billes.remove(bille)
