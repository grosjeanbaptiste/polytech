"""L'evaluation d'une expression selon les priorites du C++.

Le syllabus (2.9, « Operateurs et expressions ») : dans une meme classe de
priorite, les operateurs binaires s'evaluent de gauche a droite ; les
unaires et l'operateur ternaire `? :`, de droite a gauche. Les classes
elles-memes, et ce que chaque operateur calcule, vivent dans `priorites`.

L'analyse construit des calculs differes et ne les execute qu'a la demande :
c'est ce qui laisse `&&`, `||` et le ternaire ne pas evaluer l'operande
dont ils n'ont pas besoin.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final, NoReturn

from polytech.algorithmique.domain.priorites import (
    PRIORITES,
    UNAIRES,
    calcul_binaire,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polytech.algorithmique.domain.priorites import Calcul

_JETONS: Final = re.compile(
    r"\s*(\d+\.\d*|\d+|[A-Za-z_]\w*|\|\||&&|[<>=!]=|[-+*/%()<>!?:])|\s*(\S)"
)
_OPERANDE: Final = re.compile(r"\w")


def _refuser(raison: str) -> NoReturn:
    msg = f"expression mal formee : {raison}"
    raise SyntaxError(msg)


class _Analyse:
    def __init__(self, texte: str, variables: Mapping[str, float]) -> None:
        trouves = _JETONS.findall(texte)
        inconnus = [inconnu for _, inconnu in trouves if inconnu]
        if inconnus:
            _refuser(f"caractere inattendu {inconnus[0]}")
        self.jetons: list[str] = [jeton for jeton, _ in trouves]
        self._variables = variables

    def _lu(self, *attendus: str) -> str:
        if self.jetons and self.jetons[0] in attendus:
            return self.jetons.pop(0)
        return ""

    def expression(self) -> Calcul:
        condition = self._classe(0)
        if not self._lu("?"):
            return condition
        oui = self.expression()
        if not self._lu(":"):
            _refuser("il manque les deux-points du ternaire")
        non = self.expression()
        return lambda: oui() if condition() else non()

    def _classe(self, rang: int) -> Calcul:
        if rang == len(PRIORITES):
            return self._unaire()
        gauche = self._classe(rang + 1)
        while operateur := self._lu(*PRIORITES[rang]):
            gauche = calcul_binaire(operateur, gauche, self._classe(rang + 1))
        return gauche

    def _unaire(self) -> Calcul:
        if operateur := self._lu(*UNAIRES):
            operande = self._unaire()
            return lambda: UNAIRES[operateur](operande())
        if self._lu("("):
            interieur = self.expression()
            if not self._lu(")"):
                _refuser("parenthese non fermee")
            return interieur
        return self._operande()

    def _operande(self) -> Calcul:
        if not self.jetons or not _OPERANDE.match(self.jetons[0]):
            _refuser("un operande est attendu")
        jeton = self.jetons.pop(0)
        if jeton[0].isdigit():
            valeur = float(jeton) if "." in jeton else int(jeton)
            return lambda: valeur
        return lambda: self._valeur(jeton)

    def _valeur(self, nom: str) -> float:
        if nom not in self._variables:
            msg = f"variable non declaree : {nom}"
            raise NameError(msg)
        return self._variables[nom]


def evaluer(expression: str, variables: Mapping[str, float] | None = None) -> Any:  # noqa: ANN401 -- entier, reel ou booleen
    """La valeur de l'expression, comme un programme C++ la calculerait."""
    analyse = _Analyse(expression, variables or {})
    calcul = analyse.expression()
    if analyse.jetons:
        _refuser(f"jeton inattendu {analyse.jetons[0]}")
    return calcul()
