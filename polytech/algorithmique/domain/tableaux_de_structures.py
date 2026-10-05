"""Les gestes des examens types sur un tableau de structures.

Le cours (seance 3, planches des examens types) : trois enonces de deux
heures, batis sur le meme canevas -- saisir un tableau de structures, y
ajouter a une position, en afficher une partie, en supprimer selon un
critere. Le tableau lui-meme, son insertion et sa capacite sont ceux de
`tableaux.TableauStatique` ; ce module porte les gestes que les trois
enonces ajoutent :

* trouver l'element le plus grand et le plus petit selon un champ -- et,
  quand la question le demande, **tous** les ex aequo du plus grand ;
* n'afficher qu'un element sur deux ;
* supprimer tous ceux qui satisfont une condition ;
* trouver ou inserer « apres qui », en refusant un nom absent ;
* refuser une valeur saisie hors de l'intervalle impose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def un_sur_deux(elements: Sequence[Any]) -> tuple[Any, ...]:
    """Le premier element, le troisieme, le cinquieme, ..."""
    return tuple(elements[::2])


def _exiger_non_vide(elements: Sequence[Any]) -> None:
    if not elements:
        msg = "tableau vide : aucun element a comparer"
        raise ValueError(msg)


def extremes(elements: Sequence[Any], cle: Callable[[Any], Any]) -> tuple[Any, Any]:
    """Le plus grand et le plus petit selon la cle ; le premier en cas d'egalite."""
    _exiger_non_vide(elements)
    plus, moins = elements[0], elements[0]
    for element in elements[1:]:
        if cle(element) > cle(plus):
            plus = element
        if cle(element) < cle(moins):
            moins = element
    return plus, moins


def meilleurs(elements: Sequence[Any], cle: Callable[[Any], Any]) -> tuple[Any, ...]:
    """Tous les elements qui atteignent la plus grande valeur de la cle."""
    plus, _ = extremes(elements, cle)
    return tuple(e for e in elements if cle(e) == cle(plus))


def sans_ceux(
    elements: Sequence[Any], condition: Callable[[Any], bool]
) -> tuple[Any, ...]:
    """Le tableau, une fois supprimes tous ceux qui satisfont la condition."""
    return tuple(e for e in elements if not condition(e))


def rang_qui_suit(elements: Sequence[Any], est_lui: Callable[[Any], bool]) -> int:
    """Le rang ou inserer « apres lui » ; leve si personne ne convient."""
    for rang, element in enumerate(elements):
        if est_lui(element):
            return rang + 1
    msg = "cet element n'existe pas, donc pas d'ajout"
    raise LookupError(msg)


def exiger_strictement_entre(valeur: float, bas: float, haut: float) -> None:
    """Refuse une valeur saisie qui n'est pas strictement entre les deux bornes."""
    if valeur <= bas:
        msg = f"valeur trop petite : {valeur} n'est pas > {bas}"
        raise ValueError(msg)
    if valeur >= haut:
        msg = f"valeur trop grande : {valeur} n'est pas < {haut}"
        raise ValueError(msg)
