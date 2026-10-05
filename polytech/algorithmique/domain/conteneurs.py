"""Les conteneurs de la STL et leurs fonctions membres.

Le cours (seances 5 et 6, « Fonctions membres » ; note « Les fonctions
membres ») : la STL, ce sont des conteneurs, des iterateurs et des
algorithmes. Tous les conteneurs n'offrent pas les memes fonctions
membres, et l'oublier ne compile pas : « 'sort' : n'est pas membre de
std::vector ».

Ce fichier recopie les deux tableaux du cours -- les cinq conteneurs
sequentiels, les quatre associatifs -- ligne par ligne : chaque fonction
membre y nomme les conteneurs qui l'ont. C'est une table de donnees, d'ou
sa longueur.
"""

from __future__ import annotations

from typing import Final

SEQUENTIELS: Final = ("array", "vector", "deque", "forward_list", "list")
ASSOCIATIFS: Final = ("set", "multiset", "map", "multimap")

_TOUS: Final = "array vector deque forward_list list"
_SANS_ARRAY: Final = "vector deque forward_list list"
_A_DEUX_BOUTS: Final = "array vector deque list"
_PAR_INDICE: Final = "array vector deque"
_PAR_LA_FIN: Final = "vector deque list"
_PAR_LE_DEBUT: Final = "deque forward_list list"
_CHAINES: Final = "forward_list list"
_ASSOCIATIFS: Final = "set multiset map multimap"

_LIGNES: Final = {
    "front": _TOUS,
    "back": _A_DEUX_BOUTS,
    "operator[]": f"{_PAR_INDICE} map",
    "at": f"{_PAR_INDICE} map",
    "assign": _SANS_ARRAY,
    "emplace": f"{_PAR_LA_FIN} {_ASSOCIATIFS}",
    "insert": f"{_PAR_LA_FIN} {_ASSOCIATIFS}",
    "erase": f"{_PAR_LA_FIN} {_ASSOCIATIFS}",
    "emplace_back": _PAR_LA_FIN,
    "push_back": _PAR_LA_FIN,
    "pop_back": _PAR_LA_FIN,
    "emplace_front": _PAR_LE_DEBUT,
    "push_front": _PAR_LE_DEBUT,
    "pop_front": _PAR_LE_DEBUT,
    "clear": f"{_SANS_ARRAY} {_ASSOCIATIFS}",
    "swap": f"{_TOUS} {_ASSOCIATIFS}",
    "splice": "list",
    "remove": _CHAINES,
    "remove_if": _CHAINES,
    "unique": _CHAINES,
    "merge": _CHAINES,
    "sort": _CHAINES,
    "reverse": _CHAINES,
    "get_allocator": f"{_SANS_ARRAY} {_ASSOCIATIFS}",
    "data": "array vector",
    "begin": f"{_TOUS} {_ASSOCIATIFS}",
    "end": f"{_TOUS} {_ASSOCIATIFS}",
    "rbegin": f"{_A_DEUX_BOUTS} {_ASSOCIATIFS}",
    "rend": f"{_A_DEUX_BOUTS} {_ASSOCIATIFS}",
    "size": f"{_A_DEUX_BOUTS} {_ASSOCIATIFS}",
    "max_size": f"{_TOUS} {_ASSOCIATIFS}",
    "capacity": "vector",
    "empty": f"{_TOUS} {_ASSOCIATIFS}",
    "resize": _SANS_ARRAY,
    "emplace_hint": _ASSOCIATIFS,
    "count": _ASSOCIATIFS,
    "find": _ASSOCIATIFS,
    "equal_range": _ASSOCIATIFS,
    "lower_bound": _ASSOCIATIFS,
    "upper_bound": _ASSOCIATIFS,
    "key_comp": _ASSOCIATIFS,
    "value_comp": _ASSOCIATIFS,
}
MEMBRES: Final = {membre: frozenset(qui.split()) for membre, qui in _LIGNES.items()}
"""Chaque fonction membre, et les conteneurs qui la portent."""


def possede(conteneur: str, membre: str) -> bool:
    """Vrai si le conteneur a cette fonction membre."""
    if conteneur not in (*SEQUENTIELS, *ASSOCIATIFS):
        msg = f"conteneur inconnu : {conteneur}"
        raise ValueError(msg)
    return conteneur in MEMBRES.get(membre, frozenset())


def exiger(conteneur: str, membre: str) -> None:
    """L'erreur du compilateur quand le membre manque au conteneur."""
    if not possede(conteneur, membre):
        msg = f"{membre} n'est pas membre de std::{conteneur}"
        raise AttributeError(msg)
