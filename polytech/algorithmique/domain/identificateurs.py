"""Les identificateurs du C++, ses mots reserves, et le type que deduit `auto`.

Le cours (partie 1, planches 5, 10 et 14) : un identificateur nomme une
variable ou une fonction. Il se compose de lettres, de chiffres et de
traits bas, ne commence pas par un chiffre, et n'est pas un mot reserve.
Le langage distingue les majuscules des minuscules.

Une variable `auto` prend le type de la valeur qui l'initialise : elle ne
peut donc pas etre declaree sans valeur.
"""

from __future__ import annotations

from typing import Final

_MOTS: Final = """
alignas alignof and and_eq asm auto bitand bitor bool break case catch
char char8_t char16_t char32_t class compl concept const consteval
constexpr constinit const_cast continue co_await co_return co_yield
decltype default delete do double dynamic_cast else enum explicit
export extern false float for friend goto if inline int long mutable
namespace new noexcept not not_eq nullptr operator or or_eq private
protected public register reinterpret_cast requires return short
signed sizeof static static_assert static_cast struct switch template
this thread_local throw true try typedef typeid typename union
unsigned using virtual void volatile wchar_t while xor xor_eq
"""
MOTS_RESERVES: Final = frozenset(_MOTS.split())
"""Les mots-cles standard du C++ (planche 5), tous en minuscules."""


def motif_de_refus(nom: str) -> str | None:
    """Pourquoi ce nom n'est pas un identificateur, ou `None` s'il en est un."""
    if not nom:
        return "identificateur vide"
    if nom[0].isdigit():
        return "commence par un chiffre"
    for caractere in nom:
        if not (caractere.isascii() and (caractere.isalnum() or caractere == "_")):
            lisible = "espace" if caractere == " " else caractere
            return f"caractere non autorise ({lisible})"
    if nom in MOTS_RESERVES:
        return "mot reserve du langage"
    return None


def est_un_identificateur(nom: str) -> bool:
    """Vrai si le nom peut designer une variable ou une fonction."""
    return motif_de_refus(nom) is None


def type_deduit(valeur: object, guillemets: str = '"') -> str:
    """Le type qu'`auto` donne a une variable initialisee par cette valeur."""
    if valeur is None:
        msg = "auto exige une initialisation : sans valeur, pas de type a deduire"
        raise ValueError(msg)
    if isinstance(valeur, bool):
        return "bool"
    if isinstance(valeur, int):
        return "int"
    if isinstance(valeur, float):
        return "double"
    return "char" if guillemets == "'" else "const char*"
