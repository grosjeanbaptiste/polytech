"""Comparer les traces d'une figure a une reference que `polytech` calcule."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from outils.fiches.pgf import Trace, Valeur, traces
from tests.fiches_toml import figure

Reference = Callable[[float], Valeur]
ECHANTILLONS = 9


def _en_tuple(valeur: Valeur) -> tuple[float, ...]:
    return valeur if isinstance(valeur, tuple) else (valeur,)


def constante(valeur: float) -> Reference:
    """Une reference qui ne depend pas du parametre (une asymptote)."""
    return lambda _t: valeur


def modulo(valeur: Valeur, tour: float) -> float:
    """Un angle ramene dans `[0, tour[`, pour comparer des angles."""
    return _en_tuple(valeur)[0] % tour


def _proches(a: Valeur, b: Valeur, relative: float) -> bool:
    paires = zip(_en_tuple(a), _en_tuple(b), strict=True)
    return all(abs(x - y) <= relative * max(1.0, abs(y)) for x, y in paires)


def _parametres(trace: Trace) -> list[float]:
    if trace.domaine is None:
        msg = "trace sans domaine : rien a echantillonner"
        raise AssertionError(msg)
    bas, haut = trace.domaine
    return [bas + (haut - bas) * i / (ECHANTILLONS - 1) for i in range(ECHANTILLONS)]


def verifie(
    matiere: str,
    notion: str,
    references: Sequence[Reference | Sequence[tuple[float, ...]]],
    *,
    relative: float = 1e-6,
) -> None:
    """Chaque trace de la figure egale sa reference, point par point.

    Une reference est une fonction du parametre, ou, pour une trace de
    points isoles, la liste des points que la lib donne.
    """
    lues = traces(figure(matiere, notion))
    assert len(lues) == len(references), (notion, len(lues))
    for trace, reference in zip(lues, references, strict=True):
        if callable(reference):
            for t in _parametres(trace):
                assert _proches(trace.valeur(t), reference(t), relative), (notion, t)
        else:
            assert len(trace.points) == len(reference), notion
            for lu, attendu in zip(trace.points, reference, strict=True):
                assert _proches(lu, attendu, relative), (notion, lu, attendu)
