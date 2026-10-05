"""Le mode de derivation : ce que devient un membre herite.

Le syllabus de programmation objet (3.2) : `class ClasseDerivee : <mode>
ClasseBase` ou le mode vaut `public`, `protected` ou `private` -- et
`private` quand une classe n'en precise aucun. Le mode « determine quels
membres de la classe de base sont accessibles dans la classe derivee » :

* `public` : rien ne change ;
* `protected` : ce qui etait public devient protege ;
* `private` : ce qui etait public ou protege devient prive.

Dans les trois cas, un membre prive de la classe de base est inaccessible
a la classe derivee. Autrement dit, le membre herite prend le plus
restrictif des deux statuts, le sien et le mode.
"""

from __future__ import annotations

from typing import Final

from polytech.algorithmique.domain.encapsulation import (
    PRIVE,
    PROTEGE,
    PUBLIC,
    visibilite_par_defaut,
)

INACCESSIBLE: Final = "inaccessible"
"""Le statut d'un membre prive de la base, vu de la classe derivee."""

_DU_PLUS_OUVERT_AU_PLUS_FERME: Final = (PUBLIC, PROTEGE, PRIVE)


def mode_par_defaut(mot_cle: str) -> str:
    """Le mode d'une derivation qui n'en precise aucun : prive pour une classe."""
    return visibilite_par_defaut(mot_cle)


def statut_dans_la_derivee(mode: str, statut: str) -> str:
    """Le statut, dans la classe derivee, d'un membre de la classe de base."""
    for mot in (mode, statut):
        if mot not in _DU_PLUS_OUVERT_AU_PLUS_FERME:
            msg = f"ni public, ni protected, ni private : {mot}"
            raise ValueError(msg)
    if statut == PRIVE:
        return INACCESSIBLE
    rangs = [_DU_PLUS_OUVERT_AU_PLUS_FERME.index(mot) for mot in (mode, statut)]
    return _DU_PLUS_OUVERT_AU_PLUS_FERME[max(rangs)]
