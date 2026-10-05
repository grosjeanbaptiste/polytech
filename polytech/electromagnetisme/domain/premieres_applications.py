"""Les premieres applications de l'electromagnetisme.

Le chapitre I des planches, apres sa chronologie : « les premieres
applications font leur apparition ». Deux sont datees. Tesla invente en
1887 le moteur electrique -- l'annee meme ou Hertz confirme Maxwell -- et
defend le courant alternatif, seul a permettre alors d'abaisser la
tension par des transformateurs. Marconi, « en se basant sur les travaux
de Hertz », realise en 1901 la premiere liaison radio transatlantique.
"""

from __future__ import annotations

from typing import Final

from polytech.electromagnetisme.domain.chronologie import Jalon

PREMIERES_APPLICATIONS: Final = (
    Jalon(
        1887,
        "Nikola Tesla",
        "le moteur electrique, reversible en generateur, et l'usage du "
        "courant alternatif pour transporter l'energie au loin",
    ),
    Jalon(
        1901,
        "Guglielmo Marconi",
        "la premiere liaison radio transatlantique, sur les travaux de Hertz",
    ),
)
"""Les deux applications que le fascicule date, apres ses dix jalons."""


def application(auteur: str) -> Jalon:
    """La premiere application dont l'auteur porte ce nom."""
    for candidate in PREMIERES_APPLICATIONS:
        if auteur.casefold() in candidate.auteur.casefold():
            return candidate
    msg = f"aucune application pour {auteur!r} dans les planches du chapitre I"
    raise KeyError(msg)
