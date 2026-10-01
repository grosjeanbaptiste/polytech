"""Le plan du cours, tel que sa planche 0 l'annonce.

Le chapitre 0 est administratif -- qui enseigne, comment on est evalue,
combien de seances d'exercices -- a une exception pres : il annonce les
quatre chapitres du cours. C'est son seul contenu, et le porter comme
**donnee** vaut mieux que de declarer le chapitre sans contenu.

Ce que cela sert : repondre a « ou cela se trouve-t-il ? ». Un corrige
qui cite une notion doit pouvoir dire de quel chapitre elle vient, et
c'est ce que `chapitre_du_sujet` fournit.
"""

from __future__ import annotations

from typing import Final

CHAPITRES: Final = (
    "circuits lineaires : Laplace, phaseurs, Fourier, Bode, nodale, quadripoles",
    "amplificateurs operationnels et leurs applications",
    "filtrage analogique : circuits passifs LC et circuits actifs",
    "instrumentation : chaine d'acquisition, bruit, conversion",
)
"""Les quatre chapitres, dans l'ordre de la planche 0."""

SUJETS: Final = {
    "laplace": 0,
    "phaseurs": 0,
    "fourier": 0,
    "bode": 0,
    "analyse nodale": 0,
    "quadripole": 0,
    "longueur electrique": 0,
    "amplificateur operationnel": 1,
    "retroaction": 1,
    "cmrr": 1,
    "miller": 1,
    "gabarit": 2,
    "butterworth": 2,
    "chebyshev": 2,
    "bessel": 2,
    "sallen-key": 2,
    "transposition": 2,
    "echantillonnage": 3,
    "quantification": 3,
    "bruit": 3,
    "friis": 3,
    "temps d'ouverture": 3,
}
"""A quel chapitre chaque sujet du cours se rattache.

La table est une **donnee** et non une regle : le rattachement vient du
plan, et le deviner par mots-cles donnerait des surprises -- « bruit »
appartient a l'instrumentation, pas aux circuits lineaires, bien qu'il y
soit mentionne.
"""


def nombre_de_chapitres() -> int:
    """Quatre, et c'est ce que la planche 0 annonce."""
    return len(CHAPITRES)


def chapitre_du_sujet(sujet: str) -> str:
    """Le chapitre qui traite ce sujet ; leve s'il est etranger au cours.

    Le dire plutot que de rattacher au hasard : un corrige qui citerait un
    chapitre faux egarerait son lecteur plus sûrement qu'un corrige qui
    n'en citerait aucun.
    """
    cherche = sujet.casefold()
    for cle, rang in SUJETS.items():
        if cle in cherche or cherche in cle:
            return CHAPITRES[rang]
    msg = f"sujet hors du cours : {sujet!r}"
    raise ValueError(msg)
