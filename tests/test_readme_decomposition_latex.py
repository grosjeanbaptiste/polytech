"""Le README decompose son total ; la decomposition doit tenir aussi.

Le nombre total de documents LaTeX est garde depuis longtemps. La phrase
qui le **decompose** -- « les quatre de l'activite d'integration, celui
du TP 1, les neuf corriges d'electromagnetisme, et les 398 planches » --
ne l'etait pas, et elle a derive deux fois pendant la campagne : chaque
corrige ajoute change un terme de la somme, et le terme est ecrit en
lettres.

Une somme qui ne fait pas son total est pire qu'une somme absente : elle
donne au lecteur un moyen de verifier, et ce moyen mentirait.

Le test porte sur le terme qui bouge -- le compte des corriges
d'electromagnetisme -- et sur la somme entiere. Les trois autres termes
sont stables, mais la somme les attrape si l'un change.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

RACINE = Path(__file__).resolve().parent.parent
README = RACINE / "README.md"
TRAVAUX = RACINE / "travaux" / "Electromagnetisme"
TRAVAUX_SIGNAUX = RACINE / "travaux" / "SignauxSystemes"
TRAVAUX_THERMIQUE = RACINE / "travaux" / "ThermiqueFondements"
TRAVAUX_ELECTRONIQUE = RACINE / "travaux" / "ElectroniqueInstrumentation"
TRAVAUX_MATHEMATIQUE = RACINE / "travaux" / "Mathematique"
TRAVAUX_OPTIMISATION = RACINE / "travaux" / "Optimisation"
TRAVAUX_PROGRAMMATION = RACINE / "travaux" / "ProgrammationAlgorithmique"

NUMERAUX: Final = {
    "un": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "dix-sept": 17,
    "dix-huit": 18,
    "dix-neuf": 19,
    "vingt": 20,
    "vingt et un": 21,
    "vingt-deux": 22,
    "vingt-trois": 23,
    "vingt-quatre": 24,
    "vingt-cinq": 25,
    "vingt-six": 26,
    "vingt-sept": 27,
    "vingt-huit": 28,
    "vingt-neuf": 29,
    "trente": 30,
    "trente et un": 31,
    "trente-deux": 32,
    "trente-trois": 33,
    "trente-quatre": 34,
    "trente-cinq": 35,
    "trente-six": 36,
    "trente-sept": 37,
    "trente-huit": 38,
    "trente-neuf": 39,
}
"""De un a trente-neuf : seances, laboratoires, puis interrogations cotees."""

ANNONCE = re.compile(r"\*\*(\d+) documents LaTeX\*\*")
DECOMPOSITION = re.compile(
    r"les [\w-]+ des deux activités\s+d'intégration de mécanique des fluides, "
    r"les [\w-]+ du devoir coté,\s+"
    r"(?:celui du TP 1|[\w-]+ corrigés) de thermique,\s+"
    r"(?:une?|[\w-]+)\s+corrigés? d'électronique,\s+"
    r"(?:une?|[\w-]+)\s+corrigés? de mathématique,\s+"
    r"(?:une?|[\w-]+)\s+corrigés? d'optimisation,\s+"
    r"(?:une?|[\w-]+)\s+corrigés? de programmation,\s+"
    r"(?:une?|[\w-]+)\s+corrigés? de signaux, "
    r"(?:une?|[\w-]+)\s+fiches? de révision, "
    r"les\s+([\w-]+(?: et un)?) corrigés "
    r"d'électromagnétisme déjà écrits, et les (\d+) planches",
)


FICHE = "fiche-de-revision"
"""Le dossier d'une fiche de revision : un document compose, pas un corrige."""


def _corriges_de(matiere: Path) -> int:
    """Les dossiers de travail qui portent un document compose, fiche exceptee."""
    return sum(
        1
        for dossier in matiere.iterdir()
        if dossier.is_dir()
        and dossier.name != FICHE
        and any(f.name != "resultats.tex" for f in dossier.glob("*.tex"))
    )


def _fiches_de_revision() -> int:
    """Les fiches de revision, une par matiere qui en a une."""
    return len(list((RACINE / "travaux").glob(f"*/{FICHE}/*.tex")))


ACTIVITES = re.compile(r"les ([\w-]+) des deux activités\s+d'intégration")
"""Le terme des fluides : il a cesse d'etre « les quatre » avec l'activite 2."""


def _documents_des_activites() -> int:
    """Les documents des activites d'integration : corriges et mises en page."""
    fluides = RACINE / "travaux" / "MecaniqueFluides"
    return sum(
        1
        for source in fluides.glob("activite-integration-*/*.tex")
        if r"\begin{document}" in source.read_text(encoding="utf-8")
    )


DEVOIR = re.compile(r"les ([\w-]+) du devoir coté")
"""Le devoir cote de fluides : corrige detaille et rapport en deux mises en page."""


def _documents_du_devoir_cote() -> int:
    dossier = RACINE / "travaux" / "MecaniqueFluides" / "devoir-cote-cylindre"
    return sum(
        1
        for source in dossier.glob("*.tex")
        if r"\begin{document}" in source.read_text(encoding="utf-8")
    )


def test_le_nombre_de_documents_du_devoir_cote_annonce_en_lettres_est_exact() -> None:
    trouve = DEVOIR.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    assert NUMERAUX[trouve.group(1)] == _documents_du_devoir_cote()


def test_le_nombre_de_documents_des_activites_annonce_en_lettres_est_exact() -> None:
    """Quatre pour la premiere, trois pour la seconde -- qui n'a pas d'apercu."""
    trouve = ACTIVITES.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    assert NUMERAUX[trouve.group(1)] == _documents_des_activites()


FICHES = re.compile(r"(une?|[\w-]+)\s+fiches? de révision, ")
"""Le terme des fiches, ouvert par celle de mathematique."""


def test_le_nombre_de_fiches_de_revision_annonce_en_lettres_est_exact() -> None:
    trouve = FICHES.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    annonce = 1 if trouve.group(1) in {"un", "une"} else NUMERAUX[trouve.group(1)]
    assert annonce == _fiches_de_revision()


def _corriges_d_electromagnetisme() -> int:
    """Les corriges d'electromagnetisme, un par dossier de travail."""
    return _corriges_de(TRAVAUX)


SIGNAUX = re.compile(r"(une?|[\w-]+)\s+corrigés? de signaux")
"""Le terme des signaux, ecrit en lettres comme celui d'electromagnetisme."""


def test_le_nombre_de_corriges_de_signaux_annonce_en_lettres_est_exact() -> None:
    """Le TP 1 de signaux a ouvert un second terme qui bougera, lui aussi."""
    trouve = SIGNAUX.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    annonce = 1 if trouve.group(1) in {"un", "une"} else NUMERAUX[trouve.group(1)]
    assert annonce == _corriges_de(TRAVAUX_SIGNAUX)


THERMIQUE = re.compile(r"([\w-]+) corrigés de thermique")
"""Le terme de la thermique, qui a cesse d'etre un document unique au TP 2."""


def test_le_nombre_de_corriges_de_thermique_annonce_en_lettres_est_exact() -> None:
    trouve = THERMIQUE.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    assert NUMERAUX[trouve.group(1)] == _corriges_de(TRAVAUX_THERMIQUE)


ELECTRONIQUE = re.compile(r"(une?|[\w-]+)\s+corrigés? d'électronique")
"""Le terme de l'electronique, ouvert par son TP 1."""


def test_le_nombre_de_corriges_d_electronique_annonce_en_lettres_est_exact() -> None:
    trouve = ELECTRONIQUE.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    annonce = 1 if trouve.group(1) in {"un", "une"} else NUMERAUX[trouve.group(1)]
    assert annonce == _corriges_de(TRAVAUX_ELECTRONIQUE)


MATHEMATIQUE = re.compile(r"(une?|[\w-]+)\s+corrigés? de mathématique")
"""Le terme de la mathematique, ouvert par sa premiere feuille d'exercices."""


def test_le_nombre_de_corriges_de_mathematique_annonce_en_lettres_est_exact() -> None:
    trouve = MATHEMATIQUE.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    annonce = 1 if trouve.group(1) in {"un", "une"} else NUMERAUX[trouve.group(1)]
    assert annonce == _corriges_de(TRAVAUX_MATHEMATIQUE)


OPTIMISATION = re.compile(r"(une?|[\w-]+)\s+corrigés? d'optimisation")
"""Le terme de l'optimisation, ouvert par son premier jeu de planches."""


def test_le_nombre_de_corriges_d_optimisation_annonce_en_lettres_est_exact() -> None:
    trouve = OPTIMISATION.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    annonce = 1 if trouve.group(1) in {"un", "une"} else NUMERAUX[trouve.group(1)]
    assert annonce == _corriges_de(TRAVAUX_OPTIMISATION)


PROGRAMMATION = re.compile(r"(une?|[\w-]+)\s+corrigés? de programmation")
"""Le terme de la programmation, ouvert par ses deux feuilles de TP."""


def test_le_nombre_de_corriges_de_programmation_annonce_en_lettres_est_exact() -> None:
    trouve = PROGRAMMATION.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    annonce = 1 if trouve.group(1) in {"un", "une"} else NUMERAUX[trouve.group(1)]
    assert annonce == _corriges_de(TRAVAUX_PROGRAMMATION)


def test_la_decomposition_est_bien_presente_dans_le_readme() -> None:
    """Sans elle, les deux tests suivants passeraient a vide.

    C'est le piege des tests qui s'appuient sur une recherche : si le
    motif ne trouve rien, l'assertion suivante peut tenir par vacuite.
    """
    assert DECOMPOSITION.search(README.read_text(encoding="utf-8")) is not None


def test_le_nombre_de_corriges_annonce_en_lettres_est_exact() -> None:
    """« neuf corrigés d'électromagnétisme » doit en compter neuf."""
    trouve = DECOMPOSITION.search(README.read_text(encoding="utf-8"))
    assert trouve is not None
    assert NUMERAUX[trouve.group(1)] == _corriges_d_electromagnetisme()


def test_la_decomposition_fait_bien_le_total_annonce() -> None:
    """Quatre, plus un, plus les deux comptes de corriges, plus les planches.

    C'est le controle que le lecteur ferait, et il doit aboutir.
    """
    texte = README.read_text(encoding="utf-8")
    total = ANNONCE.search(texte)
    trouve = DECOMPOSITION.search(texte)
    assert total is not None
    assert trouve is not None
    signaux = _corriges_de(TRAVAUX_SIGNAUX)
    thermique = _corriges_de(TRAVAUX_THERMIQUE)
    electronique = _corriges_de(TRAVAUX_ELECTRONIQUE)
    mathematique = _corriges_de(TRAVAUX_MATHEMATIQUE)
    optimisation = _corriges_de(TRAVAUX_OPTIMISATION)
    programmation = _corriges_de(TRAVAUX_PROGRAMMATION)
    somme = (
        _documents_des_activites()
        + _documents_du_devoir_cote()
        + thermique
        + electronique
        + mathematique
        + optimisation
        + programmation
        + signaux
        + _fiches_de_revision()
        + NUMERAUX[trouve.group(1)]
        + int(trouve.group(2))
    )
    assert somme == int(total.group(1))


MENTIONS = re.compile(r"(\d+) documents LaTeX")
"""Toute mention du total, en gras ou non, ou qu'elle soit dans le README."""


def test_toutes_les_mentions_du_total_s_accordent() -> None:
    """Le total est ecrit a plusieurs endroits, et un seul etait garde.

    La table des outils disait encore « compose les 410 documents LaTeX »
    quand l'annonce en gras en comptait 418 : la garde du total cherchait
    la premiere occurrence et s'arretait la. Un nombre ecrit deux fois est
    deux nombres a tenir.
    """
    texte = README.read_text(encoding="utf-8")
    annonce = ANNONCE.search(texte)
    assert annonce is not None
    mentions = {int(m) for m in MENTIONS.findall(texte)}
    assert mentions == {int(annonce.group(1))}
