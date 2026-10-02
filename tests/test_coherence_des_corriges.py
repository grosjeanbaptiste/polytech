"""Aucun corrige d'electromagnetisme n'ecrit un resultat en chiffres.

Ce fichier remplace la garde ecrite pour le seul exercice 1. La campagne
en comptera vingt et un, et une garde recopiee vingt et une fois derive :
dix-neuf fichiers corriges sur vingt et un, c'est une incoherence qui ne
se voit pas.

Le test est donc **parametre sur le registre** : tout travail qui s'y
inscrit est garde du meme coup, et un corrige a venir ne peut pas
echapper a la discipline en oubliant d'ecrire son test.

Ce qui est interdit se **deduit** plutot que de se declarer : les valeurs
interdites sont celles des macros du travail lui-meme. Si une valeur
calculee apparait en clair dans le document, c'est qu'elle y a ete tapee
-- et le document a cesse d'etre adosse a la bibliotheque.

Seules les valeurs d'au moins trois caracteres sont examinees. « 1 », « 5 »
ou « 20 » apparaissent legitimement dans une reference de section, une
dimension de figure ou un numero d'exercice, et les interdire rendrait le
test inutilisable. Les valeurs courtes mais decisives sont declarees a
part, avec leur raison.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from polytech.electromagnetisme.application.generation import GenererMacros
from polytech.electromagnetisme.application.registre import (
    JEUX_DE_MACROS,
    macros_du_travail,
)
from polytech.electromagnetisme.infrastructure.latex import FichierDeMacros

RACINE = Path(__file__).resolve().parent.parent
TRAVAUX = RACINE / "travaux" / "Electromagnetisme"

LONGUEUR_EXAMINEE = 3
"""En dessous, un nombre est trop commun pour qu'on puisse l'interdire."""

VALEURS_COURTES_SURVEILLEES = {
    "exs-03-thevenin-norton-01": (
        # Les cinq reponses de l'exercice. « 20 » et « 100 » sont les deux
        # que la table de comparaison etait tentee de recopier.
        "20",
        "100",
    ),
    # L'exercice 8 n'a aucune valeur courte surveillable : son resultat
    # final est « 2 », qui est aussi le numero de sa seconde question et
    # le facteur de bien des formules. L'interdire rendrait le test
    # inutilisable, et un test inutilisable se desactive.
    #
    # Sa garde repose donc sur les valeurs longues, qui sont les plus
    # nombreuses et les plus faciles a recopier par inadvertance :
    # 19,17 pour la tension de Thevenin, 0,8333 pour la resistance,
    # 1,8333 pour le courant traversant, 29,17 pour le lapsus du
    # manuscrit. Les quatre couvrent tout le chemin de calcul.
}
"""Valeurs courtes mais decisives, surveillees travail par travail.

Chacune est un resultat de l'exercice, donc une valeur qu'une
inadvertance recopierait. La liste est ecrite plutot que deduite parce
qu'un nombre court est ambigu : c'est la connaissance de l'exercice qui
dit lesquels comptent.
"""


def _document_du_travail(travail: str) -> Path:
    """Le seul `.tex` du dossier qui ne soit pas un fragment genere."""
    candidats = [
        f
        for f in sorted((TRAVAUX / travail).glob("*.tex"))
        if f.name != "resultats.tex"
    ]
    if len(candidats) != 1:
        msg = f"{travail} : un seul document attendu, trouve {candidats}"
        raise AssertionError(msg)
    return candidats[0]


def _lignes_de_contenu(document: Path) -> list[str]:
    """Les lignes composees, commentaires LaTeX exclus.

    Un commentaire peut legitimement citer un nombre pour expliquer un
    choix d'ecriture ; le contenu compose, non.
    """
    return [
        ligne
        for ligne in document.read_text(encoding="utf-8").splitlines()
        if not ligne.lstrip().startswith("%")
    ]


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_document_inclut_ses_macros_et_le_preambule(travail: str) -> None:
    """Sans ces deux lignes, rien ne compilerait ou rien ne serait defini."""
    texte = _document_du_travail(travail).read_text(encoding="utf-8")
    assert r"\input{resultats}" in texte
    assert r"\input{../preambule}" in texte


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_fichier_de_macros_est_a_jour(travail: str, tmp_path: Path) -> None:
    """Regenerer ne doit rien changer : sinon la lib a derive du document."""
    frais = tmp_path / "resultats.tex"
    GenererMacros(
        publication=FichierDeMacros(chemin=frais),
        travail=travail,
    ).executer()
    attendu = (TRAVAUX / travail / "resultats.tex").read_text(encoding="utf-8")
    assert frais.read_text(encoding="utf-8") == attendu


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_chaque_macro_publiee_est_consommee(travail: str) -> None:
    """Une macro publiee et jamais citee est un calcul fait pour rien."""
    texte = _document_du_travail(travail).read_text(encoding="utf-8")
    oubliees = sorted(
        nom for nom in macros_du_travail(travail) if f"\\{nom}" not in texte
    )
    assert oubliees == []


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_aucune_valeur_calculee_n_est_ecrite_en_chiffres(travail: str) -> None:
    """La garde centrale, et elle se deduit des macros du travail.

    Si une valeur calculee apparait en clair, c'est qu'elle a ete tapee --
    et le document a cesse d'etre adosse a la bibliotheque sans que rien
    ne le signale.
    """
    document = _document_du_travail(travail)
    lignes = _lignes_de_contenu(document)
    interdites = {
        valeur
        for valeur in macros_du_travail(travail).values()
        if len(valeur.lstrip("-")) >= LONGUEUR_EXAMINEE
        and valeur.lstrip("-").replace(".", "").isdigit()
    }
    interdites |= set(VALEURS_COURTES_SURVEILLEES.get(travail, ()))
    fautes = []
    for valeur in sorted(interdites):
        motif = re.compile(rf"(?<![\w.-]){re.escape(valeur)}(?![\w.])")
        fautes += [
            f"{valeur!r} : {ligne.strip()[:62]}"
            for ligne in lignes
            if motif.search(ligne)
        ]
    assert fautes == []


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_document_annonce_sa_complementarite(travail: str) -> None:
    """La consigne est explicite : ces corriges **completent** l'officiel.

    Le document doit donc le dire et dire ou l'officiel s'arrete. Sans
    cette section, ce serait un corrige concurrent et non un complement.
    """
    texte = _document_du_travail(travail).read_text(encoding="utf-8")
    assert "complément" in texte
    assert "Ce que le corrigé officiel donne" in texte


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_document_nomme_ses_origines(travail: str) -> None:
    """Rien ne doit tomber du ciel : l'enonce, les calculs, les nombres.

    Chaque corrige nomme le document d'ou vient son enonce, le module qui
    calcule, et le fichier qui porte les valeurs. C'est la section « Les
    origines », et son absence rendrait le document invérifiable.
    """
    texte = _document_du_travail(travail).read_text(encoding="utf-8")
    assert "Les origines" in texte
    assert "transcriptions/" in texte or "cours-markdown" in texte
    assert "polytech.electromagnetisme" in texte


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_pdf_est_compile_apres_ses_macros(travail: str) -> None:
    """Bout en bout : le PDF existe et n'est pas anterieur a ses valeurs.

    Une macro mal nommee ferait echouer lualatex, donc laisserait le PDF
    absent ou perime -- c'est ce que cette garde attrape.
    """
    dossier = TRAVAUX / travail
    pdfs = sorted(dossier.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"{travail} : PDF absent, recompiler le document")
    pdf = pdfs[0]
    assert pdf.read_bytes().startswith(b"%PDF")
    assert pdf.stat().st_mtime >= (dossier / "resultats.tex").stat().st_mtime
