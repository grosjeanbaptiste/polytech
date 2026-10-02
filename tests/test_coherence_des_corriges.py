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
import shutil
import subprocess
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


PREAMBULE = TRAVAUX / "preambule.tex"
"""Le preambule partage par les vingt et un corriges d'electromagnetisme."""


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


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_pdf_est_compile_apres_le_preambule_partage(travail: str) -> None:
    r"""Et pas anterieur au preambule, qui est une source comme une autre.

    Trou constate le jour ou le preambule a gagne la macro `\oiint` : la
    garde precedente ne regarde que `resultats.tex`, si bien que les dix
    PDF deja composes restaient « a jour » alors qu'ils avaient ete
    produits par un preambule anterieur. Un changement de preambule qui
    deplacerait une marge ou changerait une police passerait donc
    inapercu sur tous les documents a la fois.

    Le preambule etant partage, cette garde vaut pour les vingt et un
    corriges a venir : une seule ligne modifiee la fera tomber partout,
    ce qui est exactement ce qu'on veut -- il faut alors tout
    recomposer.
    """
    dossier = TRAVAUX / travail
    pdfs = sorted(dossier.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"{travail} : PDF absent, recompiler le document")
    assert pdfs[0].stat().st_mtime >= PREAMBULE.stat().st_mtime


AVERTISSEMENTS_DE_TEX = (
    ("Missing character", "un glyphe absent de la police, perdu en silence"),
    ("Undefined", "un renvoi mort"),
    ("Overfull", "du texte qui deborde dans la marge"),
    ("Underfull", "une ligne lachee, aux espaces distendus"),
    ("has been deprecated", "une commande que le paquet ne maintient plus"),
)
"""Les cinq familles d'avertissement qui signalent un defaut visible.

Reprises du controle qui existait pour le corrige de mecanique des
fluides. Celui-ci ne portait que sur les quatre documents de ce
dossier-la : les corriges d'electromagnetisme y echappaient tous, et
neuf debordements de marge y ont vecu sans etre vus -- un par document,
tous dans la section « Les origines », ou un chemin de fichier en
`\\texttt` ne se coupe nulle part.

Le perimetre est donc desormais le **registre**, comme pour les autres
gardes de ce fichier : un corrige a venir est couvert du seul fait de
s'y inscrire.
"""


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
@pytest.mark.parametrize(("motif", "ce_que_cela_signale"), AVERTISSEMENTS_DE_TEX)
def test_la_compilation_ne_laisse_aucun_avertissement(
    travail: str,
    motif: str,
    ce_que_cela_signale: str,
) -> None:
    """Un avertissement de TeX est un defaut differe, pas un bruit de fond.

    « Missing character » en est le cas extreme : le caractere disparait
    du PDF sans que rien n'echoue. « Overfull » est le plus visible : du
    texte sort dans la marge, et le lecteur le voit avant tout le reste.
    """
    dossier = TRAVAUX / travail
    journaux = [
        fichier for fichier in sorted(dossier.glob("*.log")) if fichier.stem != "texput"
    ]
    if not journaux:
        pytest.skip(f"{travail} : journal absent, lancer lualatex d'abord")
    contenu = journaux[0].read_text(encoding="utf-8", errors="replace")
    occurrences = contenu.count(motif)
    assert occurrences == 0, f"{travail} : {occurrences} fois, {ce_que_cela_signale}"


LIGATURES_A_COMPOSER = (
    ("œ", r"\oe{}"),
    ("æ", r"\ae{}"),
    ("—", "---"),
    ("«", r"\og"),
    ("»", r"\fg{}"),
)
"""Caracteres que la police T1 ne porte pas, et leur ligature LaTeX.

Les guillemets francais sont le cas le plus sournois des cinq : la police
ne les **perd** pas, elle les **remplace**. Leurs deux points de code
tombent, dans le codage T1, sur les positions de « n accent aigu » et de
« z point suscrit » -- si bien que le PDF composait `ń ... ż` autour de
chaque citation, sans le moindre avertissement de lualatex. Onze corriges
sur seize en portaient, et la garde des « Missing character » ne pouvait
rien y voir : aucun caractere ne manquait. Le defaut n'a ete vu qu'en
regardant une page rendue.

Meme defaut que le tiret cadratin litteral, deja garde pour le corrige
de mecanique des fluides : sous lualatex, une police T1 n'a pas ces
points de code et le caractere **disparait du PDF sans que rien
n'echoue**. « nœuds » devenait « nuds », et trois occurrences avaient
passe deux relectures.

La garde est ecrite ici et non dans l'audit des fluides parce que son
perimetre doit etre le registre : chaque corrige a venir est couvert du
seul fait de s'y inscrire. C'est la lecon des neuf debordements de
marge, qui vivaient dans un angle mort du meme genre.
"""


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
@pytest.mark.parametrize(("litteral", "ligature"), LIGATURES_A_COMPOSER)
def test_aucun_document_n_emploie_de_ligature_litterale(
    travail: str,
    litteral: str,
    ligature: str,
) -> None:
    """Le caractere n'est admis que dans les commentaires LaTeX.

    Dans un commentaire il ne se compose pas, donc il ne peut pas
    disparaitre -- et interdire aux commentaires de nommer le caractere
    dont ils expliquent l'absence serait absurde.
    """
    document = _document_du_travail(travail)
    # On numerote sur le fichier entier et non sur les lignes retenues :
    # un rang calcule apres filtrage ne designerait aucune ligne reelle,
    # et le message enverrait le lecteur au mauvais endroit.
    fautives = [
        f"ligne {rang} : {ligne.strip()[:58]}"
        for rang, ligne in enumerate(
            document.read_text(encoding="utf-8").splitlines(),
            1,
        )
        if litteral in ligne and not ligne.lstrip().startswith("%")
    ]
    assert fautives == [], f"{travail} : employer {ligature} plutot que {litteral!r}"


ENVIRONNEMENTS_DE_FIGURE = (r"\begin{tikzpicture}", r"\begin{circuitikz}")
"""Ce qui compte pour une figure : un dessin compose, pas une image citee."""


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_corrige_porte_au_moins_une_figure(travail: str) -> None:
    """« De la meme facon que le devoir de fluides » : donc avec une figure.

    Le devoir de fluides en porte cinq. Le defaut avait ete releve une
    premiere fois sur l'exercice 1 de Thevenin-Norton, corrige a la main,
    et il est revenu sur les **quatorze** corriges suivants : aucun ne
    dessinait la geometrie sur laquelle il raisonnait, alors que chacun
    de ces exercices commence par un schema dans son manuscrit. Une
    correction qui n'est tenue par aucune garde ne vaut que pour le
    document ou elle a ete faite.

    Les commentaires sont exclus : citer l'environnement dans une notice
    ne dessine rien.
    """
    contenu = "\n".join(_lignes_de_contenu(_document_du_travail(travail)))
    assert any(env in contenu for env in ENVIRONNEMENTS_DE_FIGURE), (
        f"{travail} : aucune figure, alors que le raisonnement porte sur un schema"
    )


LETTRES_ETRANGERES = range(0x100, 0x250)
"""Latin etendu A et B : aucune lettre du francais n'y vit, sauf `œ`."""

LIGATURES_FRANCAISES = frozenset("œŒ")


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_pdf_ne_porte_aucune_lettre_que_le_francais_n_emploie_pas(
    travail: str,
) -> None:
    """Un caractere **remplace** ne laisse aucune trace dans le journal.

    Sous lualatex, une police T1 est une police a 256 positions. Un point
    de code Unicode qu'elle ne connait pas n'est pas toujours perdu : s'il
    est inferieur a 256, il tombe sur la position de meme numero, qui
    porte une autre lettre. Les guillemets francais sortaient ainsi en
    `n accent aigu` et `z point suscrit`, et le prefixe « micro » de
    siunitx en `t cedille` -- soixante microhenrys devenaient `60 tH`.

    Aucun avertissement : rien ne manque, donc la garde des « Missing
    character » ne voit rien. Ce test regarde le resultat, la ou les
    gardes de source ne regardent que les causes deja connues : il a
    attrape le prefixe « micro », que la garde des guillemets ne pouvait
    pas prevoir.

    Il lit un PDF, donc il se suspend quand le PDF ou `pdftotext` manque
    -- un clone frais n'a ni l'un ni forcement l'autre. C'est une
    verification, non une mesure : rien du depot n'en depend.
    """
    pdfs = sorted((TRAVAUX / travail).glob("*.pdf"))
    lecteur = shutil.which("pdftotext")
    if not pdfs or lecteur is None:
        pytest.skip(f"{travail} : PDF ou pdftotext absent")
    lu = subprocess.run(  # noqa: S603 -- binaire resolu par which, arguments fixes
        [lecteur, str(pdfs[0]), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    etrangeres = sorted(
        {
            caractere
            for caractere in lu.stdout
            if ord(caractere) in LETTRES_ETRANGERES
            and caractere not in LIGATURES_FRANCAISES
        },
    )
    assert etrangeres == [], f"{travail} : la police a substitue {etrangeres}"


RECUEIL = "Exs_Notes_manuscrites.pdf"
"""Le recueil d'exercices : il porte les enonces, les figures, les reponses."""

SANS_ENONCE_AU_RECUEIL = {
    "exs-02-statique-1-1-poisson": (
        "le manuscrit s'intitule « Par Poisson, ext. » et ne renvoie a aucun "
        "exercice numerote du recueil ; son enonce n'existe que par lui"
    ),
}
"""Les corriges dont l'enonce ne figure pas au recueil, et pourquoi."""


@pytest.mark.parametrize("travail", sorted(JEUX_DE_MACROS))
def test_le_corrige_remonte_au_recueil_et_non_au_seul_manuscrit(travail: str) -> None:
    """L'enonce se lit dans le recueil ; le manuscrit n'en est que la solution.

    Les seize premiers corriges d'electromagnetisme ont ete ecrits sans
    ouvrir le recueil. Leur enonce etait donc **reconstitue** d'apres le
    manuscrit, et trois defauts en sont sortis, tous de l'agent :

    - un enonce faux -- la « charge ponctuelle » de l'exercice 7 de
      statique est une sphere isolante chargee, ce qui est justement la
      raison pour laquelle le manuscrit commence par Gauss ;
    - des figures dites « de principe » alors que la figure officielle
      existe : le tore a deux parties du circuit magnetique, l'electro-
      aimant a noyau plongeur, les armatures de revolution ;
    - une these entiere batie sur une transcription fautive : l'exercice
      8 des champs porte sur des spires de **largeurs** 2, 4, 8 cm..., le
      manuscrit divise bien par deux, et le corrige l'accusait de deux
      erreurs qui se compensent.

    Citer le recueil ne prouve pas qu'on l'a lu ; ne pas le citer prouve
    qu'on ne l'a pas ouvert.
    """
    contenu = "\n".join(_lignes_de_contenu(_document_du_travail(travail)))
    if travail in SANS_ENONCE_AU_RECUEIL:
        assert RECUEIL not in contenu, f"{travail} : exemption devenue inutile"
        return
    assert RECUEIL in contenu, f"{travail} : le recueil n'est pas cite"
