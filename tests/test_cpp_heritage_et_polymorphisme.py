"""Programmation objet, seance 5 : heritage et polymorphisme.

Sources, dans `ProgrammationAlgorithmique/SLIDES/` :
`Slides_2023_2024_S5.pdf` (planches 8 a 27) et
`Slides_2025_2026_S5_S6.pdf` (planches 2 a 16).

Chaque affichage, chaque « OK » et chaque « Erreur » a ete confirme en
compilant le programme de la planche (clang++ 21, C++17). La compilation
etablit deux ecarts :

* 2023-2024, planche 22 : `Pixel::Pixel(int abs = 1, int ord = 2, int cl = 3)`
  donne ses valeurs par defaut dans la definition hors de la classe, ce
  qui ferait du constructeur un constructeur par defaut apres coup --
  clang le refuse ;
* planche 16 (et 6 de 2025-2026) : `EleveAssistant` herite de `Etudiant`
  et de `Prof`, qui heritent tous deux de `Personne`. Tout acces a `id`
  y est ambigu -- l'objet porte deux `Personne` -- et la planche ne le
  dit pas.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.encapsulation import (
    PRIVE,
    PROTEGE,
    PUBLIC,
    Classe,
)
from polytech.algorithmique.domain.heritage import Declaration, Hierarchie

BASE = Declaration(membres={"pub1": PUBLIC, "prot1": PROTEGE, "priv1": PRIVE})
ACCES = Hierarchie(
    {"Base": BASE, "Deriv": Declaration(meres=("Base",), membres={"pub2": PUBLIC})}
)
POINTS = Hierarchie(
    {
        "point": Declaration(
            membres={"x": PRIVE, "y": PRIVE, "initialise": PUBLIC, "affiche": PUBLIC}
        ),
        "Pixel": Declaration(
            meres=("point",), membres={"couleur": PRIVE, "colore": PUBLIC}
        ),
    }
)
MASQUAGE = Hierarchie(
    {
        "X": Declaration(membres={"f1": PUBLIC, "f2": PUBLIC, "xxx": PROTEGE}),
        "Y": Declaration(meres=("X",), membres={"f2": PUBLIC, "f3": PUBLIC}),
    }
)
PERSONNES = Hierarchie(
    {
        "Personne": Declaration(membres={"id": PUBLIC}),
        "Etudiant": Declaration(meres=("Personne",)),
        "Prof": Declaration(meres=("Personne",)),
        "EleveAssistant": Declaration(meres=("Etudiant", "Prof")),
    }
)


def _formes(*virtuelles: str) -> Hierarchie:
    affiche = {"affiche": PUBLIC}
    return Hierarchie(
        {
            "Base": Declaration(membres=affiche, virtuelles=frozenset(virtuelles)),
            "B1": Declaration(meres=("Base",), membres=affiche),
            "B2": Declaration(meres=("Base",), membres=affiche),
        }
    )


def test_la_classe_derivee_herite_des_membres_de_sa_base() -> None:
    """Planche 14 : `PC.initialise(10, 20)` et `PC.affiche("Pixel")` sur un Pixel."""
    assert POINTS.definit("Pixel", "initialise") == "point"
    assert POINTS.definit("Pixel", "colore") == "Pixel"
    assert POINTS.accede("Pixel", "affiche", "main")


def test_la_base_n_a_pas_les_membres_de_sa_derivee() -> None:
    """Planche 18 : `X.pub2()` -- la classe Base n'a pas de membre pub2."""
    with pytest.raises(AttributeError, match="n'a pas de membre"):
        ACCES.definit("Base", "pub2")


def test_les_trois_appels_de_la_planche_17() -> None:
    """`X.pub1()` passe ; `X.priv1()` et `X.prot1()` sont inaccessibles."""
    constats = [ACCES.accede("Base", m, "main") for m in ("pub1", "priv1", "prot1")]
    assert constats == [True, False, False]


def test_la_derivee_lit_le_protege_mais_pas_le_prive_de_sa_base() -> None:
    """Planche 18, dans `Deriv::pub2` : `pub1()` et `prot1()` OK, `priv1()` erreur."""
    constats = [ACCES.accede("Deriv", m, "Deriv") for m in ("pub1", "prot1", "priv1")]
    assert constats == [True, True, False]


def test_le_code_exterieur_ne_voit_que_le_public_de_la_derivee() -> None:
    """Planche 18 : `Y.pub1()`, `Y.pub2()` OK ; `Y.priv1()`, `Y.prot1()` erreur."""
    membres = ("pub1", "pub2", "priv1", "prot1")
    constats = [ACCES.accede("Deriv", m, "main") for m in membres]
    assert constats == [True, True, False, False]


def test_les_neuf_lignes_de_la_planche_24() -> None:
    """(1) OUI (2) NON (3) NON depuis `main` ; (4) OUI (5) OUI (6) NON dans `derive`."""
    hierarchie = Hierarchie(
        {
            "base": Declaration(
                membres={"publ": PUBLIC, "prot": PROTEGE, "priv": PRIVE}
            ),
            "derive": Declaration(meres=("base",)),
        }
    )
    champs = ("publ", "prot", "priv")
    assert [hierarchie.accede("derive", c, "main") for c in champs] == [
        True,
        False,
        False,
    ]
    assert [hierarchie.accede("derive", c, "derive") for c in champs] == [
        True,
        True,
        False,
    ]


def test_une_classe_etrangere_ne_lit_pas_le_protege() -> None:
    """Le protege s'ouvre a la descendance, pas a une classe sans lien."""
    hierarchie = Hierarchie({"Base": BASE, "Autre": Declaration()})
    assert not hierarchie.accede("Base", "prot1", "Autre")
    assert hierarchie.accede("Base", "priv1", "Base")


def test_la_redefinition_masque_la_fonction_de_la_base() -> None:
    """Planche 20 : dans `Y::f3`, `f1()` est celle de X et `f2()` celle de Y."""
    assert MASQUAGE.definit("Y", "f1") == "X"
    assert MASQUAGE.definit("Y", "f2") == "Y"
    assert MASQUAGE.definit("X", "f2") == "X"


def test_la_derivee_lit_le_membre_protege_de_la_planche_20() -> None:
    """`xxx = 14` dans `Y::f3` : « In F3 xxx =14 »."""
    assert MASQUAGE.accede("Y", "xxx", "Y")
    assert not MASQUAGE.accede("Y", "xxx", "main")


def test_le_constructeur_de_la_base_passe_le_premier() -> None:
    """Planche 21 : `Pixel a;` affiche « constr. point » puis « constr. Pixel »."""
    assert POINTS.ordre_de_construction("Pixel") == ("point", "Pixel")
    assert POINTS.ordre_de_construction("point") == ("point",)


def test_la_destruction_suit_l_ordre_inverse() -> None:
    """Planche 23 : base, derivee, puis destruction de la derivee, de la base."""
    assert POINTS.ordre_de_destruction("Pixel") == ("Pixel", "point")


def test_un_objet_membre_est_construit_apres_la_base() -> None:
    """Planche 23, `base b;` dans `derive` : deux bases, puis la derivee."""
    hierarchie = Hierarchie(
        {
            "base": Declaration(),
            "derive": Declaration(meres=("base",), objets_membres=("base",)),
        }
    )
    assert hierarchie.ordre_de_construction("derive") == ("base", "base", "derive")
    assert hierarchie.ordre_de_destruction("derive") == ("derive", "base", "base")


def test_les_bases_sont_construites_dans_l_ordre_de_leur_declaration() -> None:
    """Planche 13 : `class C : public A, public B` -- A, B, puis C."""
    hierarchie = Hierarchie(
        {
            "A": Declaration(),
            "B": Declaration(),
            "C": Declaration(meres=("A", "B")),
        }
    )
    assert hierarchie.ordre_de_construction("C") == ("A", "B", "C")


def test_l_eleve_assistant_porte_deux_personnes() -> None:
    """Planche 16 : heritage multiple, `Personne` construite deux fois."""
    assert PERSONNES.ordre_de_construction("EleveAssistant") == (
        "Personne",
        "Etudiant",
        "Personne",
        "Prof",
        "EleveAssistant",
    )


def test_l_acces_a_un_membre_herite_par_deux_chemins_est_ambigu() -> None:
    """Ecart de la planche 16 : `e.id` ne compile pas."""
    with pytest.raises(TypeError, match="membre ambigu"):
        PERSONNES.definit("EleveAssistant", "id")
    assert PERSONNES.definit("Etudiant", "id") == "Personne"


def test_sans_virtual_le_type_du_pointeur_decide() -> None:
    """Planche 27, colonne « Sans virtual » : Base, Base, Base."""
    formes = _formes()
    executees = [formes.methode_executee("Base", t, "affiche") for t in formes.classes]
    assert executees == ["Base", "Base", "Base"]


def test_avec_virtual_le_type_de_l_objet_decide() -> None:
    """Planche 27, colonne « Avec virtual » : Base, B1 : Derive1, B2 : Derive2."""
    formes = _formes("affiche")
    executees = [formes.methode_executee("Base", t, "affiche") for t in formes.classes]
    assert executees == ["Base", "B1", "B2"]


def test_l_appel_interne_de_la_planche_26_suit_la_meme_regle() -> None:
    """`Mere::Affiche` appelle `GetNom` : celle de la fille seulement si virtuelle."""
    for virtuelles, attendue in (
        (frozenset({"GetNom"}), "Fille"),
        (frozenset(), "Mere"),
    ):
        hierarchie = Hierarchie(
            {
                "Mere": Declaration(
                    membres={"GetNom": PUBLIC, "Affiche": PUBLIC},
                    virtuelles=virtuelles,
                ),
                "Fille": Declaration(meres=("Mere",), membres={"GetNom": PUBLIC}),
            }
        )
        assert hierarchie.methode_executee("Mere", "Fille", "GetNom") == attendue
        assert hierarchie.methode_executee("Mere", "Mere", "GetNom") == "Mere"


def test_une_fonction_virtuelle_non_redefinie_reste_celle_de_la_base() -> None:
    """La fille herite de la fonction virtuelle telle quelle."""
    hierarchie = Hierarchie(
        {
            "Mere": Declaration(
                membres={"GetNom": PUBLIC}, virtuelles=frozenset({"GetNom"})
            ),
            "Fille": Declaration(meres=("Mere",)),
        }
    )
    assert hierarchie.methode_executee("Mere", "Fille", "GetNom") == "Mere"


def test_un_pointeur_de_base_ne_designe_qu_un_objet_de_sa_descendance() -> None:
    """`Base *p = &objet` exige que l'objet « soit une » Base."""
    formes = _formes("affiche")
    with pytest.raises(TypeError, match="n'est pas une classe derivee de"):
        formes.methode_executee("B1", "B2", "affiche")
    with pytest.raises(TypeError, match="n'est pas une classe derivee de"):
        formes.methode_executee("B1", "Base", "affiche")


def test_une_classe_de_base_inconnue_est_refusee() -> None:
    """`class Etudiant : public Personne` sans `Personne`."""
    with pytest.raises(ValueError, match="classe de base inconnue"):
        Hierarchie({"Etudiant": Declaration(meres=("Personne",))})


def test_un_objet_membre_d_un_type_inconnu_est_refuse() -> None:
    """Le type d'un membre doit etre declare lui aussi."""
    with pytest.raises(ValueError, match="classe de base inconnue"):
        Hierarchie({"derive": Declaration(objets_membres=("base",))})


def test_une_classe_inconnue_n_a_pas_de_membres() -> None:
    """Interroger une classe absente de la hierarchie est une erreur."""
    with pytest.raises(KeyError, match="classe inconnue"):
        ACCES.definit("Autre", "pub1")


TP6 = Hierarchie(
    {
        "Personne": Declaration(
            membres={"Nom": PROTEGE, "saisie": PUBLIC, "affiche": PUBLIC},
            virtuelles=frozenset({"saisie"}),
        ),
        "PROF": Declaration(
            meres=("Personne",), membres={"saisie": PUBLIC, "affiche": PUBLIC}
        ),
        "Direction": Declaration(
            meres=("PROF",), membres={"saisie": PUBLIC, "affiche": PUBLIC}
        ),
    }
)


def test_la_base_passe_avant_l_objet_membre_d_un_autre_type() -> None:
    """Compile : `struct D : A { M m; }` affiche A, M, D puis ~D, ~M, ~A."""
    hierarchie = Hierarchie(
        {
            "A": Declaration(),
            "M": Declaration(),
            "D": Declaration(meres=("A",), objets_membres=("M",)),
        }
    )
    assert hierarchie.ordre_de_construction("D") == ("A", "M", "D")
    assert hierarchie.ordre_de_destruction("D") == ("D", "M", "A")


def test_le_protege_s_ouvre_a_toute_la_descendance_du_tp_6() -> None:
    """Planche 30 : Personne, PROF, Direction -- `Nom` protege lu depuis Direction."""
    assert TP6.accede("Direction", "Nom", "Direction")
    assert TP6.accede("Direction", "Nom", "PROF")
    assert not TP6.accede("Direction", "Nom", "main")


def test_une_fonction_reste_virtuelle_dans_toute_la_descendance() -> None:
    """`PROF *p = &direction; p->saisie();` execute `Direction::saisie`."""
    assert TP6.methode_executee("PROF", "Direction", "saisie") == "Direction"
    assert TP6.methode_executee("Personne", "Direction", "saisie") == "Direction"


def test_une_fonction_non_virtuelle_suit_le_pointeur_a_chaque_etage() -> None:
    """`p->affiche()` rend celle de PROF, `q->affiche()` celle de Personne."""
    assert TP6.methode_executee("PROF", "Direction", "affiche") == "PROF"
    assert TP6.methode_executee("Personne", "Direction", "affiche") == "Personne"


def test_la_hierarchie_du_tp_6_de_2025_2026() -> None:
    """Planche 67 : Personne, d'ou Etudiant et Prof ; Direction derive de Prof."""
    hierarchie = Hierarchie(
        {
            "Personne": Declaration(
                membres={"Id": PRIVE, "Nom": PRIVE, "saisie": PUBLIC}
            ),
            "Etudiant": Declaration(
                meres=("Personne",), membres={"Note": PRIVE, "getNote": PUBLIC}
            ),
            "Prof": Declaration(meres=("Personne",), membres={"Service": PRIVE}),
            "Direction": Declaration(meres=("Prof",), membres={"Titre": PRIVE}),
        }
    )
    assert hierarchie.ordre_de_construction("Direction") == (
        "Personne",
        "Prof",
        "Direction",
    )
    assert hierarchie.definit("Direction", "saisie") == "Personne"
    assert not hierarchie.accede("Etudiant", "Nom", "Etudiant")
    assert not hierarchie.accede("Etudiant", "Note", "Direction")
    assert hierarchie.accede("Etudiant", "getNote", "Direction")


def test_le_doyen_du_tp_6_lit_les_notes_par_amitie() -> None:
    """Planche 18 : `friend B; // class Direction` dans la classe Etudiant."""
    etudiant = Classe("Etudiant", {"Note": PRIVE}, amies=frozenset({"Direction"}))
    assert etudiant.accede("Note", "Direction")
    assert not etudiant.accede("Note", "Prof")
