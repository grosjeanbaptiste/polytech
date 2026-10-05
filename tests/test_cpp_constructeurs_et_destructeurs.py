"""Programmation objet, seance 4 : constructeurs, destructeurs, duree de vie.

Sources, dans `ProgrammationAlgorithmique/SLIDES/` :
`Slides_2023_2024_S4_Surcharge_des_operateurs.pdf` (planches 40 a 64),
`Slides_2024_2025_S4_Methodologie_orientee_objet.pdf` (planches 8 a 25) et
`Slides_2025_2026_S4.pdf` (planches 9 a 22).

Chaque affichage a ete confirme en compilant le programme de la planche
(clang++ 21, C++17). La compilation etablit des ecarts :

* 2025-2026, planche 14 : la variable globale est declaree `K` et employee
  `k` -- le programme ne compile pas tel quel ;
* 2025-2026, planche 15 : `static i=1;` n'a pas de type, et l'affichage
  imprime tait une ligne sur deux -- `Affiche()` est appelee dans le
  constructeur *et* dans `main`, chaque identifiant sort deux fois ;
* 2025-2026, planche 16 : la troisieme colonne redonne les valeurs par
  defaut dans la definition (« redefinition of default argument ») ;
* 2025-2026, planche 18 : `Arbre filsG, filsD;` dans `Arbre` est un type
  incomplet, et `TabD = new Tab[dim]` range des `Tab` dans un `int*` ;
* 2023-2024, planche 60 : les deux dernieres annotations sont echangees --
  l'objet local `V` est detruit a la sortie de `main`, l'objet statique
  `B` apres. Les deux lignes affichees sont identiques, l'ordre ne se
  lit donc que dans le commentaire.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.constructeurs import (
    IMPLICITE,
    Constructeur,
    constructeur_appele,
    est_constructible,
    tableau_constructible,
)
from polytech.algorithmique.domain.duree_de_vie import (
    CONSTRUCTION,
    COPIE,
    DESTRUCTION,
    Execution,
)
from polytech.algorithmique.domain.portee import Portees

C, K, D = CONSTRUCTION, COPIE, DESTRUCTION
SANS_PARAMETRE = Constructeur(0)
DEUX_PARAMETRES = Constructeur(2)
DEUX_PAR_DEFAUT = Constructeur(2, par_defaut=2)


def test_un_objet_local_vit_jusqu_a_la_fin_de_main() -> None:
    """2023-2024, planche 57 : `+++ Const` puis, apres « Fin main », `--- Destr`."""
    execution = Execution()
    execution.ouvrir()
    execution.locale("V")
    assert execution.journal == [(C, "V")]
    execution.terminer()
    assert execution.journal == [(C, "V"), (D, "V")]


def test_la_globale_nait_avant_main_et_meurt_la_derniere() -> None:
    """Planche 58 : `Glob`, `V`, puis `Obj1`, `Obj2` detruits en ordre inverse."""
    execution = Execution()
    execution.globale("Glob")
    execution.ouvrir()
    execution.locale("V")
    execution.ouvrir()
    execution.locale("Obj1")
    execution.locale("Obj2")
    execution.fermer()
    execution.terminer()
    assert execution.journal == [
        (C, "Glob"),
        (C, "V"),
        (C, "Obj1"),
        (C, "Obj2"),
        (D, "Obj2"),
        (D, "Obj1"),
        (D, "V"),
        (D, "Glob"),
    ]


def test_un_objet_de_bloc_meurt_a_chaque_tour_de_boucle() -> None:
    """Planche 59 : `b(i, 2*i)` construit et detruit a chacun des deux tours."""
    execution = Execution()
    execution.globale("a(1,1)")
    execution.ouvrir()
    execution.ouvrir()
    execution.locale("b(10,10)")
    execution.locale("c(3,3)")
    execution.fermer()
    execution.locale("d(20,20)")
    for tour in (1, 2):
        execution.ouvrir()
        execution.locale(f"b({tour},{2 * tour})")
        execution.fermer()
    execution.terminer()
    assert execution.journal == [
        (C, "a(1,1)"),
        (C, "b(10,10)"),
        (C, "c(3,3)"),
        (D, "c(3,3)"),
        (D, "b(10,10)"),
        (C, "d(20,20)"),
        (C, "b(1,2)"),
        (D, "b(1,2)"),
        (C, "b(2,4)"),
        (D, "b(2,4)"),
        (D, "d(20,20)"),
        (D, "a(1,1)"),
    ]


def test_un_objet_statique_n_est_construit_qu_une_fois() -> None:
    """Planche 60 : `fonct3` appelee deux fois, un seul `+++ Const` pour `B`."""
    execution = Execution()
    execution.ouvrir()
    execution.locale("V")
    execution.ouvrir()
    execution.locale("S")
    execution.fermer()
    for _ in range(2):
        execution.ouvrir()
        execution.statique("B")
        execution.fermer()
    execution.terminer()
    assert execution.journal == [
        (C, "V"),
        (C, "S"),
        (D, "S"),
        (C, "B"),
        (D, "V"),
        (D, "B"),
    ]


def test_la_statique_survit_au_local_de_main() -> None:
    """Ecart de la planche 60 : `V` est detruit avant `B`, non apres."""
    execution = Execution()
    execution.ouvrir()
    execution.locale("V")
    execution.statique("B")
    execution.terminer()
    assert execution.journal[-2:] == [(D, "V"), (D, "B")]


def test_la_statique_meurt_avant_la_globale_construite_avant_elle() -> None:
    """Globales et statiques meurent dans l'ordre inverse de leur naissance."""
    execution = Execution()
    execution.globale("Glob")
    execution.ouvrir()
    execution.statique("B")
    execution.terminer()
    assert execution.journal == [(C, "Glob"), (C, "B"), (D, "B"), (D, "Glob")]


def test_le_passage_par_valeur_appelle_le_constructeur_de_copie() -> None:
    """Planche 61 (et 19 de 2025-2026) : `fonct1(V)` copie, puis detruit la copie."""
    execution = Execution()
    execution.globale("Glob")
    execution.ouvrir()
    execution.locale("V")
    execution.ouvrir()
    execution.locale("a", modele="V")
    execution.fermer()
    execution.terminer()
    assert execution.journal == [
        (C, "Glob"),
        (C, "V"),
        (K, "a"),
        (D, "a"),
        (D, "V"),
        (D, "Glob"),
    ]


def test_le_passage_par_reference_ne_copie_rien() -> None:
    """2025-2026, planche 20 : `f(v)` copie, `h(v)` non -- une seule copie."""
    execution = Execution()
    execution.globale("g")
    execution.ouvrir()
    execution.locale("v")
    execution.ouvrir()
    execution.locale("p", modele="v")
    execution.fermer()
    execution.ouvrir()
    execution.fermer()
    execution.terminer()
    assert [nom for evenement, nom in execution.journal if evenement == K] == ["p"]
    assert execution.journal[-3:] == [(D, "p"), (D, "v"), (D, "g")]


def test_la_copie_d_un_objet_inconnu_est_refusee() -> None:
    """Le modele doit exister au moment de la copie."""
    execution = Execution()
    execution.ouvrir()
    with pytest.raises(NameError, match="modele inconnu : "):
        execution.locale("a", modele="V")


def test_l_objet_dynamique_meurt_au_delete_et_non_a_la_fin_du_bloc() -> None:
    """Planche 63 : `adr = new demo(3, 3)` est detruit par `delete add` dans `fct`."""
    execution = Execution()
    execution.ouvrir()
    execution.locale("a")
    execution.locale("b")
    execution.locale("c", modele="a")
    execution.nouveau("adr")
    execution.ouvrir()
    execution.locale("d", modele="a")
    execution.supprimer("adr")
    execution.fermer()
    execution.terminer()
    assert execution.journal == [
        (C, "a"),
        (C, "b"),
        (K, "c"),
        (C, "adr"),
        (K, "d"),
        (D, "adr"),
        (D, "d"),
        (D, "c"),
        (D, "b"),
        (D, "a"),
    ]
    assert execution.fuites == ()


def test_un_new_sans_delete_n_est_jamais_detruit() -> None:
    """Aucun destructeur n'est appele : c'est une fuite de memoire."""
    execution = Execution()
    execution.ouvrir()
    execution.nouveau("adr")
    execution.terminer()
    assert execution.journal == [(C, "adr")]
    assert execution.fuites == ("adr",)
    assert execution.vivants == 1


def test_un_delete_ne_se_fait_qu_une_fois() -> None:
    """Le second `delete` porte sur un objet qui n'est plus sur le tas."""
    execution = Execution()
    execution.ouvrir()
    execution.nouveau("adr")
    execution.supprimer("adr")
    with pytest.raises(RuntimeError, match="delete d'un objet absent du tas"):
        execution.supprimer("adr")


def test_un_tableau_construit_chaque_case_dans_l_ordre() -> None:
    """2025-2026, planche 13 : `Prof P[2]` appelle deux fois le constructeur."""
    execution = Execution()
    execution.ouvrir()
    execution.tableau("P", 2)
    execution.terminer()
    assert execution.journal == [(C, "P[0]"), (C, "P[1]"), (D, "P[1]"), (D, "P[0]")]


def test_le_compteur_d_objets_de_la_planche_14() -> None:
    """`k++` au constructeur, `k--` au destructeur : 1, 2, puis 1, 0."""
    execution = Execution()
    execution.ouvrir()
    releves = []
    for nom in ("a", "b"):
        execution.locale(nom)
        releves.append(execution.vivants)
    execution.fermer()
    assert releves == [1, 2]
    assert execution.vivants == 0


def test_une_globale_ne_se_declare_pas_dans_main() -> None:
    """Une fois un bloc ouvert, il est trop tard."""
    execution = Execution()
    execution.ouvrir()
    with pytest.raises(RuntimeError, match="une variable globale se construit avant"):
        execution.globale("Glob")


@pytest.mark.parametrize("geste", ["locale", "statique"])
def test_un_objet_local_demande_un_bloc(geste: str) -> None:
    """Hors de toute fonction, il n'y a que des globales."""
    with pytest.raises(RuntimeError, match="aucun bloc ouvert"):
        getattr(Execution(), geste)("V")


def test_on_ne_ferme_pas_un_bloc_qui_n_est_pas_ouvert() -> None:
    """Une accolade fermante de trop."""
    with pytest.raises(RuntimeError, match="aucun bloc ouvert"):
        Execution().fermer()


def test_le_constructeur_sans_parametre_de_la_planche_16() -> None:
    """Premiere colonne : `A Obj;` et `A Tab[3];` passent, `A Obj(3);` non."""
    seul = (SANS_PARAMETRE,)
    assert constructeur_appele(seul, 0) is SANS_PARAMETRE
    assert tableau_constructible(seul)
    assert not est_constructible(seul, 1)
    with pytest.raises(TypeError, match="aucun constructeur n'accepte"):
        constructeur_appele(seul, 1)


def test_le_constructeur_a_deux_parametres_de_la_planche_16() -> None:
    """Deuxieme colonne : `A Obj(3, 4);` passe, `A Obj;` et `A Tab[3];` non."""
    seul = (DEUX_PARAMETRES,)
    assert constructeur_appele(seul, 2) is DEUX_PARAMETRES
    assert not est_constructible(seul, 0)
    assert not tableau_constructible(seul)


def test_le_constructeur_a_valeurs_par_defaut_de_la_planche_16() -> None:
    """Troisieme colonne : zero, un ou deux arguments, et le tableau."""
    seul = (DEUX_PAR_DEFAUT,)
    assert [est_constructible(seul, n) for n in range(4)] == [True, True, True, False]
    assert tableau_constructible(seul)


def test_les_trois_structures_de_la_planche_20() -> None:
    """2024-2025 : `Etudiant()`, `Prof(int, string)`, `Dir(int = 5, string = ...)`."""
    declarations = {
        "Etudiant": (SANS_PARAMETRE,),
        "Prof": (DEUX_PARAMETRES,),
        "Dir": (DEUX_PAR_DEFAUT,),
    }
    constats = {
        nom: (
            est_constructible(liste, 0),
            est_constructible(liste, 1),
            est_constructible(liste, 2),
            tableau_constructible(liste),
        )
        for nom, liste in declarations.items()
    }
    assert constats["Etudiant"][::3] == (True, True)
    assert constats["Etudiant"][1] is False
    assert constats["Prof"] == (False, False, True, False)
    assert constats["Dir"] == (True, True, True, True)


def test_sans_constructeur_declare_le_compilateur_en_fournit_un() -> None:
    """2025-2026, planche 22 : un constructeur n'est pas obligatoire."""
    assert constructeur_appele((), 0) is IMPLICITE
    assert tableau_constructible(())
    assert not est_constructible((), 1)


def test_declarer_un_constructeur_retire_celui_du_compilateur() -> None:
    """2023-2024, planche 53 : `CRec Rc1(10, 20), Rc2;` -- aucun par defaut."""
    with pytest.raises(TypeError, match="aucun constructeur n'accepte"):
        constructeur_appele((DEUX_PARAMETRES,), 0)
    assert est_constructible((DEUX_PARAMETRES, SANS_PARAMETRE), 0)


def test_deux_constructeurs_par_defaut_rendent_l_appel_ambigu() -> None:
    """`A()` et `A(int X = 0, int Y = 5)` ensemble : `A Obj;` ne compile pas."""
    ensemble = (SANS_PARAMETRE, DEUX_PAR_DEFAUT)
    with pytest.raises(TypeError, match="appel ambigu : "):
        constructeur_appele(ensemble, 0)
    assert not est_constructible(ensemble, 0)
    assert constructeur_appele(ensemble, 1) is DEUX_PAR_DEFAUT


@pytest.mark.parametrize(("parametres", "par_defaut"), [(2, 3), (2, -1), (-1, 0)])
def test_un_constructeur_n_a_pas_plus_de_defauts_que_de_parametres(
    parametres: int, par_defaut: int
) -> None:
    """Ni un nombre negatif de l'un ou de l'autre."""
    with pytest.raises(ValueError, match="valeurs par defaut hors de 0"):
        Constructeur(parametres, par_defaut=par_defaut)


def test_l_identifiant_automatique_de_la_planche_44() -> None:
    """Lien avec `portee` : `static int i = 1;` numerote cinq objets de 1 a 5."""
    portees = Portees()
    identifiants = []
    for _ in range(5):
        portees.ouvrir()
        portees.declarer_statique("Prof", "i", 1)
        identifiants.append(portees.valeur("i"))
        portees.affecter("i", portees.valeur("i") + 1)
        portees.fermer()
    assert identifiants == [1, 2, 3, 4, 5]
