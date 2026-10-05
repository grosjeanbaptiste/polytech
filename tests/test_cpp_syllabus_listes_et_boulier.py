"""Syllabus « Introduction a la programmation en C++ » : liste chainee et boulier.

Source : `ProgrammationAlgorithmique/SYLLABUS/
Syllabus_Introduction_a_la_programmation_2015.pdf`, pages 98 a 101
(chapitre 9, « Les listes simplement chainees ») et 96 (exercice 8.7, « Le
boulier a quatre »).

L'execution de la page 101 et l'exemple de la page 96 sont recopies tels
quels.
"""

from __future__ import annotations

import pytest

from polytech.algorithmique.domain.boulier import CAPACITE, Boulier
from polytech.algorithmique.domain.liste_chainee import Cellule, ListeChainee


def _billes(boulier: Boulier) -> tuple[object, ...]:
    """Une lecture fraiche : mypy garde sinon le type etroit de l'assertion d'avant."""
    return boulier.billes


def _liste_de_la_page_101() -> ListeChainee:
    liste = ListeChainee()
    for nom in ("premier", "second", "nom3"):
        liste.inserer_en_tete(nom)
    return liste


def test_une_liste_neuve_est_vide() -> None:
    """Page 100 : `debut = NULL;` -- aucune cellule a parcourir."""
    liste = ListeChainee()
    assert liste.parcours() == ()
    assert liste.debut is None


def test_l_insertion_en_tete_inverse_l_ordre_de_saisie() -> None:
    """Page 101 : premier, second, nom3 saisis ; nom3, second, premier affiches."""
    assert _liste_de_la_page_101().parcours() == ("nom3", "second", "premier")


def test_chaque_cellule_designe_la_suivante_et_la_derniere_rien() -> None:
    """Page 99 : un contenu, un pointeur vers la suivante ; `NULL` au bout."""
    cellule = _liste_de_la_page_101().debut
    assert isinstance(cellule, Cellule)
    assert cellule.suivante is not None
    assert (cellule.contenu, cellule.suivante.contenu) == ("nom3", "second")
    derniere = cellule.suivante.suivante
    assert derniere is not None
    assert derniere.suivante is None


def test_l_insertion_apres_la_cellule_de_debut() -> None:
    """Page 101 : nom3, Nouveau, second, premier."""
    liste = _liste_de_la_page_101()
    liste.inserer_apres_la_tete("Nouveau")
    assert liste.parcours() == ("nom3", "Nouveau", "second", "premier")


def test_la_suppression_de_la_premiere_cellule() -> None:
    """Page 101 : Nouveau, second, premier."""
    liste = _liste_de_la_page_101()
    liste.inserer_apres_la_tete("Nouveau")
    assert liste.supprimer_la_tete() == "nom3"
    assert liste.parcours() == ("Nouveau", "second", "premier")


def test_la_liberation_rend_toutes_les_cellules() -> None:
    """Page 100 : `liberation` detruit chaque cellule, un `delete` par `new`."""
    liste = _liste_de_la_page_101()
    liste.inserer_apres_la_tete("Nouveau")
    assert liste.liberer() == 4
    assert liste.parcours() == ()
    assert liste.liberer() == 0


def test_une_liste_vide_n_a_ni_tete_a_supprimer_ni_tete_a_suivre() -> None:
    """`debut->next` quand `debut` vaut `NULL` : une erreur a l'execution."""
    liste = ListeChainee()
    with pytest.raises(LookupError, match="liste vide"):
        liste.supprimer_la_tete()
    with pytest.raises(LookupError, match="liste vide"):
        liste.inserer_apres_la_tete("Nouveau")


def test_le_boulier_de_la_page_96_pas_a_pas() -> None:
    """Rouge et Vert ; Bleu a gauche ; Blanc a droite ; Jaune puis Noire a gauche."""
    boulier = Boulier(["Rouge", "Vert bouteille"])
    boulier.ajouter_a_gauche("Bleu")
    assert boulier.billes == ("Bleu", "Rouge", "Vert bouteille")
    boulier.ajouter_a_droite("Blanc")
    assert _billes(boulier) == ("Bleu", "Rouge", "Vert bouteille", "Blanc")
    assert boulier.ajouter_a_gauche("Jaune") == "Blanc"
    assert _billes(boulier) == ("Jaune", "Bleu", "Rouge", "Vert bouteille")
    assert boulier.ajouter_a_gauche("Noire") == "Vert bouteille"
    assert _billes(boulier) == ("Noire", "Jaune", "Bleu", "Rouge")


def test_une_bille_ajoutee_a_droite_chasse_celle_de_gauche() -> None:
    """Page 96 : ajoutee a droite, une bille chasse celle de gauche."""
    boulier = Boulier(["a", "b", "c", "d"])
    assert boulier.ajouter_a_droite("e") == "a"
    assert boulier.billes == ("b", "c", "d", "e")


def test_un_ajout_dans_un_boulier_non_plein_ne_chasse_rien() -> None:
    """Jusqu'a quatre billes, toutes restent."""
    boulier = Boulier(["a"])
    assert boulier.ajouter_a_droite("b") is None
    assert boulier.ajouter_a_gauche("c") is None
    assert boulier.billes == ("c", "a", "b")


def test_effacer_la_bille_jaune_de_la_page_96() -> None:
    """Noire, Jaune, Bleu, Rouge : sans Jaune, il reste Noire, Bleu, Rouge."""
    boulier = Boulier(["Noire", "Jaune", "Bleu", "Rouge"])
    boulier.effacer("Jaune")
    assert boulier.billes == ("Noire", "Bleu", "Rouge")


def test_effacer_une_bille_absente_est_signale() -> None:
    """Page 96 : une bille a supprimer qui n'existe pas doit etre signalee."""
    boulier = Boulier(["Noire"])
    with pytest.raises(LookupError, match="bille absente du boulier"):
        boulier.effacer("Jaune")
    assert boulier.billes == ("Noire",)


def test_le_boulier_n_accepte_pas_plus_de_quatre_billes_a_la_saisie() -> None:
    """Page 96 : « on demandera le nombre de billes (NI <= 4) »."""
    with pytest.raises(OverflowError, match="boulier trop charge"):
        Boulier(["a", "b", "c", "d", "e"])
    assert Boulier().billes == ()
    assert len(Boulier(range(CAPACITE)).billes) == CAPACITE == 4
