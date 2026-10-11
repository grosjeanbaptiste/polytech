"""Mathematiques II, chapitre 9 d'algebre : valeurs propres, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels `cursus/notions/AlgebreLineaire2.toml`
rattache un titre portent ce que la planche enonce -- en recopiant ses
exemples, page citee.

Source : `AlgebreLineaire2/SLIDES/Slides_09_Valeurs_propres_et_vecteurs_propres.pdf`.
Les pages sont celles du PDF ; la planche imprimee est entre parentheses.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.diagonalisation import (
    est_diagonalisable,
    sous_espace_propre,
)
from polytech.maths.domain.algebre.jordan import forme_de_jordan, multiplicites
from polytech.maths.domain.algebre.reduction import (
    Nature,
    nature_de_la_reduction,
    reduire_au_maximum,
)
from polytech.maths.domain.algebre.spectre import Spectre, diagonalisee_par
from polytech.maths.domain.algebre.triangularisation import (
    completer_en_base,
    est_triangulaire_superieure,
    est_triangularisable,
    semblable_par,
    triangulariser,
)

DEUX = sp.Matrix([[2, 2], [1, 3]])
"""Pages 42, 75 et 76 (19, 37 et 38) : l'illustration de la diagonalisation."""

QUATRE = sp.Matrix([[2, 0, 1, 1], [-2, 2, 1, -1], [0, 0, 3, 1], [0, -2, -1, 3]])
"""Pages 101 a 106 (54 a 57) : l'illustration de la triangularisation."""

TROIS = sp.Matrix([[1, 1, 0], [-1, 2, 1], [1, 0, 1]])
"""Pages 108 et 109 (58 et 59) : l'illustration de la reduction."""


def test_le_polynome_caracteristique_et_le_spectre_de_la_page_44() -> None:
    """Pages 43 et 44 (20 et 21) : `t2 - tr(A) t + det(A) = (t - 1)(t - 4)`."""
    spectre = Spectre(DEUX)
    t = spectre.inconnue
    assert spectre.polynome_caracteristique() == t**2 - DEUX.trace() * t + DEUX.det()
    assert spectre.valeurs_propres() == (1, 4)
    assert [valeur for valeur, _ in spectre.couples_propres()] == [1, 4]


def test_les_sous_espaces_propres_et_la_diagonalisation_de_la_page_76() -> None:
    """Pages 75 et 76 (37 et 38) : `S(1)`, `S(4)`, puis `P^-1 A P = diag(1, 4)`."""
    assert sous_espace_propre(DEUX, 1) == ((-2, 1),)
    assert sous_espace_propre(DEUX, 4) == ((1, 1),)
    assert est_diagonalisable(DEUX)
    passage = sp.Matrix([[2, 1], [-1, 1]])
    assert diagonalisee_par(DEUX, passage) == sp.diag(1, 4)
    de_la_bibliotheque, diagonale = Spectre(DEUX).diagonalisation()
    assert semblable_par(DEUX, de_la_bibliotheque) == diagonale


def test_deux_matrices_semblables_ont_le_meme_polynome() -> None:
    """Page 62 (27) : `det(P^-1 A P - t I) = det(A - t I)`."""
    semblable = semblable_par(DEUX, sp.Matrix([[1, 1], [0, 1]]))
    assert semblable != DEUX
    avant, apres = Spectre(DEUX), Spectre(semblable)
    ecart = apres.polynome_caracteristique().subs(apres.inconnue, avant.inconnue)
    assert sp.expand(ecart - avant.polynome_caracteristique()) == 0


def test_les_multiplicites_de_la_page_101_imposent_la_triangularisation() -> None:
    """Pages 101 et 104 (54 et 55) : `(t - 2)^3 (t - 4)`, `m_g(2) = 1 < 3`."""
    assert multiplicites(QUATRE) == {2: (3, 1), 4: (1, 1)}
    assert not est_diagonalisable(QUATRE)
    assert est_triangularisable(QUATRE)
    assert nature_de_la_reduction(QUATRE) is Nature.TRIANGULARISABLE
    passage, triangulaire = triangulariser(QUATRE)
    assert est_triangulaire_superieure(triangulaire)
    assert semblable_par(QUATRE, passage) == triangulaire
    assert sorted(triangulaire.diagonal()) == [2, 2, 2, 4]


def test_la_reduction_de_la_page_109_laisse_un_bloc_sans_racine_reelle() -> None:
    """Pages 108 et 109 (58 et 59) : une seule racine reelle, 2 ; `P` de la planche."""
    assert nature_de_la_reduction(TROIS) is Nature.NI_L_UN_NI_L_AUTRE
    passage = completer_en_base([sp.Matrix([1, 1, 1])])
    assert passage.det() != 0
    de_la_planche = sp.Matrix([[1, 0, 0], [1, 1, 0], [1, 0, 1]])
    reduite = sp.Matrix([[2, 1, 0], [0, 1, 1], [0, -1, 1]])
    assert semblable_par(TROIS, de_la_planche) == reduite
    obtenue = reduire_au_maximum(TROIS).reduite
    assert obtenue[:, 0] == sp.Matrix([2, 0, 0])
    assert not any(valeur.is_real for valeur in obtenue[1:, 1:].eigenvals())


def test_la_forme_de_jordan_de_la_matrice_de_la_page_101() -> None:
    """Pages 128 et 129 (67 et 68) : un bloc de taille 3 pour 2, un de 1 pour 4."""
    passage, jordan = forme_de_jordan(QUATRE)
    bloc = sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
    assert jordan == sp.diag(bloc, 4)
    assert passage * jordan * passage.inv() == QUATRE


def test_la_troisieme_illustration_du_syllabus_se_diagonalise_dans_c() -> None:
    """Syllabus d'algebre, pages 71 et 72 : `P^-1 A P = diag(2, 1 + i, 1 - i)`.

    Regression : `diagonalisee_par` refusait cette matrice de passage
    complexe, faute de simplifier le produit avant de le juger diagonal.
    """
    assert set(Spectre(TROIS).valeurs_propres()) == {2, 1 + sp.I, 1 - sp.I}
    passage = sp.Matrix([[1, sp.I, -sp.I], [1, -1, -1], [1, 1, 1]])
    assert diagonalisee_par(TROIS, passage) == sp.diag(2, 1 + sp.I, 1 - sp.I)
