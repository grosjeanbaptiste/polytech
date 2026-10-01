"""L'algebre lineaire du cours, exposee par ses gestes.

Les enonces du fascicule `Exs_01_Integration_matrices_determinants.pdf`
sont des **images** : ni les matrices ni les systemes n'en sortent. Rien
n'est donc ancre sur eux -- et c'est tant mieux, parce que l'ancrage
retenu est plus fort qu'une concordance de chiffres.

Chaque resultat se verifie **intrinsequement** : le developpement par
cofacteurs doit retomber sur le determinant, `A * adj(A)` doit valoir
`det(A) * I`, une solution de systeme doit satisfaire le systeme, et un
vecteur propre doit verifier `A v = lambda v`. Ces controles ne supposent
aucune table et aucune lecture : ils se suffisent.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.spectre import Spectre
from polytech.maths.domain.algebre.systemes import SystemeLineaire

A = sp.Symbol("a", real=True)
CIRCULANTE = sp.Matrix([[0, A, 1], [1, 0, A], [A, 1, 0]])
SYMETRIQUE = sp.Matrix([[2, 1], [1, 2]])


def test_le_developpement_par_cofacteurs_retombe_sur_le_determinant() -> None:
    """Le geste du cours, verifie par le resultat que sympy calcule autrement."""
    determinant = Determinant(matrice=CIRCULANTE)
    assert sp.simplify(determinant.par_cofacteurs(ligne=0) - CIRCULANTE.det()) == 0


def test_le_developpement_nomme_chaque_cofacteur() -> None:
    """Un corrige pas a pas montre les mineurs, pas seulement leur somme."""
    determinant = Determinant(matrice=CIRCULANTE)
    termes = determinant.termes_du_developpement(ligne=0)
    assert len(termes) == CIRCULANTE.cols
    assert sp.simplify(sum(termes) - CIRCULANTE.det()) == 0


def test_la_matrice_adjointe_verifie_l_identite_fondamentale() -> None:
    """`A adj(A) = det(A) I` : l'inverse en decoule, et s'en verifie."""
    determinant = Determinant(matrice=CIRCULANTE)
    produit = sp.simplify(CIRCULANTE * determinant.adjointe())
    assert produit == sp.simplify(CIRCULANTE.det() * sp.eye(3))


def test_les_valeurs_qui_rendent_la_matrice_singuliere_sont_nommees() -> None:
    """C'est la question du fascicule : pour quels `a` est-elle inversible."""
    determinant = Determinant(matrice=CIRCULANTE)
    assert determinant.valeurs_singulieres(A) == (sp.Integer(-1),)


def test_une_ligne_hors_de_la_matrice_est_refusee() -> None:
    """Developper selon une ligne qui n'existe pas rendrait zero en silence."""
    with pytest.raises(ValueError, match="ligne hors de la matrice"):
        Determinant(matrice=SYMETRIQUE).par_cofacteurs(ligne=5)


def test_une_matrice_non_carree_n_a_pas_de_determinant() -> None:
    with pytest.raises(ValueError, match="matrice non carree"):
        Determinant(matrice=sp.Matrix([[1, 2, 3], [4, 5, 6]]))


def test_la_solution_d_un_systeme_satisfait_le_systeme() -> None:
    """Le controle intrinseque : substituer et verifier, sans table."""
    systeme = SystemeLineaire(
        matrice=sp.Matrix([[2, 1], [1, -1]]),
        second_membre=sp.Matrix([5, 1]),
    )
    assert systeme.solution_verifiee()


def test_un_systeme_homogene_a_solutions_non_nulles_quand_il_est_degenere() -> None:
    """La question 2.3 du fascicule : a quelle condition, et lesquelles."""
    systeme = SystemeLineaire(
        matrice=sp.Matrix([[1, 1], [A, A]]),
        second_membre=sp.Matrix([0, 0]),
    )
    assert systeme.admet_des_solutions_non_nulles()


def test_un_systeme_homogene_regulier_n_a_que_la_solution_nulle() -> None:
    """Reponse 2.2.1 du fascicule : (0, 0, 0)."""
    systeme = SystemeLineaire(
        matrice=sp.Matrix([[2, 1], [1, -1]]),
        second_membre=sp.Matrix([0, 0]),
    )
    assert not systeme.admet_des_solutions_non_nulles()


def test_un_second_membre_de_mauvaise_taille_est_refuse() -> None:
    """Il donnerait un systeme resolu sur des equations qui n'existent pas."""
    with pytest.raises(ValueError, match="second membre"):
        SystemeLineaire(
            matrice=sp.Matrix([[1, 1], [1, 2]]),
            second_membre=sp.Matrix([1, 2, 3]),
        )


def test_le_polynome_caracteristique_a_les_valeurs_propres_pour_racines() -> None:
    """Le geste du cours : `det(A - lambda I) = 0`, puis ses racines."""
    spectre = Spectre(matrice=SYMETRIQUE)
    polynome = spectre.polynome_caracteristique()
    for valeur in spectre.valeurs_propres():
        assert sp.simplify(polynome.subs(spectre.inconnue, valeur)) == 0


def test_chaque_vecteur_propre_verifie_sa_definition() -> None:
    """`A v = lambda v` : le controle intrinseque, sans aucune table."""
    spectre = Spectre(matrice=SYMETRIQUE)
    for valeur, vecteur in spectre.couples_propres():
        assert sp.simplify(SYMETRIQUE * vecteur - valeur * vecteur) == sp.zeros(2, 1)


def test_la_diagonalisation_reconstitue_la_matrice() -> None:
    """`P D P^-1 = A` : c'est la verification que le cours demande."""
    spectre = Spectre(matrice=SYMETRIQUE)
    passage, diagonale = spectre.diagonalisation()
    assert sp.simplify(passage * diagonale * passage.inv()) == SYMETRIQUE


def test_une_matrice_non_diagonalisable_est_nommee() -> None:
    """Un bloc de Jordan n'a pas assez de vecteurs propres : le dire.

    Sympy leverait de lui-meme, mais avec un message qui ne parle pas la
    langue du cours. La garde traduit, et surtout elle est testee.
    """
    with pytest.raises(ValueError, match="non diagonalisable"):
        Spectre(matrice=sp.Matrix([[1, 1], [0, 1]])).diagonalisation()
