"""La programmation lineaire, ancree sur l'exemple resolu du cours.

Le fascicule `Exs_Exemples_resolus.pdf` porte un programme lisible --
`max 6 x1 + 5 x2` sous `x1 + x2 <= 8`, `-2 x1 + 3 x2 <= 6`,
`x1 - x2 <= 2` -- et son tableau simplexe affiche `z = -45`, c'est-a-dire
un optimum de `45` pour la maximisation.

Quatre chemins y menent dans ce fichier, et aucun ne partage de formule
avec les autres :

* l'enumeration des solutions de base realisables, qui est la
  « caracterisation algebrique des sommets » du chapitre I ;
* le simplexe primal du chapitre II ;
* le **dual**, resolu par le dual simplexe du chapitre V, dont la valeur
  doit egaler celle du primal -- c'est la dualite forte ;
* la separation et evaluation du chapitre XIV, qui retrouve le meme point
  puisqu'il est deja entier.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.dual_simplexe import DualSimplexe
from polytech.optimisation.domain.dualite import PaireDuale, dual_de
from polytech.optimisation.domain.extensions import (
    VariableBornee,
    facteur_elementaire,
    inverse_de_base,
)
from polytech.optimisation.domain.forme_standard import ProgrammeLineaire
from polytech.optimisation.domain.simplexe import Simplexe
from polytech.optimisation.domain.variables_entieres import (
    premiere_fractionnaire,
    resoudre_en_entiers,
    separer,
)

DU_COURS = ProgrammeLineaire(
    objectif=sp.Matrix([6, 5]),
    contraintes=sp.Matrix([[1, 1], [-2, 3], [1, -1]]),
    seconds_membres=sp.Matrix([8, 6, 2]),
)
OPTIMUM = sp.Integer(45)
SOMMET = sp.Matrix([5, 3])


def _dual_resolu() -> sp.Matrix:
    """Le dual du programme du cours, resolu par le dual simplexe."""
    dual = dual_de(DU_COURS)
    base = tuple(dual.variables + rang for rang in range(dual.equations))
    tableau = dual.matrice_etendue().row_join(dual.seconds_membres)
    depart = DualSimplexe(programme=dual, base=base, tableau=tableau)
    return depart.retablir_la_faisabilite().point()


# --- Chapitre I : forme standard et sommets --------------------------------


def test_la_forme_standard_ajoute_une_variable_d_ecart_par_contrainte() -> None:
    etendue = DU_COURS.matrice_etendue()
    assert etendue.shape == (3, 5)
    assert etendue[:, 2:] == sp.eye(3)


def test_les_sommets_du_domaine_sont_ceux_de_la_figure() -> None:
    """Cinq sommets, dont l'origine et l'optimum."""
    sommets = {tuple(s) for s in DU_COURS.sommets()}
    attendus = {
        (5, 3),
        (sp.Rational(18, 5), sp.Rational(22, 5)),
        (2, 0),
        (0, 2),
        (0, 0),
    }
    assert sommets == attendus


def test_chaque_sommet_est_realisable() -> None:
    """Un sommet qui violerait une contrainte ne serait pas un sommet de D."""
    assert all(DU_COURS.est_realisable(s) for s in DU_COURS.sommets())


def test_l_enumeration_des_sommets_donne_l_optimum_du_cours() -> None:
    """Premier chemin : le chapitre I, sans aucun pivot."""
    point, valeur = DU_COURS.optimum_par_enumeration()
    assert point == SOMMET
    assert valeur == OPTIMUM


def test_un_objectif_de_mauvaise_dimension_est_refuse() -> None:
    with pytest.raises(ValueError, match="objectif de dimension"):
        ProgrammeLineaire(
            objectif=sp.Matrix([1, 2, 3]),
            contraintes=sp.Matrix([[1, 1]]),
            seconds_membres=sp.Matrix([1]),
        )


def test_un_second_membre_de_mauvaise_dimension_est_refuse() -> None:
    with pytest.raises(ValueError, match="second membre de dimension"):
        ProgrammeLineaire(
            objectif=sp.Matrix([1, 2]),
            contraintes=sp.Matrix([[1, 1]]),
            seconds_membres=sp.Matrix([1, 2]),
        )


def test_un_domaine_vide_est_nomme() -> None:
    """Aucune solution de base realisable : le dire plutot que rendre n'importe quoi."""
    impossible = ProgrammeLineaire(
        objectif=sp.Matrix([1]),
        contraintes=sp.Matrix([[1], [-1]]),
        seconds_membres=sp.Matrix([1, -2]),
    )
    with pytest.raises(ValueError, match="domaine realisable vide"):
        impossible.optimum_par_enumeration()


# --- Chapitre II : le simplexe ---------------------------------------------


def test_le_simplexe_retrouve_l_optimum_du_cours() -> None:
    """Deuxieme chemin : le pivotage du chapitre II."""
    point, valeur = Simplexe(programme=DU_COURS).optimum()
    assert point == SOMMET
    assert valeur == OPTIMUM


def test_le_simplexe_expose_chacune_de_ses_iterations() -> None:
    """Un corrige pas a pas doit montrer chaque tableau, pas le seul dernier."""
    iterations = Simplexe(programme=DU_COURS).iterations()
    minimum = 2
    assert len(iterations) >= minimum
    assert iterations[0].valeur == 0
    assert iterations[-1].valeur == OPTIMUM


def test_la_valeur_croit_a_chaque_iteration() -> None:
    """C'est la propriete qui garantit la terminaison sans cyclage."""
    valeurs = [i.valeur for i in Simplexe(programme=DU_COURS).iterations()]
    assert valeurs == sorted(valeurs)


def test_un_programme_non_borne_est_nomme() -> None:
    """Rendre un optimum sur un domaine non borne serait faux en silence."""
    non_borne = ProgrammeLineaire(
        objectif=sp.Matrix([1]),
        contraintes=sp.Matrix([[-1]]),
        seconds_membres=sp.Matrix([0]),
    )
    with pytest.raises(ValueError, match="non borne"):
        Simplexe(programme=non_borne).optimum()


def test_le_simplexe_primal_refuse_un_second_membre_negatif() -> None:
    """Sa base de depart ne serait pas realisable : c'est le dual qu'il faut."""
    with pytest.raises(ValueError, match="employer le dual simplexe"):
        Simplexe(programme=dual_de(DU_COURS)).optimum()


# --- Chapitre IV et V : dualite et dual simplexe ---------------------------


def test_le_dual_a_autant_de_variables_que_le_primal_de_contraintes() -> None:
    dual = dual_de(DU_COURS)
    assert dual.variables == DU_COURS.equations
    assert dual.equations == DU_COURS.variables


def test_la_dualite_forte_est_verifiee_sur_l_exemple_du_cours() -> None:
    """Troisieme chemin : `b.y` doit valoir exactement `c.x`."""
    point_dual = _dual_resolu()
    valeur_duale = (DU_COURS.seconds_membres.T * point_dual)[0, 0]
    assert sp.simplify(valeur_duale - OPTIMUM) == 0


def test_l_ecart_de_dualite_est_nul_a_l_optimum() -> None:
    paire = PaireDuale(
        primal=DU_COURS,
        point_primal=SOMMET,
        point_dual=_dual_resolu(),
    )
    assert paire.verifie_la_dualite_faible()
    assert sp.simplify(paire.ecart_de_dualite()) == 0


def test_les_ecarts_complementaires_sont_satisfaits() -> None:
    """Pour chaque contrainte : ou l'ecart primal est nul, ou la duale l'est."""
    paire = PaireDuale(
        primal=DU_COURS,
        point_primal=SOMMET,
        point_dual=_dual_resolu(),
    )
    assert paire.verifie_les_ecarts_complementaires()


def test_la_dualite_faible_encadre_tout_couple_realisable() -> None:
    """Et pas seulement l'optimum : c'est ce qui en fait une borne utile."""
    paire = PaireDuale(
        primal=DU_COURS,
        point_primal=sp.Matrix([2, 0]),
        point_dual=_dual_resolu(),
    )
    assert paire.verifie_la_dualite_faible()
    assert paire.ecart_de_dualite() > 0


def test_le_dual_simplexe_refuse_de_pivoter_une_base_deja_realisable() -> None:
    """Il n'a alors rien a retablir, et le dire vaut mieux qu'un pivot gratuit."""
    base = tuple(DU_COURS.variables + r for r in range(DU_COURS.equations))
    tableau = DU_COURS.matrice_etendue().row_join(DU_COURS.seconds_membres)
    depart = DualSimplexe(programme=DU_COURS, base=base, tableau=tableau)
    assert depart.ligne_sortante() is None
    with pytest.raises(ValueError, match="deja primalement realisable"):
        depart.pivoter()


# --- Chapitre XIV : variables entieres -------------------------------------


def test_la_premiere_composante_fractionnaire_est_reperee() -> None:
    assert premiere_fractionnaire(sp.Matrix([2, sp.Rational(3, 2)])) == 1
    assert premiere_fractionnaire(sp.Matrix([2, 3])) is None


def test_la_separation_ajoute_une_contrainte_et_une_seule() -> None:
    separe = separer(DU_COURS, variable=0, seuil=3, vers_le_haut=False)
    assert separe.equations == DU_COURS.equations + 1
    assert separe.variables == DU_COURS.variables


def test_une_relaxation_deja_entiere_est_sondee_sans_brancher() -> None:
    """Quatrieme chemin : l'optimum du cours est entier, donc un seul noeud."""
    resolution = resoudre_en_entiers(DU_COURS)
    assert resolution.point == SOMMET
    assert resolution.valeur == OPTIMUM
    assert resolution.noeuds == 1


def test_une_relaxation_fractionnaire_oblige_a_brancher() -> None:
    """`max x1 + x2` sous `2 x1 + 2 x2 <= 3` : relaxation 3/2, optimum entier 1."""
    fractionnaire = ProgrammeLineaire(
        objectif=sp.Matrix([1, 1]),
        contraintes=sp.Matrix([[2, 2]]),
        seconds_membres=sp.Matrix([3]),
    )
    relaxe = Simplexe(programme=fractionnaire).optimum()[1]
    assert relaxe == sp.Rational(3, 2)
    resolution = resoudre_en_entiers(fractionnaire)
    assert resolution.valeur == 1
    assert resolution.noeuds > 1


# --- Extensions : inverse de base et variables bornees ---------------------


def test_l_inverse_de_base_est_bien_un_inverse() -> None:
    etendue = DU_COURS.matrice_etendue()
    base = (0, 1, 4)
    produit = etendue[:, list(base)] * inverse_de_base(etendue, base)
    assert sp.simplify(produit) == sp.eye(3)


def test_une_base_singuliere_est_refusee() -> None:
    """Ses colonnes sont dependantes : il n'y a pas de solution de base."""
    etendue = sp.Matrix([[1, 2], [2, 4]])
    with pytest.raises(ValueError, match="base singuliere"):
        inverse_de_base(etendue, (0, 1))


def test_le_facteur_elementaire_met_l_inverse_a_jour_sans_l_inverser() -> None:
    """C'est la forme produit des planches : un pivot, une colonne.

    Verification : le facteur applique a l'inverse de l'ancienne base doit
    donner l'inverse de la nouvelle.
    """
    etendue = DU_COURS.matrice_etendue()
    ancienne = (2, 3, 4)
    inverse = inverse_de_base(etendue, ancienne)
    entrante, rang = 0, 0
    colonne = inverse * etendue[:, entrante]
    nouvelle = (entrante, 3, 4)
    attendu = inverse_de_base(etendue, nouvelle)
    assert sp.simplify(facteur_elementaire(colonne, rang) * inverse - attendu) == (
        sp.zeros(3, 3)
    )


def test_un_pivot_nul_interdit_le_facteur_elementaire() -> None:
    with pytest.raises(ValueError, match="pivot nul"):
        facteur_elementaire(sp.Matrix([0, 1, 2]), 0)


def test_une_variable_bornee_se_translate_et_se_retablit() -> None:
    """L'aller-retour doit etre exact, sinon la solution rendue est fausse."""
    bornee = VariableBornee(borne_basse=sp.Integer(2), borne_haute=sp.Integer(7))
    assert bornee.amplitude == 5
    assert bornee.translater(sp.Integer(5)) == 3
    assert bornee.retablir(bornee.translater(sp.Integer(5))) == 5


def test_une_variable_bornee_reconnait_ce_qui_sort_de_ses_bornes() -> None:
    bornee = VariableBornee(borne_basse=sp.Integer(2), borne_haute=sp.Integer(7))
    assert bornee.contient(sp.Integer(2))
    assert bornee.contient(sp.Integer(7))
    assert not bornee.contient(sp.Integer(8))


def test_un_intervalle_vide_est_refuse() -> None:
    with pytest.raises(ValueError, match="intervalle vide"):
        VariableBornee(borne_basse=sp.Integer(7), borne_haute=sp.Integer(2))
