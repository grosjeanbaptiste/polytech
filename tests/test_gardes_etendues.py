"""Les gardes des modules ajoutes a la bibliotheque, declenchees une a une.

L'audit du depot exige qu'aucun `raise` ne reste jamais execute. Deux de
celles-ci ne se declenchent par aucune donnee : elles protegent d'un
amont -- sympy, ou un arbre de recherche qui n'aurait pas ete elague --
et le seul moyen de les eprouver est de faire mentir cet amont. C'est
assume, et dit.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.fluides.domain.fondamentales import navier_stokes
from polytech.optimisation.domain import variables_entieres
from polytech.optimisation.domain.forme_standard import ProgrammeLineaire
from polytech.thermique.domain.echangeurs import (
    Echangeur,
    efficacite_courants_croises_sans_melange,
)

ECHANGEUR = Echangeur(conductance=775.61)


def test_une_surface_nulle_interdit_le_coefficient_rapporte() -> None:
    """`K_gl = (K_gl S) / S` : diviser par zero rendrait un infini compose."""
    with pytest.raises(ValueError, match="surface nulle"):
        ECHANGEUR.coefficient_rapporte(0.0)


def test_une_capacite_calorifique_nulle_interdit_le_nut() -> None:
    """Un fluide de capacite nulle ne transporte rien : le NUT n'existe pas."""
    with pytest.raises(ValueError, match="capacite calorifique minimale"):
        ECHANGEUR.nombre_d_unites_de_transfert(0.0)


def test_un_nut_nul_interdit_l_efficacite() -> None:
    """Sans surface d'echange, l'efficacite n'est pas definie, elle est nulle."""
    with pytest.raises(ValueError, match="NUT nul"):
        efficacite_courants_croises_sans_melange(0.0, 0.5)


def test_les_conditions_de_couette_parlent_si_sympy_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le systeme est lineaire et non degenere : sympy le resout toujours.

    La garde existe pour le jour ou ce ne serait plus vrai, et ce test la
    declenche en faisant mentir `solve` -- faute de pouvoir la declencher
    par une donnee.
    """
    monkeypatch.setattr(sp, "solve", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError, match="conditions de Couette"):
        navier_stokes.profil_de_couette()


def test_un_arbre_de_separation_non_elague_est_arrete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le garde-fou contre l'explosion combinatoire, declenche en le baissant.

    Un branch and bound dont l'elagage echouerait tournerait sans fin sur
    un probleme de taille moyenne. Le plafond est abaisse a un noeud pour
    verifier qu'il parle, au lieu d'attendre une explosion reelle.
    """
    monkeypatch.setattr(variables_entieres, "NOEUDS_MAXIMAUX", 0)
    programme = ProgrammeLineaire(
        objectif=sp.Matrix([1, 1]),
        contraintes=sp.Matrix([[2, 2]]),
        seconds_membres=sp.Matrix([3]),
    )
    with pytest.raises(ValueError, match="arbre de separation non elague"):
        variables_entieres.resoudre_en_entiers(programme)
