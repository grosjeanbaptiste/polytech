"""Les raisonnements du chapitre I d'optimisation, etape par etape.

L'oracle est encore l'illustration de la page 25 du cours : une variable
libre, un maximum, une inegalite de chaque sens et une egalite. Le
raisonnement doit dire, dans l'ordre du cours, ce que l'on fait de
chacune et pourquoi.
"""

from __future__ import annotations

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.explications import expliquer_la_forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral

INF, SUP, EGA = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
SOURCE = "Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf"


def _illustration() -> ProgrammeGeneral:
    return ProgrammeGeneral(
        optimisation=Optimisation.MAX,
        objectif=(1, -4),
        contraintes=(
            Contrainte((1, 0), INF, 7),
            Contrainte((3, -2), SUP, 4),
            Contrainte((1, 1), EGA, 5),
        ),
        libres=frozenset({1}),
    )


def test_l_illustration_demande_quatre_etapes() -> None:
    """La variable libre, le maximum, puis un ecart par inegalite."""
    assert len(expliquer_la_forme_standard(_illustration()).etapes) == 4


def test_la_variable_libre_est_traitee_la_premiere() -> None:
    premiere = expliquer_la_forme_standard(_illustration()).etapes[0]
    assert "x_2 = x_2^+ - x_2^-" in premiere.affirmation
    assert "signe quelconque" in premiere.justification


def test_le_maximum_devient_un_minimum() -> None:
    deuxieme = expliquer_la_forme_standard(_illustration()).etapes[1]
    assert "min -z" in deuxieme.affirmation


def test_chaque_inegalite_recoit_son_ecart_avec_son_signe() -> None:
    """`<=` : on ajoute `x_4` ; `>=` : on retranche `x_5` ; l'egalite n'a rien."""
    _, _, inferieure, superieure = expliquer_la_forme_standard(_illustration()).etapes
    assert "+ x_4" in inferieure.affirmation
    assert "- x_5" in superieure.affirmation


def test_chaque_etape_cite_le_cours() -> None:
    raisonnement = expliquer_la_forme_standard(_illustration())
    assert all(e.source.startswith(SOURCE) for e in raisonnement.etapes)
    assert raisonnement.sources == (f"{SOURCE}, p. 24", f"{SOURCE}, p. 25")


def test_la_conclusion_compte_les_variables() -> None:
    conclusion = expliquer_la_forme_standard(_illustration()).conclusion
    assert "5 variables" in conclusion
    assert "2 variables d'ecart" in conclusion


def test_un_programme_deja_standard_le_dit() -> None:
    standard = ProgrammeGeneral(Optimisation.MIN, (1, 1), (Contrainte((1, 1), EGA, 2),))
    raisonnement = expliquer_la_forme_standard(standard)
    assert len(raisonnement.etapes) == 1
    assert "deja sous forme standard" in raisonnement.etapes[0].affirmation
