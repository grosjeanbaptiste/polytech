"""Un raisonnement explique : des etapes, chacune justifiee et sourcee.

La bibliotheque ne rend pas seulement des resultats : elle doit pouvoir
dire par quelles etapes on y arrive, et pourquoi chacune est licite.
« Rien ne doit tomber du ciel » est ici un invariant : une etape sans
justification est refusee a la construction.
"""

from __future__ import annotations

import pytest

from polytech.raisonnement.domain.etapes import Etape, Raisonnement


def _etape(affirmation: str = "x = 2", source: str = "cours, p. 1") -> Etape:
    return Etape(affirmation, justification="on divise par 3", source=source)


def test_une_etape_sans_justification_est_refusee() -> None:
    with pytest.raises(ValueError, match="sans justification"):
        Etape("x = 2", justification="  ")


def test_une_etape_sans_affirmation_est_refusee() -> None:
    with pytest.raises(ValueError, match="n'affirme rien"):
        Etape("", justification="on divise par 3")


def test_un_raisonnement_sans_etape_est_refuse() -> None:
    with pytest.raises(ValueError, match="aucune etape"):
        Raisonnement(question="Que vaut x ?", etapes=(), conclusion="x = 2")


def test_un_raisonnement_sans_conclusion_est_refuse() -> None:
    with pytest.raises(ValueError, match="sans conclusion"):
        Raisonnement(question="Que vaut x ?", etapes=(_etape(),), conclusion="")


def test_les_sources_sont_rendues_une_fois_dans_l_ordre() -> None:
    raisonnement = Raisonnement(
        question="Que vaut x ?",
        etapes=(
            _etape(source="B"),
            _etape(source="A"),
            _etape(source="B"),
            _etape(source=""),
        ),
        conclusion="x = 2",
    )
    assert raisonnement.sources == ("B", "A")


def test_le_texte_numerote_les_etapes_et_donne_le_pourquoi() -> None:
    raisonnement = Raisonnement(
        question="Que vaut x ?",
        etapes=(Etape("3 x = 6", justification="c'est l'enonce"), _etape()),
        conclusion="x = 2",
    )
    assert raisonnement.en_texte().splitlines() == [
        "Que vaut x ?",
        "1. 3 x = 6 -- parce que c'est l'enonce",
        "2. x = 2 -- parce que on divise par 3 [cours, p. 1]",
        "Donc : x = 2",
    ]


def test_deux_raisonnements_s_enchainent() -> None:
    """La conclusion du premier devient le point de depart du second."""
    premier = Raisonnement("A ?", (_etape("a"),), "a")
    second = Raisonnement("B ?", (_etape("b"),), "b")
    enchaine = premier.puis(second)
    assert [e.affirmation for e in enchaine.etapes] == ["a", "b"]
    assert enchaine.question == "A ?"
    assert enchaine.conclusion == "b"
