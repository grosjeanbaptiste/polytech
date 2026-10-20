"""Les cas de l'etude parametrique : trois cylindres que le cours chiffre.

Ni l'enonce ni le cours de fluides ne chiffrent de cylindre dans le vent.
Le cours de thermique en chiffre trois, et ce sont eux qui sont etudies :
deux cheminees et une serre, chacune avec son diametre, son vent et sa
temperature d'air, que `test_sources_devoir_cylindre` retrouve dans les
enonces.
"""

from __future__ import annotations

import pytest

from polytech.fluides.application import devoir_cylindre_cas as cas
from polytech.fluides.application import devoir_cylindre_fluides as dcf


def test_les_trois_cas_et_leurs_cles() -> None:
    assert [c.cle for c in cas.CAS] == ["ChemineeFour", "ChemineeUsine", "Serre"]


@pytest.mark.parametrize(
    ("cle", "diametre", "vitesse", "celsius"),
    [
        ("ChemineeFour", 0.5, 5.0, -10.0),
        ("ChemineeUsine", 1.04, 5.0, -10.0),
        ("Serre", 5.0, 0.75, 25.0),
    ],
)
def test_les_donnees_de_chaque_cas(
    cle: str, diametre: float, vitesse: float, celsius: float
) -> None:
    point = cas.cas(cle).point
    assert point.diametre == pytest.approx(diametre)
    assert point.vitesse == pytest.approx(vitesse)
    assert point.fluide == dcf.air_a(celsius)


def test_chaque_cas_cite_son_document() -> None:
    for c in cas.CAS:
        assert c.source.startswith("ThermiqueFondements/EXAMENS/")


def test_un_cas_inconnu_leve() -> None:
    with pytest.raises(KeyError, match="aucun cas d'etude"):
        cas.cas("Pylone")
