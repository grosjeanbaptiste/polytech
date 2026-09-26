"""L'hypothese (H3) ne doit dependre d'aucun choix arbitraire.

Le corrige la justifiait avec une temperature et une hauteur de ciel gazeux
choisies par commodite, et en comparant a p1 -- la grandeur que l'exercice
cherche justement a calculer. Ces tests exigent mieux : que la pression se
simplifie (donc aucune circularite) et que la conclusion tienne sur toute
une plage de conditions plausibles.
"""

import pytest

from polytech.fluides.domain.grandeurs import Hauteur, Pression
from polytech.fluides.domain.statique.gaz import (
    AIR,
    TEMPERATURE_AMBIANTE,
    majorant_de_la_part_negligee,
    part_negligee,
    part_negligee_litterale,
)

CIEL = Hauteur(metres=0.5)
PRECISION_DE_LECTURE = 1e-3
"""Un millimetre de mercure pese environ un millieme de la mesure."""


def test_la_part_negligee_ne_depend_pas_de_la_pression() -> None:
    """P se simplifie : justifier (H3) par p1 n'est donc pas circulaire."""
    basse = part_negligee(AIR, Pression(pascals=50_000.0), CIEL, TEMPERATURE_AMBIANTE)
    haute = part_negligee(AIR, Pression(pascals=500_000.0), CIEL, TEMPERATURE_AMBIANTE)
    assert basse == pytest.approx(haute, rel=1e-12)


def test_l_expression_litterale_montre_la_simplification() -> None:
    """M g h / (R T) : le document doit pouvoir l'afficher.

    On compare les *noms* des symboles : deux symboles de memes nom mais
    d'hypotheses differentes ne sont pas egaux pour sympy, et l'assertion
    passerait alors pour une mauvaise raison.
    """
    noms = {symbole.name for symbole in part_negligee_litterale().free_symbols}
    assert noms == {"M", "g", "h", "R", "T"}
    assert "p" not in noms


def test_le_pire_cas_plausible_reste_sous_la_precision_de_lecture() -> None:
    """Air le plus froid et ciel gazeux le plus haut : (H3) tient encore."""
    majorant = majorant_de_la_part_negligee(
        AIR,
        Hauteur(metres=2.0),
        temperature_minimale=273.15,
    )
    assert majorant < PRECISION_DE_LECTURE


def test_le_majorant_refuse_une_plage_de_temperature_absurde() -> None:
    """Une temperature minimale nulle ferait diverger le majorant."""
    with pytest.raises(ValueError, match="temperature"):
        majorant_de_la_part_negligee(AIR, CIEL, temperature_minimale=0.0)
