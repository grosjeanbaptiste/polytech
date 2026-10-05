"""Le syllabus de thermique, tome 2 : l'isolation par une lame d'air.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 112
a 114 (deuxieme partie, chapitre 1, section 1.7).

Les oracles : `K_gl = 6 W/m^2 K` pour la vitre simple, puis pour le
double vitrage `Gr_b = 1000`, `K_c = 3,83`, `K_r = 3,8`, leur somme 7,63
et `K_gl = 3,3` -- « environ 45 % » de deperditions en moins.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.convection.naturelle import CONDUCTION_PURE, grashof
from polytech.thermique.domain.double_vitrage import (
    Vitrage,
    coefficient_de_la_lame,
    coefficient_de_rayonnement_entre_vitres,
    ecart_entre_les_vitres,
)
from polytech.thermique.domain.rayonnement.vitre_selective import (
    coefficient_entre_plans_gris,
)

VITRAGE = Vitrage(epaisseur=0.003, conductivite=0.7, echanges=(23.0, 8.0))
MOYENNE = 283.0
LAME, AIR = 0.006, 0.023


def test_la_vitre_simple() -> None:
    """Page 112 : `1 / K_gl = 0,043 + 0,004 + 0,12 = 0,173`, soit 6 W/m^2 K.

    Le dernier terme est `1 / 8 = 0,125` ; la somme vaut bien 0,173.
    """
    assert pytest.approx(0.043, abs=5e-4) == 1 / 23.0
    assert pytest.approx(0.004, abs=5e-4) == 0.003 / 0.7
    simple = VITRAGE.coefficient_global(())
    assert 1 / simple == pytest.approx(0.173, abs=5e-4)
    assert simple == pytest.approx(5.79, abs=0.01)
    assert round(simple) == 6


def test_l_air_de_la_lame_reste_au_repos() -> None:
    """Page 114 : `Gr_b = 9,81 x 20 x 0,006^3 / (283 x (1,236e-5)^2) ~ 1000 < 1400`."""
    nombre = grashof(1 / MOYENNE, 20.0, LAME, 1.236e-5)
    assert nombre == pytest.approx(980.0, abs=1.0)
    assert nombre < CONDUCTION_PURE
    assert coefficient_de_la_lame(LAME, AIR, nombre * 0.71) == pytest.approx(
        3.83, abs=0.005
    )


def test_la_lame_convecte_au_dela_du_seuil() -> None:
    """Page 112 : `lambda_eq / lambda_a = 0,11 (Gr_b Pr)^0,29` au-dela de 1400."""
    assert coefficient_de_la_lame(LAME, AIR, 1e5) == pytest.approx(
        AIR / LAME * 0.11 * 1e5**0.29
    )
    with pytest.raises(ValueError, match="epaisseur de lame nulle ou negative"):
        coefficient_de_la_lame(0.0, AIR, 500.0)


def test_le_rayonnement_entre_les_deux_vitres() -> None:
    """Page 113 : `K_r = 4 sigma T_m^3 / (2 / epsilon - 1) = 3,8 W/m^2 K`."""
    rayonnement = coefficient_de_rayonnement_entre_vitres(0.85, MOYENNE)
    assert rayonnement == pytest.approx(3.8, abs=0.01)
    assert rayonnement == pytest.approx(
        coefficient_entre_plans_gris(0.85, 0.85, MOYENNE, MOYENNE)
    )
    with pytest.raises(ValueError, match="facteur d'emission hors de"):
        coefficient_de_rayonnement_entre_vitres(0.0, MOYENNE)


def test_le_double_vitrage() -> None:
    """Page 114 : `K_c + K_r = 7,63` et `K_gl = 3,3 W/m^2 K`.

    Les 3,3 viennent de la somme des termes arrondis (3,26) ; sans
    arrondi intermediaire, 3,25.
    """
    lame = AIR / LAME + coefficient_de_rayonnement_entre_vitres(0.85, MOYENNE)
    assert lame == pytest.approx(7.63, abs=0.01)
    assert 1 / lame == pytest.approx(0.131, abs=5e-4)
    double = VITRAGE.coefficient_global((lame,))
    arrondi = 1 / (0.043 + 0.004 + 0.004 + 0.125 + 0.131)
    assert double == pytest.approx(arrondi, rel=5e-3)
    assert round(arrondi, 1) == 3.3
    assert double == pytest.approx(3.25, abs=0.005)


def test_le_double_vitrage_reduit_les_deperditions() -> None:
    """Page 114 : « d'environ 45 % ». Avec ses propres nombres, 44 %."""
    lame = AIR / LAME + coefficient_de_rayonnement_entre_vitres(0.85, MOYENNE)
    gain = 1 - VITRAGE.coefficient_global((lame,)) / VITRAGE.coefficient_global(())
    assert gain == pytest.approx(0.437, abs=0.003)
    triple = VITRAGE.coefficient_global((lame, lame))
    assert triple < VITRAGE.coefficient_global((lame,))


def test_l_ecart_reel_entre_les_vitres() -> None:
    """Page 113 : `((t_i - t_e) - (t_1 - t_2)) / R = (K_c + K_r)(t_1 - t_2)`.

    Les 20 degres de l'estimation sont une borne superieure : l'ecart
    reel est de 8,5 degres, et l'air est plus encore au repos.
    """
    lame = AIR / LAME + coefficient_de_rayonnement_entre_vitres(0.85, MOYENNE)
    ecart = ecart_entre_les_vitres(VITRAGE, 20.0, lambda _: lame)
    externe = 1 / 23.0 + 2 * 0.003 / 0.7 + 1 / 8.0
    assert (20.0 - ecart) / externe == pytest.approx(lame * ecart)
    assert ecart == pytest.approx(8.5, abs=0.05)
    assert VITRAGE.coefficient_global((lame,)) * 20.0 == pytest.approx(lame * ecart)
    assert grashof(1 / MOYENNE, ecart, LAME, 1.236e-5) < 0.5 * CONDUCTION_PURE


def test_un_vitrage_refuse_un_coefficient_nul() -> None:
    with pytest.raises(ValueError, match="coefficient de transmission nul ou negatif"):
        Vitrage(epaisseur=0.003, conductivite=0.7, echanges=(0.0, 8.0))
    with pytest.raises(ValueError, match="coefficient de lame nul ou negatif"):
        VITRAGE.coefficient_global((0.0,))
