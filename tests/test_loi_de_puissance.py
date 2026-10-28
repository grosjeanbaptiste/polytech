"""Le profil turbulent en loi de puissance 1/7 (fiche : regimes laminaire et turbulent).

Oracle : l'integrale du profil sur la section, faite par sympy, et la
figure de la fiche, `(1 - |y - 1|)^(1/7)` : plus plat que la parabole.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.fluides.domain.loi_de_puissance import (
    EXPOSANT_DE_BLASIUS,
    rapport_vitesse_moyenne_sur_maximale,
    vitesse_en_loi_de_puissance,
)
from polytech.maths.domain.comparaison import egales, proches_relativement


def test_l_exposant_du_cours_est_un_septieme() -> None:
    assert EXPOSANT_DE_BLASIUS == 7


def test_la_vitesse_est_nulle_a_la_paroi_et_maximale_sur_l_axe() -> None:
    assert vitesse_en_loi_de_puissance(0.0, 0.1, 2.0) == 0.0
    assert vitesse_en_loi_de_puissance(0.1, 0.1, 2.0) == pytest.approx(2.0)


def test_la_vitesse_vaut_la_racine_septieme_du_rapport_des_distances() -> None:
    assert vitesse_en_loi_de_puissance(0.05, 0.1, 1.0) == pytest.approx(0.5 ** (1 / 7))


def test_le_profil_est_plus_plat_que_la_parabole_de_poiseuille() -> None:
    assert vitesse_en_loi_de_puissance(0.01, 0.1, 1.0) > 1 - (1 - 0.1) ** 2


@pytest.mark.parametrize(("y", "r"), [(-0.01, 0.1), (0.11, 0.1), (0.05, 0.0)])
def test_hors_de_la_section_la_loi_leve(y: float, r: float) -> None:
    with pytest.raises(ValueError, match=r"section|rayon"):
        vitesse_en_loi_de_puissance(y, r, 1.0)


def test_l_exposant_doit_etre_strictement_positif() -> None:
    with pytest.raises(ValueError, match="exposant"):
        vitesse_en_loi_de_puissance(0.05, 0.1, 1.0, exposant=0)


def test_la_moyenne_sur_la_section_est_49_sur_60_du_maximum() -> None:
    r, rayon, y = sp.symbols("r R y", positive=True)
    profil = (y / rayon) ** sp.Rational(1, 7)
    moyenne = sp.integrate(profil.subs(y, rayon - r) * r, (r, 0, rayon)) * 2 / rayon**2
    assert egales(rapport_vitesse_moyenne_sur_maximale(7), moyenne)
    assert proches_relativement(rapport_vitesse_moyenne_sur_maximale(7), 0.8167, 1e-3)


def test_le_rapport_tend_vers_un_quand_le_profil_s_aplatit() -> None:
    assert rapport_vitesse_moyenne_sur_maximale(1000) > 0.99


def test_un_exposant_non_positif_leve_pour_le_rapport() -> None:
    with pytest.raises(ValueError, match="exposant"):
        rapport_vitesse_moyenne_sur_maximale(0)
