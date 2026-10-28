"""La reponse en frequence du biquad passe-bas : coupure, resonance, cascade.

Oracles : le corrige officiel du TP 7 d'electronique
(`ElectroniqueInstrumentation/TP/TP_7_Solution.pdf`, exercice 2.1, pages 2
a 4) -- `|H(j w_0)| = Q`, la coupure en `w_0` pour `Q = 1/racine(2)`, la
formule de la page 3, et le point `(1552,387 rad/s ; -2,9922 dB)` que sa
figure de la page 4 marque sur la courbe `xi = 0,02`.

Un ecart : la page 3 imprime `racine(1 + racine(2)) ~ 1.558` ; ce nombre
vaut 1,5538, et le point de la figure le confirme.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from polytech.electronique.domain.reponse_du_biquad import (
    COUPURE_SANS_AMORTISSEMENT,
    coupure_a_trois_decibels,
    module_carre,
    premiere_pulsation_d_ecart,
    pulsation_de_resonance,
    surtension,
)

if TYPE_CHECKING:
    from collections.abc import Callable

RACINE = 1 / math.sqrt(2)


def test_le_module_a_la_pulsation_propre_vaut_q() -> None:
    """Page 2 : `|H(j w_0)|^2 = Q^2`."""
    for qualite in (0.3, RACINE, 1.0, 25.0):
        assert module_carre(1.0, qualite) == pytest.approx(qualite**2, rel=1e-12)
    assert module_carre(0.0, 3.0) == 1.0


def test_le_module_carre_de_la_page_trois() -> None:
    """`w_0^4 / ((w_0^2 - w^2)^2 + w^2 w_0^2 / Q^2)`, en `x = w / w_0`."""
    assert module_carre(2.0, 0.5) == pytest.approx(1 / (9 + 16), rel=1e-12)
    assert module_carre(0.5, 2.0) == pytest.approx(1 / (0.5625 + 0.0625), rel=1e-12)


def test_le_point_marque_sur_la_figure_du_corrige() -> None:
    """Page 4 : `xi = 0,02`, `w = 1552,387 rad/s` pour `w_0 = 1000` ; -2,9922 dB."""
    carre = module_carre(1.552387, 1 / (2 * 0.02))
    assert 10 * math.log10(carre) == pytest.approx(-2.9922, abs=5e-5)


def test_a_un_sur_racine_de_deux_la_coupure_est_la_pulsation_propre() -> None:
    """Page 2, sous-question 4."""
    assert coupure_a_trois_decibels(RACINE) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("qualite", [0.1, 0.5, RACINE, 1.0, 2.0, 25.0])
def test_la_coupure_divise_la_puissance_par_deux(qualite: float) -> None:
    """La definition, refaite par le module : `|H(j w_c)|^2 = 1/2`."""
    coupure = coupure_a_trois_decibels(qualite)
    assert module_carre(coupure, qualite) == pytest.approx(0.5, rel=1e-9)


def test_la_coupure_sans_amortissement() -> None:
    """Page 3 : `w_c -> w_0 racine(1 + racine(2))`, soit 1,5538 et non 1,558."""
    assert coupure_a_trois_decibels(1e12) == pytest.approx(
        COUPURE_SANS_AMORTISSEMENT, rel=1e-12
    )
    assert pytest.approx(1.5538, abs=5e-5) == COUPURE_SANS_AMORTISSEMENT
    assert abs(COUPURE_SANS_AMORTISSEMENT - 1.558) > 4e-3
    proche = coupure_a_trois_decibels(1e6)
    assert proche == pytest.approx(COUPURE_SANS_AMORTISSEMENT, rel=1e-9)
    assert coupure_a_trois_decibels(25.0) == pytest.approx(1.5533, abs=5e-5)


def test_les_coupures_des_deux_exercices_actifs() -> None:
    """`Q = 1` et `Q = 2` : la coupure n'est pas `w_0`, mais 1,272 et 1,485 fois."""
    un = math.sqrt((1 + math.sqrt(5)) / 2)
    assert coupure_a_trois_decibels(1.0) == pytest.approx(un, rel=1e-12)
    deux = math.sqrt((1.75 + math.sqrt(1.75**2 + 4)) / 2)
    assert coupure_a_trois_decibels(2.0) == pytest.approx(deux, rel=1e-12)


def test_la_coupure_d_une_cascade_de_cellules_identiques() -> None:
    """Trois cellules : chacune ne doit perdre que 1 dB, `|H|^2 = 2^(-1/3)`."""
    trois = coupure_a_trois_decibels(1.0, cellules=3)
    assert module_carre(trois, 1.0) ** 3 == pytest.approx(0.5, rel=1e-9)
    assert trois < coupure_a_trois_decibels(1.0)
    assert trois == pytest.approx(1.1019, abs=5e-5)
    with pytest.raises(ValueError, match="nombre de cellules"):
        coupure_a_trois_decibels(1.0, cellules=0)


def test_la_resonance_n_existe_qu_au_dela_de_un_sur_racine_de_deux() -> None:
    """`d|H|^2/dw = 0` hors de l'origine demande `1 - 1/(2 Q^2) > 0`."""
    assert pulsation_de_resonance(2.0) == pytest.approx(math.sqrt(0.875), rel=1e-12)
    assert surtension(2.0) == pytest.approx(2 / math.sqrt(1 - 1 / 16), rel=1e-12)
    sommet = pulsation_de_resonance(2.0)
    assert module_carre(sommet, 2.0) == pytest.approx(surtension(2.0) ** 2, rel=1e-9)
    for voisin in (sommet - 1e-3, sommet + 1e-3):
        assert module_carre(voisin, 2.0) < surtension(2.0) ** 2
    for plat in (RACINE, 0.5):
        with pytest.raises(ValueError, match="pas de resonance"):
            pulsation_de_resonance(plat)
        with pytest.raises(ValueError, match="pas de resonance"):
            surtension(plat)


def test_la_bande_ou_le_gain_reste_plat() -> None:
    """Le gain d'une cellule resonante quitte un gabarit de 1 dB avant `w_0`."""
    bord = premiere_pulsation_d_ecart(2.0, 1.0)
    assert 10 * math.log10(module_carre(bord, 2.0)) == pytest.approx(1.0, abs=1e-9)
    assert 0 < bord < pulsation_de_resonance(2.0)
    assert bord == pytest.approx(0.3559, abs=5e-5)
    with pytest.raises(ValueError, match="jamais atteint"):
        premiere_pulsation_d_ecart(1.0, 3.0)
    with pytest.raises(ValueError, match="jamais atteint"):
        premiere_pulsation_d_ecart(0.5, 1.0)
    with pytest.raises(ValueError, match="ecart nul ou negatif"):
        premiere_pulsation_d_ecart(2.0, 0.0)


@pytest.mark.parametrize("fautif", [0.0, -1.0])
def test_un_facteur_de_qualite_non_positif_est_refuse(fautif: float) -> None:
    """Sans `Q > 0`, le terme en `s` n'a pas de sens."""
    appels: tuple[Callable[[], float], ...] = (
        lambda: module_carre(1.0, fautif),
        lambda: coupure_a_trois_decibels(fautif),
        lambda: pulsation_de_resonance(fautif),
        lambda: premiere_pulsation_d_ecart(fautif, 1.0),
    )
    for appel in appels:
        with pytest.raises(ValueError, match="facteur de qualite"):
            appel()
    with pytest.raises(ValueError, match="pulsation negative"):
        module_carre(-1.0, 1.0)
