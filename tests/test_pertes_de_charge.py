"""Chapitre V des fluides : diametre hydraulique, frottement, pertes de charge.

Oracles : `MecaniqueFluides/SLIDES/Slides_04_EcoulementConduite_Similitudes.pdf`
(planches citees), et un second chemin quand le cours en donne un --
Poiseuille (`polytech.fluides.domain.conduite`) pour la contrainte
parietale, Blasius pour Colebrook en paroi lisse.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.conduite import (
    coefficient_de_blasius,
    perte_de_charge_de_poiseuille,
    profil_de_poiseuille,
)
from polytech.fluides.domain.pertes_de_charge import (
    ENTREE_A_BORD_ARRONDI,
    ENTREE_A_BORD_VIF,
    chute_de_pression_par_la_paroi,
    coefficient_de_colebrook,
    coefficient_de_nikuradse,
    contrainte_parietale_laminaire,
    diametre_hydraulique,
    perte_de_charge_singuliere,
    pertes_de_charge_totales,
    vitesse_moyenne,
)


def test_le_diametre_hydraulique_d_un_tube_circulaire_est_son_diametre() -> None:
    """Planche 9 : `4 (pi D^2 / 4) / (pi D) = D`."""
    d = 0.04
    assert diametre_hydraulique(math.pi * d**2 / 4, math.pi * d) == pytest.approx(d)


def test_le_diametre_hydraulique_d_un_conduit_carre_est_son_cote() -> None:
    """Planche 9 : `4 a^2 / (4 a) = a`."""
    assert diametre_hydraulique(0.3**2, 4 * 0.3) == pytest.approx(0.3)


def test_le_diametre_hydraulique_d_un_conduit_rectangulaire() -> None:
    """Planche 9 : `4 a b / (2 (a + b)) = 2 a b / (a + b)`."""
    a, b = 0.2, 0.5
    assert diametre_hydraulique(a * b, 2 * (a + b)) == pytest.approx(
        2 * a * b / (a + b)
    )


@pytest.mark.parametrize(("aire", "perimetre"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 2.0)])
def test_sans_section_leve(aire: float, perimetre: float) -> None:
    with pytest.raises(ValueError, match="section"):
        diametre_hydraulique(aire, perimetre)


def test_la_vitesse_moyenne_de_poiseuille_vaut_la_moitie_du_maximum() -> None:
    """Planche 15 : `V_moy = (2 / R^2) int_0^R u r dr`, ici `u_max / 2`."""
    r, rayon, mu, g = sp.symbols("r R mu G", positive=True)
    profil = profil_de_poiseuille(r, rayon, mu, g)
    assert sp.simplify(vitesse_moyenne(profil, r, rayon) - profil.subs(r, 0) / 2) == 0


def test_la_paroi_redonne_la_perte_de_charge_de_poiseuille() -> None:
    """Planches 11 et 19 : `Delta p = 4 L tau_w / D` avec `tau_w = 4 mu V / R`.

    Second chemin : le debit de Poiseuille, `Delta p = 8 mu L Q / (pi R^4)`.
    """
    mu, v, rayon, longueur = 1.0e-3, 0.05, 0.01, 12.0
    tau = contrainte_parietale_laminaire(mu, v, rayon)
    par_la_paroi = chute_de_pression_par_la_paroi(tau, longueur, 2 * rayon)
    debit = v * math.pi * rayon**2
    assert par_la_paroi == pytest.approx(
        perte_de_charge_de_poiseuille(debit, rayon, longueur, mu)
    )


def test_nikuradse_suit_sa_formule_de_la_planche_23() -> None:
    """`1 / sqrt(lambda) = 1,14 - 0,868 ln(eps / D)`."""
    lam = coefficient_de_nikuradse(0.001)
    assert 1 / math.sqrt(lam) == pytest.approx(1.14 - 0.868 * math.log(0.001))


def test_colebrook_rejoint_nikuradse_en_regime_rugueux() -> None:
    """Planche 21 : a tres grand Re, `lambda` ne depend plus que de `eps / D`."""
    assert coefficient_de_colebrook(1e12, 0.01) == pytest.approx(
        coefficient_de_nikuradse(0.01), rel=5e-3
    )


def test_colebrook_rejoint_blasius_en_paroi_lisse() -> None:
    """Planche 23, paroi lisse : Blasius, a quelques pour cent pres a `Re = 10^5`."""
    assert coefficient_de_colebrook(1e5, 0.0) == pytest.approx(
        coefficient_de_blasius(1e5), rel=3e-2
    )


def test_colebrook_verifie_son_equation() -> None:
    """Planche 21 : `1/sqrt(l) = -2 log10(eps / (3,71 D) + 2,51 / (Re sqrt(l)))`."""
    re, eps = 1e6, 0.001
    lam = coefficient_de_colebrook(re, eps)
    droite = -2 * math.log10(eps / 3.71 + 2.51 / (re * math.sqrt(lam)))
    assert 1 / math.sqrt(lam) == pytest.approx(droite, rel=1e-10)


def test_colebrook_lit_le_diagramme_de_moody() -> None:
    """La lecture courante du diagramme : `lambda ~ 0,0199` a `eps / D = 0,001`."""
    assert coefficient_de_colebrook(1e6, 0.001) == pytest.approx(0.0199, abs=2e-4)


@pytest.mark.parametrize(("re", "eps"), [(2000.0, 0.0), (1e5, -0.1)])
def test_colebrook_hors_de_son_domaine_leve(re: float, eps: float) -> None:
    with pytest.raises(ValueError, match="Colebrook"):
        coefficient_de_colebrook(re, eps)


def test_nikuradse_sans_rugosite_leve() -> None:
    with pytest.raises(ValueError, match="rugosite"):
        coefficient_de_nikuradse(0.0)


def test_une_entree_a_bord_vif_perd_la_moitie_de_la_pression_dynamique() -> None:
    """Planche 24 : `K_L = 0,50` a bord vif, `0,03` arrondi."""
    assert ENTREE_A_BORD_VIF == 0.50
    assert ENTREE_A_BORD_ARRONDI == 0.03
    assert perte_de_charge_singuliere(ENTREE_A_BORD_VIF, 1000.0, 2.0) == 1000.0


def test_un_coefficient_singulier_negatif_leve() -> None:
    with pytest.raises(ValueError, match="singulier"):
        perte_de_charge_singuliere(-0.1, 1000.0, 2.0)


def test_les_pertes_totales_somment_regulieres_et_singulieres() -> None:
    """Planche 26 : `Delta P_tot = sum reparties + sum singulieres`."""
    assert pertes_de_charge_totales([100.0, 50.0], [25.0]) == 175.0
    assert pertes_de_charge_totales([], []) == 0.0


def test_une_perte_negative_leve() -> None:
    with pytest.raises(ValueError, match="positive"):
        pertes_de_charge_totales([100.0], [-1.0])
