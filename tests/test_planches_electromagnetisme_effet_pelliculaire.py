"""Planches IV.4 (29 a 31) et note manuscrite : l'effet pelliculaire.

Sources : `Electromagnetisme/SLIDES/Slides_04_Etats_variables_partie_4.pdf`,
planches 29 a 31, lues sur les pages rendues, et
`Slides_Note_Effet_pelliculaire.pdf` (trois pages manuscrites, adaptees de
Perez, Carles et Fleckinger).

La note mene le calcul de bout en bout : equation de diffusion, regime
sinusoidal, equation caracteristique, condition a l'infini. Chaque etape
est eprouvee, et le resultat est recoupe par l'equation de diffusion du
chapitre 15 et par l'epaisseur de peau que la bibliotheque calculait deja.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.phaseurs import valeur_instantanee
from polytech.electromagnetisme.domain.circuits.resistivite import METAUX
from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.diffusion import residu_de_diffusion_plane
from polytech.electromagnetisme.domain.effet_pelliculaire import epaisseur_de_peau
from polytech.electromagnetisme.domain.profondeur_de_penetration import (
    courant_efficace_en_profondeur,
    profondeur_de_penetration,
    racines_caracteristiques,
    retard_de_phase_en_profondeur,
    substitut_du_courant_en_profondeur,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE

RHO, MU, OMEGA, SURFACE = sp.symbols("rho_e mu omega j_0", positive=True)
PEAU = profondeur_de_penetration(RHO, OMEGA, MU)


def test_la_profondeur_de_penetration_sous_ses_deux_ecritures() -> None:
    """Planche 29 : `delta = racine(2 rho / (omega mu)) = racine(rho / (pi f mu))`."""
    assert sp.sqrt(2 * RHO / (OMEGA * MU)) == PEAU
    frequence = sp.Symbol("f", positive=True)
    par_la_frequence = profondeur_de_penetration(RHO, 2 * sp.pi * frequence, MU)
    assert sp.simplify(par_la_frequence - sp.sqrt(RHO / (sp.pi * frequence * MU))) == 0


def test_elle_s_accorde_avec_l_epaisseur_de_peau_du_cuivre() -> None:
    """Planche 29 : « 9 mm a 50 Hz » ; le calcul donne 9,3 mm."""
    cuivre = METAUX["cuivre"][0]
    peau = profondeur_de_penetration(cuivre, 100 * math.pi, PERMEABILITE_DU_VIDE)
    assert float(peau) == pytest.approx(epaisseur_de_peau(50.0, 1 / cuivre))
    assert float(peau) == pytest.approx(9e-3, rel=0.04)


def test_une_resistivite_nulle_n_a_pas_de_profondeur() -> None:
    with pytest.raises(ValueError, match="nulle ou negative"):
        profondeur_de_penetration(0, OMEGA, MU)
    with pytest.raises(ValueError, match="nulle ou negative"):
        profondeur_de_penetration(RHO, 0, MU)
    with pytest.raises(ValueError, match="nulle ou negative"):
        profondeur_de_penetration(RHO, OMEGA, -1)


def test_les_racines_de_l_equation_caracteristique() -> None:
    """Note : `r^2 = j omega mu / rho`, soit `r = +-(1 + j) / delta`."""
    positive, negative = racines_caracteristiques(PEAU)
    for racine in (positive, negative):
        assert sp.simplify(racine**2 - sp.I * OMEGA * MU / RHO) == 0
    assert sp.simplify(positive + negative) == 0
    assert sp.re(positive).is_positive


def test_le_substitut_resout_l_equation_en_regime_sinusoidal() -> None:
    """Planche 31 et note : `d2j/dz2 = j omega (mu / rho) j`, borne a l'infini."""
    substitut = substitut_du_courant_en_profondeur(SURFACE, ABSCISSE, PEAU)
    residu = sp.diff(substitut, ABSCISSE, 2) - sp.I * OMEGA * MU / RHO * substitut
    assert sp.simplify(residu) == 0
    # Conditions aux limites : la valeur de surface, et rien a l'infini.
    assert substitut.subs(ABSCISSE, 0) == SURFACE
    assert sp.limit(sp.Abs(substitut.subs({PEAU: 1})), ABSCISSE, sp.oo) == 0


def test_le_courant_instantane_resout_l_equation_de_diffusion() -> None:
    """Planche 30 : `d2i/dz2 = (mu / rho) di/dt`, celle du chapitre 15."""
    substitut = substitut_du_courant_en_profondeur(SURFACE, ABSCISSE, PEAU)
    instantane = valeur_instantanee(substitut, OMEGA)
    assert residu_de_diffusion_plane(instantane, MU, RHO) == 0


def test_la_valeur_efficace_decroit_exponentiellement() -> None:
    """Planche 31 : `i_eff(z) = i_0 exp(-z / delta)`, soit `i_0 / e` en `z = delta`."""
    efficace = courant_efficace_en_profondeur(SURFACE, ABSCISSE, PEAU)
    substitut = substitut_du_courant_en_profondeur(SURFACE, ABSCISSE, PEAU)
    assert sp.simplify(sp.Abs(substitut) - efficace) == 0
    assert sp.simplify(efficace.subs(ABSCISSE, PEAU) - SURFACE / sp.E) == 0
    # A cinq profondeurs de penetration, il reste moins d'un centieme.
    assert float(courant_efficace_en_profondeur(1, 5, 1)) < 0.01


def test_le_courant_retarde_d_un_radian_par_profondeur_de_penetration() -> None:
    """Note : `j = j_eff(z) racine(2) cos(omega t - phi(z))`, `phi = z / delta`."""
    retard = retard_de_phase_en_profondeur(ABSCISSE, PEAU)
    assert retard == ABSCISSE / PEAU
    efficace = courant_efficace_en_profondeur(SURFACE, ABSCISSE, PEAU)
    substitut = substitut_du_courant_en_profondeur(SURFACE, ABSCISSE, PEAU)
    assert sp.simplify(substitut - efficace * sp.exp(-sp.I * retard)) == 0


def test_une_profondeur_de_penetration_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="profondeur de penetration nulle"):
        courant_efficace_en_profondeur(1, 1, 0)
