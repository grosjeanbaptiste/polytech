"""Le corps noir et les proprietes spectrales de la fiche, refaits par `polytech/`.

Les constantes sont recalculees depuis `h`, `k`, `c` : `sigma` et la loi de
Wien ne sont pas prises dans la bibliotheque pour verifier la bibliotheque.
"""

from __future__ import annotations

import itertools
import math

import sympy as sp
from scipy.integrate import quad

from polytech.maths.domain.comparaison import egales, proches, proches_relativement
from polytech.thermique.domain.rayonnement.bandes import absorptivite_par_bandes
from polytech.thermique.domain.rayonnement.corps_noir import (
    BOLTZMANN,
    LUMIERE,
    PLANCK,
    emittance,
    exitance_spectrale,
    fraction_emise_sous,
    longueur_d_onde_du_maximum,
    proprietes_d_un_solide_reel,
)
from polytech.thermique.domain.rayonnement.lambert import (
    ALPHA,
    exitance_par_integration,
    luminance_du_corps_noir,
)

MICRO = 1e-6


def test_les_trois_facteurs_d_un_rayon_font_un() -> None:
    """`alpha + rho + tau = 1` ; un corps opaque ne transmet rien."""
    absorbe, reflechi, transmis = proprietes_d_un_solide_reel(0.3, 0.5)
    assert proches(absorbe + reflechi + transmis, 1.0, 1e-12)
    assert proches(proprietes_d_un_solide_reel(0.4, 0.6)[2], 0.0, 1e-12)


def test_l_exitance_d_une_surface_de_lambert_vaut_pi_fois_la_luminance() -> None:
    """`M = pi L` : l'integrale sur l'hemisphere d'une luminance constante."""
    luminance = sp.Symbol("L", positive=True)
    assert egales(exitance_par_integration(luminance + 0 * ALPHA), sp.pi * luminance)
    assert proches_relativement(
        emittance(500.0), math.pi * luminance_du_corps_noir(500.0), 1e-12
    )


def test_la_constante_de_stefan_vient_de_h_k_c() -> None:
    """`M_o = sigma T^4` avec `sigma = 2 pi^5 k^4 / (15 h^3 c^2)`."""
    sigma = 2 * math.pi**5 * BOLTZMANN**4 / (15 * PLANCK**3 * LUMIERE**2)
    assert proches_relativement(emittance(300.0), sigma * 300.0**4, 1e-8)


def test_la_loi_de_wien_donne_2898_micrometres_kelvins() -> None:
    """Le maximum de Planck en `x = hc/(lambda k T)` : `x e^x/(e^x - 1) = 5`."""
    x = sp.nsolve(
        sp.Symbol("x") * sp.exp(sp.Symbol("x")) / (sp.exp(sp.Symbol("x")) - 1) - 5,
        5,
        prec=20,
    )
    constante = PLANCK * LUMIERE / (BOLTZMANN * float(x))
    assert proches(constante / MICRO, 2898.0, 0.5)
    assert proches_relativement(longueur_d_onde_du_maximum(1.0), constante, 1e-8)


def test_le_soleil_est_visible_et_le_corps_humain_infrarouge() -> None:
    """`5800 K` : vers `0,5 um` ; `310 K` : vers `9,4 um`."""
    assert proches(longueur_d_onde_du_maximum(5800.0) / MICRO, 0.5, 0.01)
    assert proches(longueur_d_onde_du_maximum(310.0) / MICRO, 9.4, 0.1)


def test_la_fraction_k_ne_depend_que_du_produit_lambda_t() -> None:
    """`k = (1 / sigma T^4) integrale de m_lambda` ; meme `lambda T`, meme `k`."""
    t, borne = 1000.0, 3.0 * MICRO
    integrale, _ = quad(lambda lam: exitance_spectrale(lam, t), 1e-9, borne, limit=200)
    assert proches_relativement(
        fraction_emise_sous(borne, t), integrale / emittance(t), 1e-6
    )
    assert proches_relativement(
        fraction_emise_sous(borne * 2, t / 2), fraction_emise_sous(borne, t), 1e-12
    )


def test_l_absorptivite_par_bandes_pondere_par_la_part_d_energie() -> None:
    """`alpha = somme alpha_i [k(l_i T) - k(l_(i-1) T)]`, `epsilon` a `T_corps`."""
    coupures, facteurs, t = (2.0 * MICRO, 5.0 * MICRO), (0.9, 0.2, 0.1), 3000.0

    def flux(lam: float) -> float:
        bande = sum(lam >= c for c in coupures)
        return facteurs[bande] * exitance_spectrale(lam, t)

    bornes = (1e-9, *coupures, 2e-3)
    haut = sum(quad(flux, a, b, limit=200)[0] for a, b in itertools.pairwise(bornes))
    obtenue = absorptivite_par_bandes(facteurs, coupures, t)
    assert proches_relativement(obtenue, haut / emittance(t), 1e-4)


def test_la_methode_rapide_3000_sur_t_s_ecarte_de_3_5_pour_cent_de_wien() -> None:
    """`lambda_max = 3000 / T` contre 2898 ; 98 % de l'energie entre `l/2` et `8 l`."""
    assert proches(3000.0 / 2898.0 - 1, 0.035, 0.001)
    t = 1000.0
    lm = 3000.0 * MICRO / t
    part = fraction_emise_sous(8 * lm, t) - fraction_emise_sous(lm / 2, t)
    assert proches(part, 0.98, 0.005)
