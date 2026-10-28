"""Les formules de conduction de la fiche de thermique, refaites par `polytech/`.

Les articles de `cursus/fiches/ThermiqueFondements.toml` sont tapes a la
main : chaque formule est ici ecrite comme la fiche l'ecrit, puis comparee
a ce que la bibliotheque calcule (`polytech.maths.domain.comparaison`).
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.maths.domain.comparaison import egales, proches_relativement
from polytech.thermique.domain.conduction.barre_isolee import profil_de_barre_isolee
from polytech.thermique.domain.conduction.fourier import (
    ESPACE,
    conductivite_moyenne,
    diffusivite,
    residu_de_fourier_kirchhoff,
)
from polytech.thermique.domain.conduction.parois import (
    Couche,
    Tube,
    flux_lineique_de_tubes,
    profil_de_plaque,
    profil_de_tube,
    temperatures_d_interface,
)
from polytech.thermique.domain.conduction.rayon_critique import rayon_critique
from polytech.thermique.domain.conduction.spherique import CoqueSpherique
from polytech.thermique.domain.continuite_du_flux import temperature_superficielle
from polytech.thermique.domain.convection.naturelle import (
    resistance_de_paroi_cylindrique,
    resistance_de_paroi_plane,
    resistance_de_surface,
)
from polytech.thermique.domain.geometrie import Rayon
from polytech.thermique.domain.materiaux import Conductivite
from polytech.thermique.domain.reseaux import en_parallele, en_serie

X, L, S, T0, R1, R2 = sp.symbols("x L S T_0 r_1 r_2", positive=True)


def test_le_reseau_donne_le_flux_par_la_somme_des_resistances() -> None:
    """`Phi = Dt / somme R`, `R_plan = e/(lambda S)`, `R_surface = 1/(K S)`."""
    plan = resistance_de_paroi_plane(0.2, 0.8, 3.0)
    surface = resistance_de_surface(10.0, 3.0)
    assert proches_relativement(plan, 0.2 / (0.8 * 3.0), 1e-12)
    assert proches_relativement(surface, 1 / (10.0 * 3.0), 1e-12)
    flux = 30.0 / en_serie(plan, surface)
    assert proches_relativement(flux, 30.0 / (plan + surface), 1e-12)


def test_au_noeud_les_flux_entrants_valent_les_flux_sortants() -> None:
    """Deux branches entre les memes temperatures : leurs flux s'ajoutent."""
    branche_1, branche_2 = 2.0, 3.0
    somme = 20.0 / branche_1 + 20.0 / branche_2
    assert proches_relativement(20.0 / en_parallele(branche_1, branche_2), somme, 1e-12)


def test_la_fausse_position_trouve_l_equilibre_des_flux() -> None:
    """Flux entrant = flux sortant : `t_p = (K_1 t_1 + K_2 t_2) / (K_1 + K_2)`."""
    k1, t1, k2, t2 = 12.0, 90.0, 5.0, 10.0
    trouvee = temperature_superficielle(
        lambda t: k1 * (t1 - t), lambda t: k2 * (t - t2), (t2, t1)
    )
    assert proches_relativement(trouvee, (k1 * t1 + k2 * t2) / (k1 + k2), 1e-9)


def test_l_equation_de_fourier_kirchhoff_admet_le_profil_avec_source() -> None:
    """`dt/dtau = a lap t + q/(rho c)`, profil `t = -q x^2/(2 lambda)`."""
    lam, rho_c, q = sp.symbols("lambda rho_c q", positive=True)
    assert proches_relativement(
        diffusivite(0.6, 1000.0, 4000.0), 0.6 / (1000 * 4000), 1e-12
    )
    champ = -q * ESPACE[0] ** 2 / (2 * lam)
    assert egales(residu_de_fourier_kirchhoff(champ, lam / rho_c, source=q / rho_c), 0)


def test_la_coque_spherique_a_la_resistance_de_la_fiche() -> None:
    """`R = (r_2 - r_1) / (4 pi lambda r_1 r_2)`."""
    coque = CoqueSpherique(Conductivite(0.04), Rayon(0.5), Rayon(0.8))
    fiche = (0.8 - 0.5) / (4 * math.pi * 0.04 * 0.5 * 0.8)
    assert proches_relativement(coque.resistance().kelvins_par_watt, fiche, 1e-12)


def test_la_plaque_et_le_tube_suivent_les_profils_de_la_fiche() -> None:
    """`R_tube = ln(r_2/r_1)/(2 pi lambda L)` ; `t(r)` en `ln r`, `t(x)` affine."""
    tube = resistance_de_paroi_cylindrique(0.02, 0.03, 50.0, 4.0)
    assert proches_relativement(tube, math.log(1.5) / (2 * math.pi * 50.0 * 4.0), 1e-12)
    t1, t2 = sp.symbols("t_1 t_2", real=True)
    r = sp.Symbol("r", positive=True)
    fiche = t1 + (t2 - t1) * sp.log(r / R1) / sp.log(R2 / R1)
    assert egales(profil_de_tube(r, (R1, R2), (t1, t2)), fiche)
    assert egales(profil_de_plaque(X, L, t1, t2), t1 + (t2 - t1) * X / L)


def test_des_couches_accolees_sont_des_resistances_en_serie() -> None:
    """`e/lambda` s'ajoutent : la densite de flux vaut `Dt / somme`."""
    couches = [Couche(0.1, 0.5), Couche(0.05, 0.1)]
    _, flux = temperatures_d_interface(couches, 100.0, 20.0)
    assert proches_relativement(flux, 80.0 / (0.1 / 0.5 + 0.05 / 0.1), 1e-12)


def test_la_conductivite_moyenne_est_celle_de_la_temperature_moyenne() -> None:
    """Loi `a + b t` : moyenne sur `[t_2, t_1]` = `a + b (t_1 + t_2)/2`."""
    a, b, t = sp.symbols("a b t", real=True)
    t1, t2 = sp.symbols("t_1 t_2", real=True)
    moyenne = sp.integrate(a + b * t, (t, t2, t1)) / (t1 - t2)
    assert egales(conductivite_moyenne(a, b, t1, t2), moyenne)


def test_le_rayon_critique_est_lambda_sur_k() -> None:
    """`r_c = lambda / K_e` : le flux par metre est maximal quand `r_e = r_c`."""
    lam, k = 0.04, 8.0
    critique = rayon_critique(lam, k)
    assert proches_relativement(critique, lam / k, 1e-12)

    def flux(re: float) -> float:
        return flux_lineique_de_tubes([Tube(0.002, re, lam)], 100.0, 20.0, exterieur=k)

    assert flux(critique) > flux(critique * 0.9)
    assert flux(critique) > flux(critique * 1.5)


def test_la_barre_refroidie_a_le_profil_de_la_fiche() -> None:
    """`t'' + S = 0`, `t(0) = T_0`, `t'(L) = 0` : `t = T_0 + S (L x - x^2/2)`."""
    profil = profil_de_barre_isolee(X, L, T0, S)
    assert egales(profil, T0 + S * (L * X - X**2 / 2))
    assert egales(sp.diff(profil, X, 2) + S, 0)
    assert egales(sp.diff(profil, X).subs(X, L), 0)
