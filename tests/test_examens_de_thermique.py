"""Les examens de thermique : des resultats des corriges refaits par la bibliotheque.

Le releve `cursus/examens/ThermiqueFondements.toml` nomme les notions que
chaque sujet mobilise. Ce fichier refait, par les fonctions de
`polytech.thermique` et par sympy, les valeurs que les corriges officiels
ecrivent en clair -- l'oracle est le corrige, le chemin est celui de la
bibliotheque -- et les resultats que la fiche de revision derive elle-meme.

Les corriges : `Exam_2016_Aout_Exercice_cote_Solution`,
`Exam_2017_Mai_Exercice_cote_Solution` et
`Exam_2022_2025_Exercices_cotes_resolus`, dans
`ThermiqueFondements/EXAMENS/`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_exact,
)
from polytech.thermique.domain.convection.forcee import (
    coefficient_depuis_le_nusselt,
    nusselt_plaque,
)
from polytech.thermique.domain.convection.naturelle import nusselt_lame_de_fluide
from polytech.thermique.domain.convection.naturelle_externe import (
    nusselt_de_la_paroi_verticale,
    nusselt_de_la_plaque_horizontale,
)
from polytech.thermique.domain.echangeurs_a_courants_paralleles import Sens
from polytech.thermique.domain.rayonnement.corps_noir import (
    longueur_d_onde_du_maximum,
)
from polytech.thermique.domain.rayonnement.facteurs_globaux import (
    Bande,
    facteur_global_par_bandes,
)
from polytech.thermique.domain.rendement_d_echangeur import rendement

AIR_A_300_K = 0.02624
"""Conductivite de l'air a 300 K, W/(m K) : table du sujet d'aout 2016."""


# --- Aout 2016, « Retour de vacances » : le tableau de la section 1.6 ------


@pytest.mark.parametrize(("toit", "attendu"), [(303.0, 5.59), (353.0, 7.15)])
def test_le_coefficient_de_rayonnement_du_toit(toit: float, attendu: float) -> None:
    assert coefficient_de_rayonnement_exact(0.9, toit, 300.0) == pytest.approx(
        attendu, abs=0.01
    )


def test_le_nusselt_du_toit_qui_roule_est_turbulent() -> None:
    """Section 1.4 : `0,037 Re^0,8 Pr^1/3 = 6620` pour `Re = 4,25e6`."""
    assert nusselt_plaque(4.25e6, 0.707) == pytest.approx(6620, rel=1e-3)


def test_le_nusselt_du_toit_a_l_arret() -> None:
    """Le tableau donne `Gr = 5,6e10` et `Nu = 468`, plaque chaude vers le haut."""
    assert nusselt_de_la_plaque_horizontale(5.6e10 * 0.707) == pytest.approx(
        468, rel=5e-3
    )


def _flux_solaire_absorbe() -> float:
    """700 W/m2, absorbes a 0,5 sous 1,5 micrometre et a 0,9 au-dela."""
    bandes = (Bande(limite=1.5e-6, facteur=0.5), Bande(limite=math.inf, facteur=0.9))
    return 700.0 * facteur_global_par_bandes(bandes, 5800.0)


@pytest.mark.parametrize(
    ("rayonnement", "convection", "attendu"),
    [(5.59, 6620 * AIR_A_300_K / 2.0, 304.2), (7.15, 6.1, 328.8)],
)
def test_la_temperature_du_toit_par_le_bilan_au_noeud(
    rayonnement: float, convection: float, attendu: float
) -> None:
    """Le flux solaire absorbe part par rayonnement et par convection."""
    toit = 300.0 + _flux_solaire_absorbe() / (rayonnement + convection)
    assert toit == pytest.approx(attendu, abs=0.3)


# --- Mai 2017, refroidisseur de LED : la question subsidiaire ---------------


def test_a_debit_constant_la_puissance_suit_l_ecart_a_l_air() -> None:
    """Le corrige rend 350,14 W a -10 degres et 116,71 W a 30 degres.

    A debit de capacite constant l'efficacite ne bouge pas : la puissance
    est proportionnelle a l'ecart entre la paroi a 50 degres et l'air.
    """
    puissances = {-10.0: 350.14, 30.0: 116.71}
    par_degre = {air: flux / (50.0 - air) for air, flux in puissances.items()}
    assert par_degre[-10.0] == pytest.approx(par_degre[30.0], rel=1e-3)


# --- Mai 2022 a juin 2025, le recueil resolu ---------------------------------


def test_la_convection_naturelle_autour_de_la_bouteille() -> None:
    """Mai 2022 : `Gr Pr = 14 647 183`, `Nu = 34,64`, `K = 4,54 W/m2K`."""
    nusselt = nusselt_de_la_paroi_verticale(14_647_183)
    assert nusselt == pytest.approx(34.64, abs=0.01)
    assert coefficient_depuis_le_nusselt(nusselt, AIR_A_300_K, 0.20) == pytest.approx(
        4.54, abs=0.01
    )


def test_la_lame_d_air_de_la_marmite() -> None:
    """Mai 2023 : `Gr_b Pr = 129 172`, `Nu_b = 3,34`, `K = 2,006 W/m2K`."""
    nusselt = nusselt_lame_de_fluide(129_172)
    assert nusselt == pytest.approx(3.34, abs=0.01)
    assert coefficient_depuis_le_nusselt(nusselt, 0.03003, 0.05) == pytest.approx(
        2.006, abs=0.005
    )


def test_le_rayonnement_exterieur_de_la_caisse() -> None:
    """Mai 2023 : `K_ray,e = 5,30 W/m2K`, bois a 299,2 K face a 293,15 K."""
    assert coefficient_de_rayonnement_exact(0.9, 299.2, 293.15) == pytest.approx(
        5.30, abs=0.01
    )


@pytest.mark.parametrize(
    ("temperature", "rapide"), [(318.15, 9.43), (358.15, 8.38), (288.15, 10.41)]
)
def test_la_methode_rapide_surestime_le_maximum_de_wien(
    temperature: float, rapide: float
) -> None:
    """Les corriges ecrivent `lambda_max = 3000 / T` ; Wien donne 2898.

    L'ecart, de 3,5 pour cent, ne change aucune conclusion : la methode ne
    sert qu'a situer la bande utile par rapport a la coupure du verre.
    """
    assert 3000.0 / temperature == pytest.approx(rapide, abs=0.01)
    wien = longueur_d_onde_du_maximum(temperature) * 1e6
    assert rapide / wien == pytest.approx(3000 / 2898, rel=2e-3)


# --- Ce que la fiche derive elle-meme ---------------------------------------


def test_un_echangeur_a_paroi_isotherme_suit_une_exponentielle() -> None:
    """Article « L'efficacite d'un echangeur » : `1 - exp(-K S / q c)`."""
    for sens in Sens:
        assert rendement(sens, 1.3, 0.0) == pytest.approx(1 - math.exp(-1.3))


def test_le_rayon_critique_d_un_isolant() -> None:
    """Article « Le coefficient global » : `r = lambda / K` rend le flux maximal."""
    rayon, interne, conductivite, coefficient = sp.symbols(
        "r r_1 lambda K", positive=True
    )
    resistance = sp.log(rayon / interne) / (2 * sp.pi * conductivite) + 1 / (
        2 * sp.pi * rayon * coefficient
    )
    critique = sp.solve(sp.diff(resistance, rayon), rayon)
    assert critique == [conductivite / coefficient]
    assert sp.diff(resistance, rayon, 2).subs(rayon, critique[0]) > 0


def test_la_resistance_d_une_coque_spherique() -> None:
    """Article « La loi de Fourier », sur le reservoir d'aout 2022."""
    r, r1, r2, conductivite, t1, t2 = sp.symbols(
        "r r_1 r_2 lambda T_1 T_2", positive=True
    )
    champ = sp.Function("T")
    solution = sp.dsolve(
        sp.Eq(sp.diff(r**2 * sp.diff(champ(r), r), r), 0),
        champ(r),
        ics={champ(r1): t1, champ(r2): t2},
    ).rhs
    flux = -conductivite * sp.diff(solution, r) * 4 * sp.pi * r**2
    resistance = (r2 - r1) / (4 * sp.pi * conductivite * r1 * r2)
    assert sp.simplify(flux - (t1 - t2) / resistance) == 0


def test_la_barre_refroidie_a_un_profil_parabolique() -> None:
    """Article « La conduction avec generation », question de janvier 2022."""
    x, longueur, source, base = sp.symbols("x L S T_0", real=True)
    champ = sp.Function("t")
    solution = sp.dsolve(
        sp.Eq(sp.diff(champ(x), x, 2) + source, 0),
        champ(x),
        ics={champ(0): base, sp.diff(champ(x), x).subs(x, longueur): 0},
    ).rhs
    assert sp.simplify(solution - (base + source * (longueur * x - x**2 / 2))) == 0
