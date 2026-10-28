"""Les valeurs ecrites dans le texte de la fiche de fluides viennent de la lib.

Chaque nombre cite par un article est recalcule par `polytech.fluides`,
mis en forme comme la fiche l'ecrit, et cherche dans le texte du TOML :
un chiffre modifie a la main, ou une lib qui change, fait echouer.
"""

from __future__ import annotations

import math

import pytest

from polytech.fluides.domain.aerodynamique import Ecoulement, SurfacePortante
from polytech.fluides.domain.compressibilite import variation_relative_a_l_arret
from polytech.fluides.domain.conduite import (
    COEFFICIENT_DE_BLASIUS,
    coefficient_de_blasius,
)
from polytech.fluides.domain.fondamentales.continuite import vitesse_dans_la_section
from polytech.fluides.domain.grandeurs import PESANTEUR, MasseVolumique
from polytech.fluides.domain.regime_de_reynolds import (
    REYNOLDS_LAMINAIRE,
    REYNOLDS_TURBULENT,
    nombre_de_reynolds,
)
from polytech.fluides.domain.regimes import (
    GAMMA_DE_L_AIR,
    MACH_INCOMPRESSIBLE,
    VARIATION_ADMISE,
)
from polytech.fluides.domain.statique.gaz import AIR, CONSTANTE_DES_GAZ
from polytech.fluides.domain.turbulence import CONSTANTE_LOG, KARMAN
from tests.fiches_toml import texte

F = "MecaniqueFluides"
NU_AIR = 1.8e-5 / 0.9


def _virgule(x: float) -> str:
    return f"{x:g}".replace(".", "{,}")


def _mach_des_cinq_pour_cent() -> float:
    bas, haut = 0.0, 1.0
    for _ in range(60):
        milieu = (bas + haut) / 2
        if variation_relative_a_l_arret(milieu) < VARIATION_ADMISE:
            bas = milieu
        else:
            haut = milieu
    return bas


def _toboggan() -> tuple[float, float, float]:
    vitesse = vitesse_dans_la_section(10.6e-3, math.pi * 0.15**2 / 4)
    reynolds = nombre_de_reynolds(vitesse, 0.15, 1e-3 / 1000.0)
    return vitesse, reynolds, coefficient_de_blasius(reynolds)


def _attendus() -> list[tuple[str, str]]:
    vitesse, reynolds, lam = _toboggan()
    rapport = nombre_de_reynolds(250.0, 1.0, NU_AIR) / nombre_de_reynolds(
        30.0, 0.1, NU_AIR
    )
    portance = SurfacePortante(Ecoulement(MasseVolumique(1.225), 12.0), 30.0).force(1.4)
    celerite = math.sqrt(
        GAMMA_DE_L_AIR * CONSTANTE_DES_GAZ / AIR.masse_molaire * 293.15
    )
    similitude, bernoulli = (
        "similitude_complete",
        "equation_de_bernoulli_et_sa_generalisation",
    )
    return [
        (similitude, f"\\approx {rapport:.0f}"),
        (similitude, f"\\qty{{{250.0 * 10.0:.0f}}}"),
        (bernoulli, f"\\qty{{{vitesse:.2f}}}"),
        (bernoulli, f"\\num{{{reynolds:.1e}".replace("e+0", "e")),
        (bernoulli, f"\\num{{{lam:.3f}}}"),
        (bernoulli, f"{_virgule(COEFFICIENT_DE_BLASIUS)}"),
        (
            "regimes_laminaire_et_turbulent",
            f"\\num{{{reynolds:.1e}".replace("e+0", "e"),
        ),
        ("regimes_laminaire_et_turbulent", f"{REYNOLDS_LAMINAIRE:.0f}"),
        ("regimes_laminaire_et_turbulent", f"{REYNOLDS_TURBULENT:.0f}"),
        (
            "portance_et_trainee_efforts_de_surface",
            f"\\qty{{{portance:.1e}}}".replace("e+0", "e"),
        ),
        (
            "portance_et_trainee_efforts_de_surface",
            f"\\qty{{{portance / PESANTEUR:.1e}".replace("e+0", "e"),
        ),
        (
            "autres_classifications_des_ecoulements",
            f"\\qty{{{100 * variation_relative_a_l_arret(MACH_INCOMPRESSIBLE):.1f}}}",
        ),
        (
            "autres_classifications_des_ecoulements",
            f"Ma \\approx {_virgule(round(_mach_des_cinq_pour_cent(), 2))}",
        ),
        (
            "autres_classifications_des_ecoulements",
            f"\\qty{{{round(MACH_INCOMPRESSIBLE * celerite, -2):.0f}}}",
        ),
        (
            "ecoulement_turbulent_en_conduite_et_loi_de_paroi",
            f"\\kappa = {_virgule(KARMAN)}",
        ),
        (
            "ecoulement_turbulent_en_conduite_et_loi_de_paroi",
            f"B = {CONSTANTE_LOG:.1f}".replace(".", "{,}"),
        ),
    ]


@pytest.mark.parametrize(("notion", "fragment"), _attendus())
def test_le_texte_porte_la_valeur_que_la_lib_calcule(
    notion: str, fragment: str
) -> None:
    assert fragment in texte(F, notion)


def test_la_pression_en_a_ecrite_est_celle_de_bernoulli() -> None:
    vitesse = _toboggan()[0]
    pression = 1.0e5 + 1000.0 * PESANTEUR * 1.5 - 1000.0 * vitesse**2 / 2
    assert f"\\qty{{{pression / 1e5:.1f}}}{{\\bar}}" in texte(
        F, "equation_de_bernoulli_et_sa_generalisation"
    )
