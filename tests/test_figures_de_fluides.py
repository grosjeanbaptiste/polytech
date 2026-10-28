"""Les figures calculees de la fiche de fluides, relues par la lib.

Chaque courbe, graduation, pente ou rapport que le TikZ de
`cursus/fiches/MecaniqueFluides.toml` affiche est lu dans le TOML puis
compare a `polytech.fluides`.
"""

from __future__ import annotations

import math
import re

import mpmath
import pytest
import sympy as sp

from polytech.fluides.domain.compressibilite import (
    MACH_HYPERSONIQUE,
    MACH_SONIQUE,
    RegimeDeMach,
    regime_de_mach,
)
from polytech.fluides.domain.conduite import profil_de_poiseuille
from polytech.fluides.domain.fondamentales.navier_stokes import profil_de_couette
from polytech.fluides.domain.loi_de_puissance import (
    EXPOSANT_DE_BLASIUS,
    rapport_vitesse_moyenne_sur_maximale,
    vitesse_en_loi_de_puissance,
)
from polytech.fluides.domain.poiseuille_plan import profil_plan
from polytech.fluides.domain.regime_de_reynolds import (
    REYNOLDS_LAMINAIRE,
    REYNOLDS_TURBULENT,
)
from polytech.fluides.domain.regimes import MACH_INCOMPRESSIBLE
from polytech.fluides.domain.trainee_de_la_sphere import (
    REYNOLDS_CRITIQUE_BALLE_DE_GOLF,
    coefficient_de_trainee_de_la_sphere,
)
from polytech.fluides.domain.turbulence import (
    DEBUT_SOUS_COUCHE_LOG,
    LIMITE_SOUS_COUCHE_VISQUEUSE,
    Decomposition,
    vitesse_adimensionnee,
)
from polytech.maths.domain.comparaison import egales, proches, proches_relativement
from tests.fiches_toml import courbes, couples, expression, figure, nombres

F = "MecaniqueFluides"
REYNOLDS = "moyenne_de_reynolds_et_tenseur_de_reynolds"
PAROI = "ecoulement_turbulent_en_conduite_et_loi_de_paroi"
REGIMES = "regimes_laminaire_et_turbulent"
TRAINEE = "trainee_en_pratique"
MACH = "autres_classifications_des_ecoulements"
NAVIER = "equations_de_navier_stokes_couette_et_film_tombant"
POISEUILLE = "ecoulement_de_poiseuille_et_pertes_de_charge"
COUVERTES = {
    REYNOLDS,
    PAROI,
    REGIMES,
    TRAINEE,
    MACH,
    NAVIER,
    POISEUILLE,
    "similitude_complete",
}
Y = sp.Symbol("y", real=True)
R = sp.Symbol("r", real=True)


def test_le_signal_a_pour_moyenne_le_niveau_de_la_droite_rouge() -> None:
    signal, niveau = courbes(figure(F, REYNOLDS))
    f = sp.lambdify(sp.Symbol("x", real=True), expression(signal), "math")
    u = Decomposition(tuple(f(10 * k / 20000) for k in range(20001)))
    assert proches(u.moyenne(), float(niveau), 0.01)
    assert proches(Decomposition(u.fluctuations()).moyenne(), 0.0, 1e-9)


def test_la_loi_de_paroi_tracee_est_celle_de_la_lib_dans_ses_deux_zones() -> None:
    tikz = figure(F, PAROI)
    bornes = re.findall(r"domain=(\d+):(\d+)", tikz)
    lineaire, log = courbes(tikz)
    assert [(float(a), float(b)) for a, b in bornes] == [
        (1.0, LIMITE_SOUS_COUCHE_VISQUEUSE),
        (DEBUT_SOUS_COUCHE_LOG, 1000.0),
    ]
    x = sp.Symbol("x", real=True)
    for y in (1.0, 2.5, 4.9):
        assert proches(
            float(expression(lineaire).subs(x, y)), vitesse_adimensionnee(y), 1e-12
        )
    for y in (30.0, 100.0, 1000.0):
        assert proches(
            float(expression(log).subs(x, y)), vitesse_adimensionnee(y), 1e-12
        )


def test_la_maquette_est_au_dixieme_et_exige_dix_fois_la_vitesse() -> None:
    tikz = figure(F, "similitude_complete")
    echelle = 1 / nombres(re.search(r"L/(\d+)", tikz).group(0))[0]  # type: ignore[union-attr]
    facteur_re = nombres(re.search(r"V_m = (\d+)\\,V", tikz).group(0))[0]  # type: ignore[union-attr]
    nu, longueur, vitesse = 1.5e-5, 2.0, 7.0
    re_avion = vitesse * longueur / nu
    v_maquette = re_avion * nu / (echelle * longueur)
    assert proches_relativement(v_maquette / vitesse, facteur_re, 1e-12)
    assert "V_m = V$" in tikz


def test_les_profils_laminaire_et_turbulent_sont_ceux_de_la_lib() -> None:
    tikz = figure(F, REGIMES)
    parabole, turbulent = (
        expression(c, "y") for c in re.findall(r"plot \(\{1 \+ ([^}]*)\}", tikz)
    )
    umax_lam, umax_turb = nombres(
        re.findall(r"plot \(\{1 \+ ([0-9.]+)", tikz).__str__()
    )
    r = sp.Integer(1)
    lib = profil_de_poiseuille(R, r, r, 4 * r)
    assert egales(parabole.subs(Y, R + 1), umax_lam * lib / lib.subs(R, 0))
    assert nombres(re.search(r"\^\(1/(\d+)\)", tikz).group(0))[0] == EXPOSANT_DE_BLASIUS  # type: ignore[union-attr]
    for y in (0.2, 0.5, 0.9, 1.3, 1.8):
        distance = 1 - abs(y - 1)
        attendu = vitesse_en_loi_de_puissance(distance, 1.0, umax_turb)
        assert proches(float(turbulent.subs(Y, y)), attendu, 1e-12)
    moyenne = 2 * mpmath.quad(lambda q: q * umax_turb * (1 - q) ** (1 / 7), [0, 1])
    assert proches_relativement(
        float(moyenne) / umax_turb, rapport_vitesse_moyenne_sur_maximale(), 1e-9
    )
    seuils = nombres(re.search(r"< (\d+) ou > (\d+)", tikz).group(0))
    assert seuils == [REYNOLDS_LAMINAIRE, REYNOLDS_TURBULENT]


def test_la_courbe_de_la_sphere_est_celle_de_la_correlation_et_la_crise_de_la_balle() -> (
    None
):
    sphere, crise = (couples(c) for c in courbes(figure(F, TRAINEE)))
    for re_, cd in sphere:
        assert proches_relativement(cd, coefficient_de_trainee_de_la_sphere(re_), 1e-3)
    assert {x for x, _ in crise} == {REYNOLDS_CRITIQUE_BALLE_DE_GOLF}
    assert max(x for x, _ in sphere) == 1e6


def test_les_graduations_de_l_axe_des_mach_sont_les_bornes_de_la_lib() -> None:
    tikz = figure(F, MACH)
    paires = re.search(
        r"\\foreach \\x/\\t in \{([^}]*(?:\{[^}]*\}[^}]*)*)\}", tikz
    ).group(1)  # type: ignore[union-attr]
    graduations = [
        (float(x), float(t.replace("{,}", ".")))
        for x, t in (p.split("/") for p in re.split(r", (?=\d)", paires))
    ]
    assert [m for _, m in graduations] == [
        0.0,
        MACH_INCOMPRESSIBLE,
        MACH_SONIQUE,
        MACH_HYPERSONIQUE,
    ]
    assert [x for x, _ in graduations] == sorted(x for x, _ in graduations)
    noms = [r.value[:6] for r in RegimeDeMach]
    textes = re.findall(r"\{(incompr|subson|supers|hyper)", tikz)
    assert [t[:6] for t in textes] == [n[:6] for n in noms]
    for mach, regime in (
        (0.1, RegimeDeMach.INCOMPRESSIBLE),
        (0.6, RegimeDeMach.SUBSONIQUE),
        (3.0, RegimeDeMach.SUPERSONIQUE),
        (8.0, RegimeDeMach.HYPERSONIQUE),
    ):
        assert regime_de_mach(mach) is regime


def test_couette_est_lineaire_et_poiseuille_plan_est_la_parabole_de_la_lib() -> None:
    tikz = figure(F, NAVIER)
    hauteur = 1.6
    couette = [
        float(v)
        for v in re.search(r"\\foreach \\y in \{([^}]*)\}", tikz).group(1).split(",")
    ]  # type: ignore[union-attr]
    h, u0, y = sp.symbols("h U y", positive=True)
    lib = profil_de_couette().subs({sp.Symbol("h", positive=True): h})
    assert couette[-1] == hauteur
    for ordonnee in couette:
        assert proches(
            ordonnee / hauteur,
            float(
                (lib / u0).subs(
                    {h: hauteur, sp.Symbol("y", real=True): ordonnee, u0: 1}
                )
            ),
            1e-12,
        )
    echelle, ordonnees = 2.5, [0.2, 0.5, 0.8, 1.1, 1.4]
    assert "2.5*\\y*(1.6 - \\y)/1.6" in tikz
    for o in ordonnees:
        ref = profil_plan(sp.Float(o), sp.Float(hauteur), sp.Integer(1), sp.Integer(1))
        assert proches(
            echelle * o * (hauteur - o) / hauteur,
            float(ref) * echelle * 2 / hauteur,
            1e-12,
        )


def test_le_profil_de_la_conduite_et_la_pression_lineaire_de_la_figure() -> None:
    tikz = figure(F, POISEUILLE)
    assert "0.6 + 2.6*(1 - (\\y - 1)^2)" in tikz
    r = sp.Integer(1)
    lib = profil_de_poiseuille(R, r, r, 4 * r)
    figure_u = 2.6 * (1 - R**2)
    assert egales(figure_u / lib, 2.6)
    (x1, y1), (x2, y2) = couples(
        re.search(r"\\draw\[red, thick\] (\(.*?\)) -- (\(.*?\));", tikz).expand(
            r"\1 \2"
        )
    )  # type: ignore[union-attr]
    assert x2 > x1
    assert y2 < y1


@pytest.mark.parametrize("valeur", [0.5, 1.0, 3.0])
def test_placeholder_pour_la_pression(valeur: float) -> None:
    assert math.isfinite(valeur)
