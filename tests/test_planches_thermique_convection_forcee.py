"""Les planches de thermique : la convection forcee (jeu 09).

Source : `ThermiqueFondements/SLIDES/Slides_09_Convection_forcee.pdf`, 34
planches lues sur les pages rendues. Elles reprennent les chapitres 1 a 3
du tome 2 ; ce fichier eprouve ce que les planches chiffrent elles-memes.

Les oracles : les exposants des trois groupes (planches 17 et 20), les
correlations des planches 26, 28, 29 et 32, et la synthese de la
planche 33.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.intensite_de_turbulence import intensite_de_turbulence
from polytech.fluides.domain.turbulence import Decomposition
from polytech.thermique.domain.convection.couches_limites import (
    REYNOLDS_CRITIQUE_SUR_PLAQUE,
    angle_de_decollement,
    longueur_d_etablissement,
)
from polytech.thermique.domain.convection.definitions import densite_de_flux_turbulente
from polytech.thermique.domain.convection.faisceau_de_tubes import vitesse_maximale
from polytech.thermique.domain.convection.forcee import (
    TABLE_DU_CYLINDRE,
    nusselt_cylindre,
    nusselt_plaque,
    prandtl,
)
from polytech.thermique.domain.convection.naturelle import grashof
from polytech.thermique.domain.convection.plaque_plane import (
    rapport_des_couches_limites,
)
from polytech.thermique.domain.convection.similitude import (
    nusselt,
    prandtl_des_proprietes,
    stanton,
    temperature_de_film,
    viscosite_pres_du_melange,
)
from polytech.thermique.domain.convection.synthese_forcee import Correlation
from polytech.thermique.domain.convection.tube_laminaire import (
    GROUPE_MINIMAL,
    NUSSELT_ETABLI,
    groupe_d_entree,
    longueur_d_entree_thermique,
    nusselt_de_sieder_et_tate,
)
from polytech.thermique.domain.convection.tube_turbulent import (
    facteur_j,
    nusselt_de_colburn,
    nusselt_turbulent_de_sieder_et_tate,
)

V, X, Z, Q = sp.symbols("v x z q")


def _exposants(y: int, r: int, s: int, *, naturelle: bool) -> dict[sp.Symbol, sp.Expr]:
    """Les quatre equations aux dimensions `Q`, `L`, `Theta`, `T` de la planche."""
    temps = -V - (2 * y if naturelle else y) - Z - r - s
    theta = (-y if naturelle else 0) + Q - s
    equations = (V + s, -2 * V + X + y + 2 * Z + 2 * r - s, theta, temps)
    exposants: dict[sp.Symbol, sp.Expr] = sp.solve(equations, (V, X, Z, Q), dict=True)[
        0
    ]
    return exposants


def test_les_exposants_des_trois_groupes_de_la_convection_forcee() -> None:
    """Planche 17 : `u_0 l / nu`, puis `a / nu` et `lambda theta_0 / (phi l)`."""
    assert _exposants(1, 0, 0, naturelle=False) == {V: 0, X: 1, Z: -1, Q: 0}
    assert _exposants(0, 1, 0, naturelle=False) == {V: 0, X: 0, Z: -1, Q: 0}
    assert _exposants(0, 0, 1, naturelle=False) == {V: -1, X: -1, Z: 0, Q: 1}


def test_le_nombre_de_grashof_sort_des_memes_equations() -> None:
    """Planche 20 : `Gr = beta g theta_0 l^3 / nu^2`."""
    assert _exposants(1, 0, 0, naturelle=True) == {V: 0, X: 3, Z: -2, Q: 1}


def test_la_signification_physique_des_groupes() -> None:
    """Planche 21 : `Nu = K l / lambda`, `Pr = nu / a = mu c / lambda`.

    Et Grashof : forces d'Archimede sur forces de viscosite, `Delta rho g l
    / (mu u_0 / l)` avec la vitesse `u_0 = nu / l`.
    """
    assert nusselt(25.0, 0.1, 0.5) == pytest.approx(25.0 * 0.1 / 0.5)
    nu, rho, c, conductivite = 1.5e-5, 1.2, 1005.0, 0.026
    assert prandtl_des_proprietes(nu * rho, c, conductivite) == pytest.approx(
        prandtl(nu, conductivite / (rho * c))
    )
    beta, theta, longueur = 1 / 300.0, 40.0, 0.2
    archimede = rho * beta * theta * 9.81 * longueur
    viscosite = nu * rho * (nu / longueur) / longueur
    assert archimede / viscosite == pytest.approx(grashof(beta, theta, longueur, nu))


def test_la_turbulence_multiplie_le_flux() -> None:
    """Planches 9 et 10 : `f = f_moyen + f'`, `phi_y = -lambda dt/dy + rho c v't'`."""
    vitesse = Decomposition((0.2, -0.2, 0.2, -0.2))
    temperature = Decomposition((51.0, 49.0, 51.0, 49.0))
    assert sum(vitesse.fluctuations()) == pytest.approx(0.0)
    conduit = densite_de_flux_turbulente(
        1.2,
        1005.0,
        Decomposition((0.0,) * 4),
        temperature,
        conductivite=0.026,
        gradient=-100.0,
    )
    agite = densite_de_flux_turbulente(
        1.2, 1005.0, vitesse, temperature, conductivite=0.026, gradient=-100.0
    )
    assert conduit == pytest.approx(2.6)
    assert agite / conduit == pytest.approx(1 + 1.2 * 1005.0 * 0.2 / 2.6)
    assert agite > 90 * conduit


def test_l_intensite_de_turbulence() -> None:
    """Planche 22 : `J = racine((v_x'^2 + v_y'^2 + v_z'^2) / 3) / u_0`."""
    composantes = (
        Decomposition((10.3, 9.7, 10.3, 9.7)),
        Decomposition((0.4, -0.4, 0.4, -0.4)),
        Decomposition((0.0, 0.0, 0.0, 0.0)),
    )
    attendue = math.sqrt((0.09 + 0.16 + 0.0) / 3) / 10.0
    assert intensite_de_turbulence(composantes, 10.0) == pytest.approx(attendue)
    with pytest.raises(ValueError, match="vitesse de reference nulle ou negative"):
        intensite_de_turbulence(composantes, 0.0)


def test_les_references_de_la_planche_23() -> None:
    """Film `(t_p + t_0) / 2` ; faisceau : `u_max` dans la section minimale."""
    assert temperature_de_film(80.0, 20.0) == pytest.approx(50.0)
    assert vitesse_maximale(6.0, pas=0.038, diametre=0.025) == pytest.approx(
        6.0 * 38 / 13
    )


def test_la_viscosite_lineaire_entre_le_fluide_et_la_paroi() -> None:
    """Planche 24 : `mu / mu_0 = 1 + (mu_p / mu_0 - 1)(t - t_0) / (t_p - t_0)`."""
    reference, paroi = (40.0, 8e-3), (100.0, 2e-3)
    locale = viscosite_pres_du_melange(55.0, melange=reference, paroi=paroi)
    attendue = 1 + (2e-3 / 8e-3 - 1) * (55.0 - 40.0) / (100.0 - 40.0)
    assert locale / reference[1] == pytest.approx(attendue)


def test_la_plaque_plane() -> None:
    """Planche 26 : `0,664 Re^0,5 Pr^1/3`, `0,037 Re^0,8 Pr^1/3`, `delta_t / delta`."""
    assert nusselt_plaque(1e5, 0.7) == pytest.approx(0.664 * 1e5**0.5 * 0.7 ** (1 / 3))
    assert nusselt_plaque(1e6, 0.7) == pytest.approx(0.037 * 1e6**0.8 * 0.7 ** (1 / 3))
    assert REYNOLDS_CRITIQUE_SUR_PLAQUE == (3e5, 5e5)
    assert rapport_des_couches_limites(8.0) == pytest.approx(0.5)
    assert rapport_des_couches_limites(0.7) > 1 > rapport_des_couches_limites(7.0)


def test_l_etablissement_dans_un_tube_laminaire() -> None:
    """Planche 28 : `L_t / L_h = 1,7 Pr` et `Nu = 3,65` une fois etabli.

    Le tome 2 donne `L_h = 0,03 Re d` et `L_t = 0,05 Re Pr d` : leur
    rapport, `1,67 Pr`, est bien celui de la planche.
    """
    hydraulique = longueur_d_etablissement(0.02, 1000.0)
    thermique = longueur_d_entree_thermique(0.02, 1000.0, 5.0)
    assert thermique / hydraulique == pytest.approx(1.7 * 5.0, rel=0.02)
    assert pytest.approx(3.65) == NUSSELT_ETABLI


def test_la_condition_de_sieder_et_tate_porte_sur_l_sur_d() -> None:
    """Planche 28 : « si (1 / Re Pr)(d / L) < 0,1 », `Nu = 1,86 (Re Pr d / L)^1/3`.

    La condition est ecrite avec `d / L` ; ainsi lue, elle vaut pour tout
    tube. C'est `(L / d) / (Re Pr) < 0,1` : l'abscisse de la figure de la
    meme planche, ou le regime s'etablit a 0,05. Le syllabus porte la
    meme inversion a sa relation (3.7).
    """
    groupe = groupe_d_entree(1000.0, 5.0, 0.02, 2.0)
    assert groupe == pytest.approx(50.0)
    assert 1 / groupe < 1 / GROUPE_MINIMAL
    assert nusselt_de_sieder_et_tate(groupe, viscosites=2.0) == pytest.approx(
        1.86 * 50.0 ** (1 / 3) * 2.0**0.14
    )
    etabli = groupe_d_entree(1000.0, 5.0, 0.02, 2000.0)
    assert (0.02 / 2000.0) / 5000.0 < 1 / GROUPE_MINIMAL < 1 / etabli


def test_le_tube_turbulent() -> None:
    """Planche 29 : `0,0234 Re^0,8 Pr^1/3`, ou 0,027 avec `(mu_m / mu_p)^0,14`."""
    assert nusselt_de_colburn(5e4, 3.0) == pytest.approx(
        0.0234 * 5e4**0.8 * 3.0 ** (1 / 3)
    )
    assert nusselt_turbulent_de_sieder_et_tate(
        5e4, 3.0, viscosites=1.5
    ) == pytest.approx(0.027 * 5e4**0.8 * 3.0 ** (1 / 3) * 1.5**0.14)


def test_le_tube_isole_en_ecoulement_transversal() -> None:
    """Planches 31 et 32 : decollement vers 80 puis 110 degres ; le tableau `C`, `n`.

    La planche situe le changement a `Re_d = 10^5` et arrondit l'angle a
    80 degres ; le tome 2 (pages 18 et 19) ecrit 82 degres et `4 10^5`, que la
    bibliotheque retient.
    """
    assert angle_de_decollement(5e4) == pytest.approx(80.0, abs=2.5)
    assert angle_de_decollement(5e5) == pytest.approx(110.0)
    assert angle_de_decollement(2e5) < 110.0
    assert TABLE_DU_CYLINDRE == (
        (0.4, 4.0, 0.989, 0.330),
        (4.0, 40.0, 0.911, 0.385),
        (40.0, 4000.0, 0.683, 0.466),
        (4000.0, 40000.0, 0.193, 0.618),
        (40000.0, 400000.0, 0.0266, 0.805),
    )
    assert nusselt_cylindre(1e4, 0.7) == pytest.approx(
        0.193 * 1e4**0.618 * 0.7 ** (1 / 3)
    )


COLBURN = Correlation(coefficient=0.0234, exposant=0.8)


def test_la_synthese_les_effets_de_la_vitesse_et_de_la_dimension() -> None:
    """Planche 33 : `K ~ u_0^n` avec `n < 1`, `K ~ l^(n - 1)` avec `n - 1 < 0`."""
    assert COLBURN.nusselt(5e4, 3.0) == pytest.approx(nusselt_de_colburn(5e4, 3.0))
    assert COLBURN.effet_de_la_vitesse(2.0) == pytest.approx(2.0**0.8)
    assert 1 < COLBURN.effet_de_la_vitesse(2.0) < 2.0
    assert COLBURN.effet_de_la_dimension(2.0) == pytest.approx(2.0**-0.2)
    assert COLBURN.effet_de_la_dimension(2.0) < 1
    double = COLBURN.nusselt(1e5, 3.0) / 2.0
    assert double / COLBURN.nusselt(5e4, 3.0) == pytest.approx(
        COLBURN.effet_de_la_dimension(2.0)
    )
    with pytest.raises(ValueError, match="exposant de Reynolds hors de"):
        Correlation(coefficient=0.0234, exposant=1.2)
    with pytest.raises(ValueError, match="rapport nul ou negatif"):
        COLBURN.effet_de_la_vitesse(0.0)


def test_la_synthese_le_facteur_j() -> None:
    """Planche 33 : `j_H = Nu / (Re Pr^1/3) = K Pr^2/3 / (rho u_0 c) = C Re^(n - 1)`."""
    reynolds, nombre_de_prandtl = 5e4, 3.0
    valeur = COLBURN.nusselt(reynolds, nombre_de_prandtl)
    assert COLBURN.facteur_j(reynolds) == pytest.approx(0.0234 * reynolds**-0.2)
    assert COLBURN.facteur_j(reynolds) == pytest.approx(
        facteur_j(valeur, reynolds, nombre_de_prandtl)
    )
    de_stanton = stanton(valeur, reynolds, nombre_de_prandtl)
    assert COLBURN.facteur_j(reynolds) == pytest.approx(
        de_stanton * nombre_de_prandtl ** (2 / 3)
    )
