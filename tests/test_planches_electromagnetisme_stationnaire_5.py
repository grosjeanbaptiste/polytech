"""Planches d'electromagnetisme, chapitre III.5 : les milieux aimantes.

Source : `Electromagnetisme/SLIDES/Slides_03_Etats_stationnaires_partie_5.pdf`,
planches 2 a 23, lues sur les pages rendues.

Le fascicule resume le chapitre 10 du syllabus, et il demontre ce que le
syllabus affirme : l'aimantation diamagnetique est opposee a l'induction
(planches 4 et 5), un couple `m ^ B` aligne les moments (planche 6), et
le champ d'un tore bobine est nul au-dehors (planche 21).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.forces import (
    couple_sur_un_dipole,
)
from polytech.electromagnetisme.domain.magnetostatique.aimantation import (
    moment_amperien,
)
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    Magnetisme,
    classe_de_magnetisme,
)
from polytech.electromagnetisme.domain.magnetostatique.courbes_d_aimantation import (
    aire_du_cycle,
    champ_coercitif,
    induction_remanente,
    pente_asymptotique,
    permeabilite_initiale,
)
from polytech.electromagnetisme.domain.magnetostatique.diamagnetisme import (
    couple_sur_un_moment,
    ecart_de_larmor,
    moment_induit,
    pulsation_sous_induction,
)
from polytech.electromagnetisme.domain.magnetostatique.forces_magnetiques import (
    couple_par_l_energie,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import courant_enlace
from polytech.electromagnetisme.domain.magnetostatique.milieux_aimantes import (
    CORPS_DIAMAGNETIQUES,
    CORPS_PARAMAGNETIQUES,
    TEMPERATURE_DE_CURIE_DU_FER,
    est_ferromagnetique,
    induction_du_milieu,
)
from polytech.electromagnetisme.domain.magnetostatique.permeabilites import (
    champ_du_maximum_de_permeabilite,
    permeabilite_normale,
)
from polytech.electromagnetisme.domain.magnetostatique.reluctances import (
    flux_utile_et_de_dispersion,
)
from polytech.electromagnetisme.domain.magnetostatique.tore import (
    champ_du_tore_parfait,
)

CHARGE, MASSE, RAYON = sp.symbols("e m_e r", positive=True)
PULSATION, INDUCTION = sp.symbols("omega B", positive=True)


def test_l_induction_accelere_l_electron_de_e_b_sur_deux_m() -> None:
    """Planche 4 : `m_e omega^2 r + e v B = m_e omega_B^2 r`, d'ou `Delta omega`.

    L'equilibre est resolu exactement, avec `v = omega_B r` ; la planche
    prend `v = omega r` et `omega_B + omega = 2 omega`. Les deux
    s'accordent au premier ordre en `B`.
    """
    exacte = pulsation_sous_induction(PULSATION, CHARGE, INDUCTION, MASSE)
    equilibre = MASSE * PULSATION**2 * RAYON + CHARGE * exacte * RAYON * INDUCTION
    assert sp.simplify(equilibre - MASSE * exacte**2 * RAYON) == 0
    premier_ordre = sp.series(exacte - PULSATION, INDUCTION, 0, 2).removeO()
    larmor = ecart_de_larmor(CHARGE, INDUCTION, MASSE)
    assert sp.simplify(premier_ordre - larmor) == 0
    assert larmor == CHARGE * INDUCTION / (2 * MASSE)


def test_un_electron_sans_masse_n_a_pas_de_pulsation() -> None:
    with pytest.raises(ValueError, match="masse nulle ou negative"):
        ecart_de_larmor(1.6e-19, 1.0, 0.0)


def test_le_moment_induit_est_oppose_a_l_induction() -> None:
    """Planche 5 : `Delta m = -e^2 B r^2 / (4 m_e) u_n`.

    Recoupe par le moment amperien de l'orbite, `m = -(e / T) S u_n`,
    dont la pulsation passe de `omega` a `omega + Delta omega`.
    """
    induit = moment_induit(CHARGE, INDUCTION, RAYON, MASSE)
    assert induit == -(CHARGE**2) * INDUCTION * RAYON**2 / (4 * MASSE)
    assert induit.is_negative

    def orbital(pulsation: sp.Expr) -> sp.Expr:
        return moment_amperien(
            charge=-CHARGE,
            periode=2 * sp.pi / pulsation,
            aire=sp.pi * RAYON**2,
            normale=(0, 0, 1),
        )[2]

    larmor = ecart_de_larmor(CHARGE, INDUCTION, MASSE)
    assert sp.simplify(orbital(PULSATION + larmor) - orbital(PULSATION) - induit) == 0


def test_le_diamagnetisme_de_l_hydrogene_est_infime() -> None:
    """Un electron sur l'orbite de Bohr, dans un tesla : deux millioniemes."""
    larmor = float(ecart_de_larmor(1.602e-19, 1.0, 9.109e-31))
    assert larmor == pytest.approx(8.79e10, rel=1e-3)
    assert larmor / 4.13e16 < 1e-5


def test_un_couple_tend_a_aligner_le_moment_sur_l_induction() -> None:
    """Planche 6 : `C = m ^ B`, nul quand le moment est aligne."""
    assert couple_sur_un_moment((0, 0, 2), (0, 0, 5)) == (0, 0, 0)
    assert couple_sur_un_moment((2, 0, 0), (0, 3, 0)) == (0, 0, 6)


def test_ce_couple_derive_de_l_energie_a_courant_constant() -> None:
    """Chapitre 11 : `W_m = I phi = m B cos(alpha)`, `C = +(dW_m / d(alpha))_I`."""
    moment, angle = sp.Symbol("m", positive=True), sp.Symbol("alpha", real=True)
    energie = moment * INDUCTION * sp.cos(angle)
    par_l_energie = couple_par_l_energie(energie, angle, a_courants_constants=True)
    incline = (moment * sp.sin(angle), 0, moment * sp.cos(angle))
    vectoriel = couple_sur_un_moment(incline, (0, 0, INDUCTION))
    # L'angle se mesure de `B` vers `m`, autour de `y` : le couple le referme.
    assert sp.simplify(vectoriel[1] - par_l_energie) == 0


def test_le_couple_magnetique_est_l_homologue_du_couple_electrique() -> None:
    """Planche 23 de III.1 : `m` joue en magnetisme le role de `p`."""
    assert couple_sur_un_moment((1, 2, 3), (4, 5, 6)) == couple_sur_un_dipole(
        (1, 2, 3), (4, 5, 6)
    )


def test_le_tableau_de_la_planche_13_et_la_temperature_de_curie() -> None:
    """Planches 9 et 13 : les memes valeurs que le tableau 10.3.1 ; 1 050 K."""
    assert CORPS_DIAMAGNETIQUES["bismuth"] == 0.99984
    assert CORPS_PARAMAGNETIQUES["FeCl3"] == 1.003
    assert classe_de_magnetisme(0.99984) is Magnetisme.DIAMAGNETIQUE
    assert est_ferromagnetique(300.0, TEMPERATURE_DE_CURIE_DU_FER)
    # Planche 11 : `B = mu_0 (H + M)`.
    assert induction_du_milieu(100.0, 0.0) == pytest.approx(4e-7 * math.pi * 100.0)


H = sp.Symbol("H", positive=True)
MU0, SATURATION, ECHELLE = sp.symbols("mu_0 M_s a", positive=True)
PREMIERE_AIMANTATION = MU0 * (H + SATURATION * H**2 / (H**2 + ECHELLE**2))
"""Une courbe d'essai en S : l'aimantation part lentement, puis sature."""


def test_la_permeabilite_passe_par_un_maximum() -> None:
    """Planche 17 : elle part de `mu_i`, monte, puis tend vers `mu_0`."""
    normale = permeabilite_normale(PREMIERE_AIMANTATION, H)
    assert permeabilite_initiale(PREMIERE_AIMANTATION, H) == MU0
    assert sp.limit(normale, H, sp.oo) == MU0
    sommet = champ_du_maximum_de_permeabilite(PREMIERE_AIMANTATION, H)
    assert sommet == ECHELLE
    maximum = sp.simplify(normale.subs(H, sommet))
    assert sp.simplify(maximum - MU0 * (1 + SATURATION / (2 * ECHELLE))) == 0
    # Planche 16 : l'asymptote oblique a pour pente `mu_0`.
    assert pente_asymptotique(PREMIERE_AIMANTATION, H) == MU0


def test_une_courbe_lineaire_n_a_pas_de_maximum_de_permeabilite() -> None:
    with pytest.raises(ValueError, match="aucun maximum"):
        champ_du_maximum_de_permeabilite(3 * H, H)


def test_le_cycle_d_hysteresis_normal_est_symetrique() -> None:
    """Planche 18 : `B_r` a champ nul, `H_c` a induction nulle, de part et d'autre."""
    cycle = ((100.0, 1.0), (-50.0, 0.5), (-100.0, -1.0), (50.0, -0.5))
    retourne = [(-h, -b) for h, b in cycle]
    assert induction_remanente(cycle) == pytest.approx(induction_remanente(retourne))
    assert champ_coercitif(cycle) == pytest.approx(champ_coercitif(retourne))
    # Planche 19 : l'aire du cycle est l'energie volumique dissipee.
    assert aire_du_cycle(cycle) == pytest.approx(aire_du_cycle(retourne))
    assert aire_du_cycle(cycle) > 0


def test_le_champ_d_un_tore_parfait_est_nul_au_dehors() -> None:
    """Planche 21 : `2 pi r H = I` dans l'anneau (D2), `2 pi r H = 0` hors de lui."""
    dedans = champ_du_tore_parfait(600.0, 0.1, interieur=0.08, exterieur=0.12)
    assert dedans == pytest.approx(600.0 / (2 * math.pi * 0.1))
    assert courant_enlace(dedans, 2 * math.pi * 0.1) == pytest.approx(600.0)
    assert champ_du_tore_parfait(600.0, 0.05, interieur=0.08, exterieur=0.12) == 0
    assert champ_du_tore_parfait(600.0, 0.2, interieur=0.08, exterieur=0.12) == 0


def test_le_champ_du_tore_varie_peu_si_la_section_est_mince() -> None:
    """Planche 22 : « il depend de r », mais peu si le rayon central est grand."""
    bord_interieur = champ_du_tore_parfait(600.0, 0.99, interieur=0.99, exterieur=1.01)
    bord_exterieur = champ_du_tore_parfait(600.0, 1.01, interieur=0.99, exterieur=1.01)
    assert bord_interieur / bord_exterieur == pytest.approx(1.02, abs=1e-3)


def test_un_tore_dont_les_rayons_sont_inverses_est_refuse() -> None:
    with pytest.raises(ValueError, match="rayons invalides"):
        champ_du_tore_parfait(600.0, 0.1, interieur=0.12, exterieur=0.08)


def test_un_materiau_ferromagnetique_reduit_la_part_du_flux_de_dispersion() -> None:
    """Planche 23 : `B = mu H` plus grand dans le tore, dispersion plus faible."""
    dans_l_air = flux_utile_et_de_dispersion(300.0, 2e7, 2e7)
    dans_le_fer = flux_utile_et_de_dispersion(300.0, 2e7 / 1000, 2e7)
    part_dans_l_air = dans_l_air[1] / sum(dans_l_air)
    part_dans_le_fer = dans_le_fer[1] / sum(dans_le_fer)
    assert part_dans_le_fer < part_dans_l_air / 100
