"""Syllabus d'electricite generale, chapitre 17 : les courants periodiques.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 17-1 a 17-21, lues sur les pages rendues.

Les oracles du cours : les figures 17.2.1 a 17.2.3 (un fondamental
d'amplitude 1 et un harmonique 3 d'amplitude 0,5), et les facteurs de
crete 1,414 et de forme 1,111 de la sinusoide (page 17-12). Les
developpements de Fourier viennent du contexte des mathematiques : un
harmonique n'est pas recalcule ici, il est traduit.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.grandeur_periodique import (
    est_alternative,
    est_symetrique,
    facteur_de_crete,
    facteur_de_forme,
    harmonique,
    taux_d_harmoniques,
    valeur_moyenne,
    valeur_redressee,
)
from polytech.electromagnetisme.domain.circuits.periodiques import (
    facteur_de_puissance,
    puissance_apparente,
    puissance_deformante,
    puissance_reactive_globale,
    valeur_efficace_des_harmoniques,
)
from polytech.electromagnetisme.domain.circuits.puissances_periodiques import (
    puissance_active_des_harmoniques,
    puissance_deformante_generale,
    puissance_reactive_des_harmoniques,
)
from polytech.electromagnetisme.domain.circuits.valeurs_efficaces import (
    carre,
    triangle,
    valeur_efficace,
)
from polytech.electromagnetisme.domain.magnetostatique.courbes_d_aimantation import (
    aire_du_cycle,
    permeabilite_differentielle,
)
from polytech.electromagnetisme.domain.magnetostatique.cycle_dynamique import (
    champ_a_la_surface,
    flux_impose,
    induction_moyenne,
    pertes_magnetiques,
    puissance_dissipee_dans_le_barreau,
)
from polytech.electromagnetisme.domain.magnetostatique.permeabilites import (
    permeabilite_d_amplitude,
    permeabilite_incrementale,
    permeabilite_reversible,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.maths.domain.fourier.series import SerieDeFourier
from polytech.maths.domain.symboles import VARIABLE

PERIODE = 2 * sp.pi
FIGURE_1 = sp.cos(TEMPS) + sp.cos(3 * TEMPS) / 2
FIGURE_2 = sp.cos(TEMPS) + sp.cos(3 * TEMPS + sp.pi / 2) / 2
FIGURE_3 = sp.cos(TEMPS) + sp.cos(3 * TEMPS + sp.pi) / 2
"""Figures 17.2.1 a 17.2.3 : `theta_3 = 3 theta_1`, puis `+ pi / 2`, puis `+ pi`."""


def test_une_grandeur_alternative_a_une_valeur_moyenne_nulle() -> None:
    """(17.1.2) : alternative si la moyenne est nulle, pulsatoire sinon."""
    assert valeur_moyenne(FIGURE_1, PERIODE) == 0
    assert est_alternative(FIGURE_1, PERIODE)
    pulsatoire = FIGURE_1 + sp.Rational(3, 2)
    assert valeur_moyenne(pulsatoire, PERIODE) == sp.Rational(3, 2)
    assert not est_alternative(pulsatoire, PERIODE)


def test_les_harmoniques_des_figures_17_2_1_a_17_2_3() -> None:
    """(17.2.4) et (17.2.5) : amplitudes 1 et 0,5, phases 0, `pi / 2` et `pi`."""
    assert harmonique(FIGURE_1, PERIODE, 1) == (1 / sp.sqrt(2), 0)
    for figure, phase in ((FIGURE_1, 0), (FIGURE_2, sp.pi / 2), (FIGURE_3, sp.pi)):
        efficace, lue = harmonique(figure, PERIODE, 3)
        assert sp.simplify(efficace * sp.sqrt(2) - sp.Rational(1, 2)) == 0
        assert lue == phase


def test_l_harmonique_est_celui_de_la_serie_de_fourier_des_mathematiques() -> None:
    """`X_n = A_n / racine(2)` et la meme phase : une traduction, pas un recalcul."""
    serie = SerieDeFourier(FIGURE_2.subs(TEMPS, VARIABLE), 0, PERIODE)
    efficace, phase = harmonique(FIGURE_2, PERIODE, 3)
    assert sp.simplify(efficace - serie.amplitude(3) / sp.sqrt(2)) == 0
    assert phase == serie.phase(3)


def test_une_grandeur_symetrique_n_a_pas_d_harmonique_pair() -> None:
    """Page 17-4 : `x(t + T / 2) = -x(t)`, le cas des trois figures."""
    for figure in (FIGURE_1, FIGURE_2, FIGURE_3):
        assert est_symetrique(figure, PERIODE)
        # Un harmonique absent n'a pas de phase : zero par convention.
        assert harmonique(figure, PERIODE, 2) == (0, 0)
    paire = sp.cos(TEMPS) + sp.cos(2 * TEMPS)
    assert not est_symetrique(paire, PERIODE)
    assert harmonique(paire, PERIODE, 2)[0] != 0


def test_une_origine_des_temps_sur_un_axe_de_symetrie_annule_les_phases() -> None:
    """Page 17-5 : `x(t_1 - t) = x(t_1 + t)` donne des phases nulles ou de `pi`."""
    for figure in (FIGURE_1, FIGURE_3):
        assert sp.simplify(figure.subs(TEMPS, -TEMPS) - figure) == 0
        assert harmonique(figure, PERIODE, 3)[1] in (0, sp.pi)
    assert harmonique(FIGURE_2, PERIODE, 3)[1] not in (0, sp.pi)


def test_la_formule_de_parseval() -> None:
    """(17.3.2) : `X = racine(X_0^2 + X_1^2 + X_2^2 + ...)`."""
    pulsatoire = FIGURE_2 + sp.Rational(3, 2)
    par_l_integrale = valeur_efficace(pulsatoire, PERIODE)
    rangs = [float(harmonique(pulsatoire, PERIODE, n)[0]) for n in (1, 2, 3)]
    par_les_harmoniques = valeur_efficace_des_harmoniques(1.5, tuple(rangs))
    assert float(par_l_integrale) == pytest.approx(par_les_harmoniques)
    assert par_les_harmoniques == pytest.approx(math.sqrt(2.25 + 0.5 + 0.125))


def test_la_valeur_redressee_et_les_facteurs_de_la_sinusoide() -> None:
    """(17.6.1) a (17.6.3), page 17-12 : `2 x_m / pi`, puis 1,414 et 1,111."""
    sinusoide = 3 * sp.cos(TEMPS)
    redressee = valeur_redressee(sinusoide, PERIODE)
    assert sp.simplify(redressee - 6 / sp.pi) == 0
    efficace = valeur_efficace(sinusoide, PERIODE)
    assert float(facteur_de_crete(3, efficace)) == pytest.approx(1.414, abs=5e-4)
    assert float(facteur_de_forme(efficace, redressee)) == pytest.approx(
        1.111, abs=5e-4
    )


def test_un_facteur_de_forme_de_1_111_ne_prouve_pas_la_sinusoide() -> None:
    """Page 17-12 : « la reciproque n'est pas necessairement vraie ».

    Le creneau et le triangle encadrent la sinusoide : 1 et `2 / racine(3)`.
    """
    periode = sp.Integer(4)
    creneau, dent = carre(sp.Integer(2), periode), triangle(sp.Integer(2), periode)
    forme_du_creneau = facteur_de_forme(
        valeur_efficace(creneau, periode), valeur_redressee(creneau, periode)
    )
    forme_du_triangle = facteur_de_forme(
        valeur_efficace(dent, periode), valeur_redressee(dent, periode)
    )
    assert forme_du_creneau == 1
    assert sp.simplify(forme_du_triangle - 2 / sp.sqrt(3)) == 0
    assert float(forme_du_creneau) < 1.111 < float(forme_du_triangle)
    assert facteur_de_crete(2, valeur_efficace(creneau, periode)) == 1


def test_le_taux_d_harmoniques_se_rapporte_a_la_grandeur_entiere() -> None:
    """(17.6.5) : `h = racine(X^2 - X_1^2) / X`, ici `1 / racine(5)`."""
    efficace = valeur_efficace(FIGURE_1, PERIODE)
    fondamental = harmonique(FIGURE_1, PERIODE, 1)[0]
    taux = taux_d_harmoniques(efficace, fondamental)
    assert sp.simplify(taux - 1 / sp.sqrt(5)) == 0
    assert taux_d_harmoniques(5, 5) == 0


def test_les_gardes_des_facteurs() -> None:
    with pytest.raises(ValueError, match="valeur efficace nulle"):
        facteur_de_crete(3, 0)
    with pytest.raises(ValueError, match="valeur redressee nulle"):
        facteur_de_forme(3, 0)
    with pytest.raises(ValueError, match="valeur efficace nulle"):
        taux_d_harmoniques(0, 0)
    with pytest.raises(ValueError, match="fondamental superieur"):
        taux_d_harmoniques(3, 4)


TENSION = 10 * sp.sqrt(2) * sp.cos(TEMPS)
COURANT = 4 * sp.sqrt(2) * sp.cos(TEMPS - sp.pi / 3)
COURANT += 3 * sp.sqrt(2) * sp.cos(3 * TEMPS + sp.pi / 5)
"""Tension sinusoidale, courant deforme : 4 A au fondamental, 3 A au rang 3."""


def test_seuls_les_harmoniques_communs_portent_de_la_puissance_active() -> None:
    """(17.4.1), et (17.4.5) quand la tension est sinusoidale."""
    moyenne = sp.integrate(TENSION * COURANT, (TEMPS, 0, PERIODE)) / PERIODE
    active = puissance_active_des_harmoniques((10, 0), (4, 3), (sp.pi / 3, 0))
    assert sp.simplify(moyenne - active) == 0
    assert active == 20


def test_la_composante_continue_compte_sans_cosinus() -> None:
    """(17.4.1) : `P = U_0 I_0 + somme U_n I_n cos(phi_n)`."""
    active = puissance_active_des_harmoniques(
        (10,), (4,), (sp.pi / 3,), continues=(2, 5)
    )
    assert active == 30


def test_des_listes_de_longueurs_differentes_sont_refusees() -> None:
    with pytest.raises(ValueError, match="autant de tensions"):
        puissance_active_des_harmoniques((10, 0), (4,), (0,))
    with pytest.raises(ValueError, match="autant de tensions"):
        puissance_active_des_harmoniques((10, 0), (4,), (0, 0))


def test_le_facteur_de_puissance_ne_depasse_pas_un() -> None:
    """(17.4.4) et la demonstration de la page 17-8 : `A^2 = 2 (1 - lambda)`."""
    apparente = puissance_apparente(10.0, 5.0)
    facteur = facteur_de_puissance(20.0, apparente)
    assert facteur == pytest.approx(0.4)
    ecart = COURANT / 5 - TENSION / 10
    carre_efficace = valeur_efficace(ecart, PERIODE) ** 2
    assert float(carre_efficace) == pytest.approx(2 * (1 - facteur))


def test_les_trois_puissances_et_la_puissance_deformante() -> None:
    """(17.5.3) et (17.5.4) : `S^2 = P^2 + Q^2 + D^2`, ici `D = U I_3`."""
    reactive = puissance_reactive_des_harmoniques((10, 0), (4, 3), (sp.pi / 3, 0))
    assert reactive == 20 * sp.sqrt(3)
    deformante = puissance_deformante_generale(50, 20, reactive)
    assert deformante == 30
    # Page 17-10 : tension sinusoidale, `Q_d^2 = Q^2 - Q_1^2` avec (17.5.2).
    globale = puissance_reactive_globale(50.0, 20.0)
    assert puissance_deformante(globale, float(reactive)) == pytest.approx(30.0)


def test_une_puissance_deformante_imaginaire_est_refusee() -> None:
    with pytest.raises(ValueError, match="depasse la puissance apparente"):
        puissance_deformante_generale(10, 8, 8)


def test_le_champ_a_la_surface_et_l_induction_moyenne_du_barreau() -> None:
    """(17.7.1) `H = N i / l` et (17.7.2) `B = Phi / S`."""
    assert champ_a_la_surface(500, 0.4, 0.25) == pytest.approx(800.0)
    assert induction_moyenne(6e-4, 4e-4) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="longueur nulle ou negative"):
        champ_a_la_surface(500, 0.4, 0.0)
    with pytest.raises(ValueError, match="section nulle ou negative"):
        induction_moyenne(6e-4, 0.0)


def test_la_puissance_de_l_enroulement_se_retrouve_dans_le_barreau() -> None:
    """(17.7.3) et (17.7.4) : `N i dPhi/dt = l S H dB/dt = V H dB/dt`."""
    spires, longueur, section = sp.symbols("N l S", positive=True)
    courant, induction = sp.Function("i")(TEMPS), sp.Function("B")(TEMPS)
    cote_enroulement = spires * courant * sp.diff(induction * section, TEMPS)
    champ = spires * courant / longueur
    dans_le_barreau = puissance_dissipee_dans_le_barreau(
        longueur * section, champ, induction
    )
    assert sp.simplify(cote_enroulement - dans_le_barreau) == 0


def test_l_energie_d_un_cycle_dynamique_est_son_aire_fois_le_volume() -> None:
    """(17.7.5) : `integrale de p_t dt = V . aire`, sur un cycle elliptique.

    `H = H_m cos(omega t)`, `B = B_m cos(omega t - delta)` : l'induction
    retarde, et l'aire vaut `pi H_m B_m sin(delta)`.
    """
    crete_h, crete_b, retard, volume = 300, sp.Rational(4, 5), sp.pi / 6, 2
    champ = crete_h * sp.cos(TEMPS)
    induction = crete_b * sp.cos(TEMPS - retard)
    puissance = puissance_dissipee_dans_le_barreau(volume, champ, induction)
    energie = sp.integrate(puissance, (TEMPS, 0, PERIODE))
    assert sp.simplify(energie - volume * sp.pi * crete_h * crete_b / 2) == 0
    pas = 2000
    cycle = [
        (
            float(champ.subs(TEMPS, 2 * math.pi * k / pas)),
            float(induction.subs(TEMPS, 2 * math.pi * k / pas)),
        )
        for k in range(pas)
    ]
    assert volume * aire_du_cycle(cycle) == pytest.approx(float(energie), rel=1e-5)


def test_l_hysteresis_croit_comme_f_et_foucault_comme_f_carre() -> None:
    """Page 17-16 : doubler la frequence double les unes et quadruple les autres."""
    seules = pertes_magnetiques(hysteresis=0.02, foucault=0.0, frequence=50.0)
    assert pertes_magnetiques(0.02, 0.0, 100.0) == pytest.approx(2 * seules)
    induites = pertes_magnetiques(hysteresis=0.0, foucault=4e-4, frequence=50.0)
    assert pertes_magnetiques(0.0, 4e-4, 100.0) == pytest.approx(4 * induites)
    assert pertes_magnetiques(0.02, 4e-4, 50.0) == pytest.approx(seules + induites)
    with pytest.raises(ValueError, match="frequence negative"):
        pertes_magnetiques(0.02, 4e-4, -1.0)


def test_une_tension_sinusoidale_impose_un_flux_sinusoidal() -> None:
    """(17.8.1) sans `R i`, et (17.8.2) : `phi_max = U racine(2) / (N omega)`."""
    tension, spires, pulsation = sp.symbols("U N omega", positive=True)
    phase = sp.Symbol("theta_u", real=True)
    flux = flux_impose(tension, spires, pulsation, phase)
    appliquee = tension * sp.sqrt(2) * sp.cos(pulsation * TEMPS + phase)
    assert sp.simplify(spires * sp.diff(flux, TEMPS) - appliquee) == 0
    crete = flux.subs(TEMPS, (sp.pi / 2 - phase) / pulsation)
    # Page 16-30 : `U = N omega phi_max / racine(2)`.
    assert sp.simplify(spires * pulsation * crete / sp.sqrt(2) - tension) == 0


def test_la_saturation_deforme_le_champ_sans_harmonique_pair() -> None:
    """Page 17-20 : pour une induction sinusoidale, `H` n'est plus sinusoidal.

    Courbe d'essai sans hysteresis, `H = a B + b B^3` : le champ gagne un
    harmonique 3 d'amplitude `b B_m^3 / 4`, et reste symetrique.
    """
    induction = 2 * sp.sin(TEMPS)
    champ = 100 * induction + 30 * induction**3
    assert est_symetrique(champ, PERIODE)
    assert harmonique(champ, PERIODE, 2)[0] == 0
    troisieme = harmonique(champ, PERIODE, 3)[0] * sp.sqrt(2)
    assert sp.simplify(troisieme - 30 * 2**3 / sp.Integer(4)) == 0


H = sp.Symbol("H", positive=True)
COURBE = 5 * H - H**3 / 12
"""Une courbe d'essai, sans hysteresis : la pente decroit quand `H` monte."""


def test_la_permeabilite_d_amplitude_est_le_rapport_des_cretes() -> None:
    """Page 17-20 : `mu_a = B_crete / H_crete`."""
    assert permeabilite_d_amplitude(1.2, 400.0) == pytest.approx(3e-3)
    with pytest.raises(ValueError, match="champ de crete nul"):
        permeabilite_d_amplitude(1.2, 0.0)


def test_la_permeabilite_avec_champ_statique_superpose() -> None:
    """(17.9.1) : `mu = Delta B / Delta H` autour de `H_0`, figure 17.9.1."""
    incrementale = permeabilite_incrementale(COURBE, H, statique=2, excursion=1)
    assert incrementale == (COURBE.subs(H, 3) - COURBE.subs(H, 1)) / 2
    assert incrementale == sp.Rational(47, 12)
    with pytest.raises(ValueError, match="excursion nulle ou negative"):
        permeabilite_incrementale(COURBE, H, statique=2, excursion=0)


def test_la_permeabilite_reversible_en_est_la_limite() -> None:
    """Page 17-21 : quand la composante alternative du champ tend vers zero.

    Sur une courbe sans hysteresis, c'est la pente de la courbe au point
    de repos : la permeabilite differentielle de la section 10.5.
    """
    reversible = permeabilite_reversible(COURBE, H, statique=2)
    assert reversible == 4
    assert reversible == permeabilite_differentielle(COURBE, H).subs(H, 2)
    petite = permeabilite_incrementale(
        COURBE, H, statique=2, excursion=sp.Rational(1, 100)
    )
    assert abs(float(petite - reversible)) < 1e-4
