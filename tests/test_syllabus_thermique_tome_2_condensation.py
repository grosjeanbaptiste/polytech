"""Le syllabus de thermique, tome 2 : la condensation.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 65
a 77 (premiere partie, chapitre 5), lues sur les pages rendues.

Les oracles : les equations (5.1) a (5.21), dont chaque constante est
rederivee -- 0,943, 0,693, 0,925, 0,954, 1,47, 1,2, et le 0,725 du tube
horizontal par son integrale --, et les exercices III.5.1 et III.5.2.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp
from scipy.integrate import quad
from scipy.special import gamma

from polytech.thermique.domain.condensation import vitesse_normale_de_la_vapeur
from polytech.thermique.domain.condensation_sur_tubes import (
    REYNOLDS_CRITIQUE_DU_FILM,
    coefficient_horizontal,
    coefficient_horizontal_par_le_debit,
    coefficient_vertical_par_le_reynolds,
    rapport_horizontal_sur_vertical,
    reynolds_du_film,
    reynolds_sous_une_colonne,
)
from polytech.thermique.domain.condenseur import (
    debit_par_metre_de_colonne,
    surface_des_tubes,
    temperature_de_saturation,
)
from polytech.thermique.domain.film_de_condensat import (
    Condensat,
    debit_du_film,
    profil_de_vitesse_du_film,
)
from polytech.thermique.domain.tables.eau_saturee import (
    eau_saturee,
    pression_de_saturation,
)

EAU = Condensat(
    masse_volumique=965.2,
    vapeur=0.598,
    viscosite=0.316e-3,
    conductivite=0.675,
    chaleur_latente=2256900.0,
)
"""Exercice III.5.1 : le condensat a 90 degres, la vapeur a une atmosphere."""
SOUS_REFROIDI = EAU.avec_chaleur_latente(2256900.0 * 1.037)
"""La chaleur latente que le syllabus emploie, corrigee de son facteur 1,037."""


def test_le_profil_de_vitesse_du_film() -> None:
    """(5.1) et (5.2) : `mu u'' + (rho_l - rho_v) g sin(beta) = 0`, nul a la paroi."""
    y, delta, beta = sp.symbols("y delta beta", positive=True)
    poids, viscosite = sp.symbols("gamma mu", positive=True)
    profil = profil_de_vitesse_du_film(y, delta, poids * sp.sin(beta), viscosite)
    assert sp.simplify(viscosite * sp.diff(profil, y, 2) + poids * sp.sin(beta)) == 0
    assert profil.subs(y, 0) == 0
    assert sp.diff(profil, y).subs(y, delta) == 0
    moyenne = sp.integrate(profil, (y, 0, delta)) / delta
    assert sp.simplify(moyenne - sp.Rational(2, 3) * profil.subs(y, delta)) == 0
    attendue = poids * sp.sin(beta) * delta**2 / (3 * viscosite)
    assert sp.simplify(moyenne - attendue) == 0


def test_le_debit_du_film() -> None:
    """(5.3) : `G = delta rho_l u_m`, en `delta^3 sin(beta)`."""
    assert EAU.poids_apparent() == pytest.approx((965.2 - 0.598) * 9.81)
    epaisseur = 1e-4
    vertical = debit_du_film(EAU, epaisseur)
    attendu = (965.2 - 0.598) * 965.2 * 9.81 * epaisseur**3 / (3 * 0.316e-3)
    assert vertical == pytest.approx(attendu)
    incline = debit_du_film(EAU, epaisseur, inclinaison=math.pi / 6)
    assert incline == pytest.approx(vertical / 2)


def test_l_epaisseur_du_film_sur_une_paroi_verticale() -> None:
    """(5.4) a (5.7) : `l_v dG = (lambda / delta)(t_sat - t_p) dx`, `delta ~ x^1/4`."""
    hauteur, ecart = 3.0, 20.0
    epaisseur = EAU.epaisseur_du_film(hauteur, ecart)
    pas = 1e-4
    amont = EAU.epaisseur_du_film(hauteur - pas, ecart)
    aval = EAU.epaisseur_du_film(hauteur + pas, ecart)
    degage = EAU.chaleur_latente * (
        debit_du_film(EAU, aval) - debit_du_film(EAU, amont)
    )
    conduit = EAU.conductivite / epaisseur * ecart * 2 * pas
    assert degage == pytest.approx(conduit, rel=1e-6)
    assert EAU.epaisseur_du_film(16 * hauteur, ecart) == pytest.approx(2 * epaisseur)
    assert EAU.epaisseur_du_film(0.0, ecart) == 0.0
    with pytest.raises(ValueError, match="abscisse nulle ou negative"):
        EAU.coefficient_local(0.0, ecart)


def test_le_coefficient_moyen_vaut_quatre_tiers_du_coefficient_final() -> None:
    """(5.8) et (5.9) : `K = lambda / delta`, `K_v = (4/3) K(H) = 0,943 [...]^1/4`."""
    hauteur, ecart = 3.0, 20.0
    local = EAU.coefficient_local(hauteur, ecart)
    assert local == pytest.approx(0.675 / EAU.epaisseur_du_film(hauteur, ecart))
    moyen = EAU.coefficient_vertical(hauteur, ecart)
    assert moyen == pytest.approx(4 / 3 * local)
    integre, _ = quad(EAU.coefficient_local, 0.0, hauteur, args=(ecart,))
    assert moyen == pytest.approx(integre / hauteur, rel=1e-6)
    assert pytest.approx(0.943, abs=5e-4) == 4 / 3 * 0.25**0.25


def test_le_coefficient_en_fonction_du_debit() -> None:
    """(5.10) a (5.12) : `K(x) = 0,693 [...]^1/3`, `K_v = 0,925 [...]^1/3`."""
    assert pytest.approx(0.693, abs=5e-4) == (1 / 3) ** (1 / 3)
    assert pytest.approx(0.925, abs=1e-3) == 4 / 3 * (1 / 3) ** (1 / 3)
    hauteur, ecart = 3.0, 20.0
    debit = debit_du_film(EAU, EAU.epaisseur_du_film(hauteur, ecart))
    par_le_debit = EAU.coefficient_vertical_par_le_debit(debit)
    assert par_le_debit == pytest.approx(EAU.coefficient_vertical(hauteur, ecart))
    bilan = debit * EAU.chaleur_latente
    assert bilan == pytest.approx(hauteur * par_le_debit * ecart)


def test_le_0_725_du_tube_horizontal_sort_d_une_integrale() -> None:
    """(5.13) : le bilan du film autour du tube donne 0,728 ; le syllabus arrondit.

    `delta^4 sin^(4/3) = (4B / 3A) int sin^(1/3)`, d'ou le coefficient
    `(4 / 3 pi) I^(3/4) 2^-1/4`, avec `I = int_0^pi sin^(1/3)`.
    """
    integrale, _ = quad(lambda angle: math.sin(angle) ** (1 / 3), 0.0, math.pi)
    assert integrale == pytest.approx(math.sqrt(math.pi) * gamma(2 / 3) / gamma(7 / 6))
    exact = 4 / (3 * math.pi) * integrale**0.75 * 2**-0.25
    assert exact == pytest.approx(0.728, abs=5e-4)
    assert exact == pytest.approx(0.725, rel=0.005)
    horizontal = coefficient_horizontal(EAU, 0.016, 20.0)
    groupe = 2256900.0 * 9.81 * 965.2 * 964.602 * 0.675**3 / (0.316e-3 * 20.0 * 0.016)
    assert horizontal == pytest.approx(0.725 * groupe**0.25)


def test_le_tube_horizontal_l_emporte_sur_le_tube_vertical() -> None:
    """Page 72 : `K_h / K_v = 0,77 (H / d)^1/4`."""
    rapport = rapport_horizontal_sur_vertical(3.0, 0.016)
    imprime = 0.77 * (3.0 / 0.016) ** 0.25
    assert rapport == pytest.approx(imprime, rel=2e-3)
    assert pytest.approx(0.77, abs=0.002) == 0.725 / 0.943
    direct = coefficient_horizontal(EAU, 0.016, 20.0) / EAU.coefficient_vertical(
        3.0, 20.0
    )
    assert rapport == pytest.approx(direct)


def test_l_effet_de_ruissellement() -> None:
    """(5.14) et (5.15) : `n^-1/4` pour `n` tubes d'un meme plan ; puis 0,954."""
    seul = coefficient_horizontal(EAU, 0.016, 20.0)
    colonne = coefficient_horizontal(EAU, 0.016, 20.0, tubes=10)
    assert colonne == pytest.approx(seul * 10**-0.25)
    assert (0.725**4 * math.pi) ** (1 / 3) == pytest.approx(0.954, abs=5e-4)
    longueur = 3.0
    debit = 10 * math.pi * 0.016 * colonne * 20.0 / EAU.chaleur_latente
    par_le_debit = coefficient_horizontal_par_le_debit(EAU, debit)
    assert par_le_debit == pytest.approx(colonne, rel=1e-3)
    assert longueur * debit * EAU.chaleur_latente == pytest.approx(
        10 * math.pi * 0.016 * longueur * colonne * 20.0
    )


def test_le_reynolds_d_un_film() -> None:
    """(5.16) a (5.20) : `Re = 4 G / mu` ; 1,47 = `0,925 4^1/3`, 1,2 = `0,954 2^1/3`."""
    assert reynolds_du_film(0.10425, 0.316e-3) == pytest.approx(1319.0, abs=1.0)
    assert reynolds_sous_une_colonne(0.00497, 0.316e-3) == pytest.approx(
        31.45, abs=0.02
    )
    assert pytest.approx(1.47, abs=0.003) == 0.925 * 4 ** (1 / 3)
    assert pytest.approx(1.2, abs=0.003) == 0.954 * 2 ** (1 / 3)
    with pytest.raises(ValueError, match="viscosite nulle ou negative"):
        reynolds_du_film(0.1, 0.0)


def test_la_forme_reduite_du_coefficient_vertical() -> None:
    """(5.17) et (5.21) : `1,47 Re^-1/3` jusqu'a 1800, `0,0077 Re^0,4` au-dela."""
    assert REYNOLDS_CRITIQUE_DU_FILM == 1800.0
    laminaire = coefficient_vertical_par_le_reynolds(EAU, 1319.0)
    assert laminaire == pytest.approx(1.47 * 1319.0 ** (-1 / 3) / EAU.echelle())
    par_le_debit = EAU.coefficient_vertical_par_le_debit(1319.0 * 0.316e-3 / 4)
    assert laminaire == pytest.approx(par_le_debit, rel=2e-3)
    turbulent = coefficient_vertical_par_le_reynolds(EAU, 1e4)
    assert turbulent == pytest.approx(0.0077 * 1e4**0.4 / EAU.echelle())
    juste_avant = coefficient_vertical_par_le_reynolds(EAU, 1700.0)
    assert juste_avant == pytest.approx(1.47 * 1700.0 ** (-1 / 3) / EAU.echelle())
    assert turbulent > juste_avant


def test_le_facteur_de_sous_refroidissement_de_l_exercice_iii_5_1() -> None:
    """« 1 + 0,375 x 4205 x 20 / 2 256 900 = 1,037 » : le produit vaut 1,014.

    C'est sans le 0,375 que l'on trouve 1,037.
    """
    assert pytest.approx(1.014, abs=5e-4) == 1 + 0.375 * 4205.0 * 20.0 / 2256900.0
    assert pytest.approx(1.037, abs=5e-4) == 1 + 4205.0 * 20.0 / 2256900.0
    corrige = EAU.sous_refroidi(4205.0, 20.0)
    assert corrige.chaleur_latente == pytest.approx(2256900.0 * 1.01397, rel=1e-5)


def test_l_exercice_iii_5_1_le_tube_vertical() -> None:
    """3 m, 16 mm, 100 et 80 degres : 4069 W/m^2 K, 12 272 W, 0,00524 kg/s."""
    film = eau_saturee(90.0)
    assert film.conductivite == pytest.approx(0.675, abs=0.0015)
    assert film.viscosite == pytest.approx(0.316e-3, abs=1e-6)
    echange = SOUS_REFROIDI.coefficient_vertical(3.0, 20.0)
    assert echange == pytest.approx(4069.0, abs=1.0)
    surface = math.pi * 0.016 * 3.0
    flux = echange * surface * 20.0
    assert flux == pytest.approx(12272.0, abs=3.0)
    debit = flux / SOUS_REFROIDI.chaleur_latente
    assert debit == pytest.approx(0.00524, abs=5e-6)
    lineique = debit / (math.pi * 0.016)
    assert lineique == pytest.approx(0.10425, abs=5e-5)
    nombre = reynolds_du_film(lineique, 0.316e-3)
    assert nombre == pytest.approx(1319.0, abs=1.5)
    assert nombre < REYNOLDS_CRITIQUE_DU_FILM
    reduit = coefficient_vertical_par_le_reynolds(EAU, nombre)
    assert reduit == pytest.approx(echange, rel=3e-3)


def test_l_exercice_iii_5_1_le_tube_horizontal() -> None:
    """11 577 W/m^2 K, 34 915 W, 0,01492 kg/s, `Re_D = 31,45`.

    Le syllabus y ecrit « 946,6 » pour `rho_l - rho_v` ; son resultat
    emploie bien 964,6.
    """
    echange = coefficient_horizontal(SOUS_REFROIDI, 0.016, 20.0)
    assert echange == pytest.approx(11577.0, abs=2.0)
    flux = echange * math.pi * 0.016 * 3.0 * 20.0
    assert flux == pytest.approx(34915.0, abs=6.0)
    debit = flux / SOUS_REFROIDI.chaleur_latente
    assert debit == pytest.approx(0.01492, abs=5e-6)
    nombre = reynolds_sous_une_colonne(debit / 3.0, 0.316e-3)
    assert nombre == pytest.approx(31.45, abs=0.05)
    avec_la_coquille = echange * (946.6 / 964.6) ** 0.25
    assert avec_la_coquille == pytest.approx(11523.0, abs=3.0)


def test_la_vapeur_n_a_besoin_que_d_un_dixieme_de_metre_par_seconde() -> None:
    """Page 66 : `u_v = phi / (rho_v l_v) ~ 0,1 m/s`, d'apres l'exercice III.5.1."""
    densite = 11577.0 * 20.0
    vitesse = vitesse_normale_de_la_vapeur(densite, 0.598, 2256900.0)
    assert vitesse == pytest.approx(0.17, abs=0.01)


def test_l_exercice_iii_5_2_le_vide_d_un_condenseur() -> None:
    """75,56 kg/s sur 6230 tubes de 22 mm : `t_sat = 32,03` degres, 0,0476 bar.

    L'equation (III) imprime « 75,86 » ; son resultat emploie 75,56. Le
    `h = 5307` du syllabus est celui que la bibliotheque rend pour
    75,86 kg/s (5308,3), et non pour 75,56 (5315,3) ; le `32,03` combine
    donc ce `h` (arrondi) avec le debit 75,56 (32,02 avec le `h` exact).
    """
    surface = surface_des_tubes(6230, 8.0, 0.022)
    assert surface == pytest.approx(3444.7, abs=0.1)
    film = eau_saturee((32.0 + 22.0) / 2)
    assert film.viscosite == pytest.approx(859e-6, abs=1e-6)
    assert film.conductivite == pytest.approx(0.6138, abs=5e-4)
    condensat = Condensat(996.6, 0.0, 859e-6, 0.6138, 2425900.0)
    lineique = debit_par_metre_de_colonne(75.56, 6230, 10, 8.0)
    assert lineique == pytest.approx(0.01516, abs=5e-6)
    nombre = reynolds_sous_une_colonne(lineique, 859e-6)
    assert nombre == pytest.approx(35.3, abs=0.01)
    echange = coefficient_horizontal_par_le_debit(condensat, lineique)
    assert echange == pytest.approx(5315.3, abs=0.05)
    echange_586 = coefficient_horizontal_par_le_debit(
        condensat, debit_par_metre_de_colonne(75.86, 6230, 10, 8.0)
    )
    assert echange_586 == pytest.approx(5308.3, abs=0.05)
    assert echange_586 == pytest.approx(5307.0, rel=3e-4)

    def saturation(debit: float, h: float) -> float:
        return temperature_de_saturation(22.0, debit, 2425900.0, surface, h)

    assert saturation(75.56, echange) == pytest.approx(32.01, abs=0.005)
    assert saturation(75.86, echange_586) == pytest.approx(32.06, abs=0.005)
    assert saturation(75.56, echange_586) == pytest.approx(32.02, abs=0.005)
    assert temperature_de_saturation(
        22.0, 75.56, 2425900.0, surface, 5307.0
    ) == pytest.approx(32.03, abs=0.005)
    assert saturation(75.86, 5307.0) == pytest.approx(32.07, abs=0.005)
    assert pression_de_saturation(saturation(75.56, echange_586)) == pytest.approx(
        4760.0, abs=5.0
    )
