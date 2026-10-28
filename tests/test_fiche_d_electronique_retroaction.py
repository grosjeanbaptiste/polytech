"""Fiche d'electronique : retroaction, produit gain-bande, decalages, Bode.

Articles 3, 10, 16, 17, 19 et 20 de `cursus/fiches/ElectroniqueInstrumentation.toml`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.domain import laplace
from polytech.electronique.domain.amplification import (
    gain_maximal_par_etage,
    nombre_d_etages,
)
from polytech.electronique.domain.aop_interne import (
    courant_de_decalage,
    courant_de_polarisation,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
    SourceDeTensionCommandee,
    resoudre,
    systeme_nodal_modifie,
)
from polytech.electronique.domain.pole_zero import (
    carte_des_poles_et_zeros,
    pente_asymptotique,
)
from polytech.electronique.domain.retroaction import (
    gain_en_boucle_fermee,
    impedance_de_sortie_en_boucle_fermee,
    parametres_h_amplificateur,
    parametres_h_reseau_resistif,
)
from polytech.maths.domain.comparaison import egales, proches

S = sp.Symbol("s")


def test_les_matrices_h_s_ajoutent_en_serie_parallele() -> None:
    """Article « La retroaction vue par les quadripoles » : `1/beta`, `Z_o/(1+bA)`."""
    amplificateur = parametres_h_amplificateur(1e6, 100, 1e5)
    pont = parametres_h_reseau_resistif(1000, 9000)
    total = amplificateur + pont
    beta, a_ol = 1 / 10, 1e5
    assert proches(float(gain_en_boucle_fermee(total)), a_ol / (1 + beta * a_ol), 1e-3)
    zo = float(impedance_de_sortie_en_boucle_fermee(total))
    assert proches(zo, 100 / (1 + beta * a_ol), 1e-4)


def test_le_produit_gain_bande_fixe_les_etages_des_cas_d_examen() -> None:
    """Article « Le produit gain-bande » : 5000 a 50 kHz, 20000 a 200 kHz."""
    assert gain_maximal_par_etage(1e6, 50e3) == 20
    assert nombre_d_etages(5000, 1e6, 50e3) == 3
    assert (20**2 < 5000 <= 20**3) is True
    assert gain_maximal_par_etage(2e6, 200e3) == 10
    assert nombre_d_etages(20000, 2e6, 200e3) == 5
    assert (10**4 < 20000 <= 10**5) is True
    assert proches(12 / 200, 0.06, 1e-12)


def test_les_courants_de_polarisation_et_de_decalage() -> None:
    """Article « Le courant de polarisation » : moyenne et ecart des deux entrees."""
    assert proches(courant_de_polarisation(75e-9, 25e-9), 50e-9, 1e-18)
    assert proches(courant_de_decalage(75e-9, 25e-9), 50e-9, 1e-18)


def test_la_pente_a_l_infini_est_moins_vingt_n_moins_m_decibels_par_decade() -> None:
    """Article « Lire un filtre sur la carte » : `-20 (N - M)` dB/decade."""
    h = 3 * (S + 1) / ((S + 2) * (S + 5) * (S + 7))
    poles, zeros = carte_des_poles_et_zeros(h, S)
    assert (len(poles), len(zeros)) == (3, 1)
    haut, bas = 1e7, 1e6
    pente = 20 * math.log10(
        abs(complex(h.subs(S, 1j * haut)) / complex(h.subs(S, 1j * bas)))
    )
    assert proches(pente, pente_asymptotique(len(zeros), len(poles)), 1e-3)
    assert pente_asymptotique(1, 3) == -40


def test_un_zero_sur_l_axe_imaginaire_annule_le_module_a_cette_pulsation() -> None:
    """Janvier 2022 (3.2) : coupe-bande, `|H(j w_1)| = 0`."""
    w1 = 5
    h = (S**2 + w1**2) / ((S + 1) * (S + 2))
    assert egales(h.subs(S, sp.I * w1), 0)
    assert not egales(h.subs(S, sp.I * w1 * 2), 0)


def test_l_axe_imaginaire_donne_fourier_et_la_limite_est_h_de_zero() -> None:
    """Article « De Laplace a Fourier » : `X(f) = X(s)|_{s = j 2 pi f}`, `H(0)`."""
    t = sp.Symbol("t", positive=True)
    h = 3 / (S + 4)
    assert laplace.est_stable(h, S)
    y = laplace.reponse_a_un_echelon(h, 1, S, t)
    assert egales(sp.limit(y, t, sp.oo), h.subs(S, 0))
    f = sp.Rational(1, 7)
    assert egales(h.subs(S, 2 * sp.pi * sp.I * f), 3 / (2 * sp.pi * sp.I * f + 4))


def test_le_systeme_nodal_modifie_a_la_forme_en_blocs_de_la_fiche() -> None:
    """Article « L'analyse nodale modifiee » : `[[Y, B], [C, 0]] [V, I_S] = [J, E]`."""
    r, c, e, i1 = sp.symbols("R C E I_1")
    circuit: list[Element] = [
        Admittance(1, 2, 1 / r),
        Admittance(2, 0, S * c),
        SourceDeTensionCommandee(3, 0, 2, (2, 0), i1),
        Admittance(3, 0, 1 / r),
        SourceDeTension(1, 0, e, sp.Symbol("I_0")),
    ]
    matrice, membre = systeme_nodal_modifie(circuit, 3)
    assert matrice.shape == (5, 5)
    assert matrice[3:, 3:] == sp.zeros(2, 2)
    assert {abs(v) for v in matrice[:3, 3:] if v != 0} == {1}
    assert membre[4] == e
    inconnues = resoudre(circuit, 3)
    assert egales(inconnues[2], 2 * inconnues[1])
