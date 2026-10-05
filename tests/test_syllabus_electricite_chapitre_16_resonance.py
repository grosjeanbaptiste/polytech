"""Syllabus d'electricite generale, chapitre 16 : resonance, pertes, transformateur.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 16-38 a 16-55, lues sur les pages rendues.

Les nombres du cours : `omega_a = 0,856 omega_r` et `0,997 omega_r` pour
`Q = 1` et `Q = 3` (page 16-43) ; les profondeurs de penetration du
cuivre et les quatre rapports de resistance de la page 16-46 ; la note de
la page 16-53, `(omega tau')^2 = 0,4 . 10^-6`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.branche_sinusoidale import (
    impedance_de_la_branche,
)
from polytech.electromagnetisme.domain.circuits.pertes_des_composants import (
    bobine_en_parallele,
    condensateur_en_parallele,
    facteur_de_dissipation,
    impedance_d_une_resistance_reelle,
    resistance_en_serie_equivalente,
)
from polytech.electromagnetisme.domain.circuits.resistivite import METAUX
from polytech.electromagnetisme.domain.circuits.resonance import (
    impedance_reduite_parallele,
    impedance_reduite_serie,
    maximum_a_l_antiresonance,
    pulsation_d_antiresonance,
    pulsations_de_coupure,
)
from polytech.electromagnetisme.domain.circuits.sinusoidaux import (
    en_parallele,
    facteur_de_qualite,
    pulsation_de_resonance_serie,
)
from polytech.electromagnetisme.domain.circuits.transformateur import (
    TransformateurParfait,
    tension_d_un_enroulement,
)
from polytech.electromagnetisme.domain.diffusion import (
    residu_de_diffusion_cylindrique,
)
from polytech.electromagnetisme.domain.effet_pelliculaire import epaisseur_de_peau
from polytech.electromagnetisme.domain.magnetostatique.bobines_sur_noyau import (
    BobinesSurNoyau,
)
from polytech.electromagnetisme.domain.resistance_pelliculaire import (
    rapport_des_resistances_d_un_fil,
    resistance_effective,
)
from polytech.electromagnetisme.domain.symboles import RAYON, TEMPS

R, L, C, OMEGA = sp.symbols("R L C omega", positive=True)
X = sp.Symbol("x", positive=True)
CUIVRE = METAUX["cuivre"][0]


def test_l_impedance_reduite_de_la_branche_serie() -> None:
    """(16.21.7) : `z_s = Z_s / R = 1 + j Q (x - 1 / x)`, `x = omega / omega_r`."""
    propre = pulsation_de_resonance_serie(L, C)
    qualite = facteur_de_qualite(R, L, C)
    physique = impedance_de_la_branche(R, L, C, X * propre) / R
    assert sp.simplify(physique - impedance_reduite_serie(qualite, X)) == 0
    # (16.21.4) : a la resonance, l'impedance se reduit a `R`.
    assert impedance_reduite_serie(qualite, 1) == 1


def test_le_facteur_de_qualite_est_un_facteur_de_surtension() -> None:
    """(16.21.6) et (16.21.14) : `U_L(omega_r) / U = (1 / R) racine(L / C) = Q`."""
    propre = pulsation_de_resonance_serie(L, C)
    courant = sp.Symbol("U", positive=True) / R
    aux_bornes_de_la_bobine = propre * L * courant
    aux_bornes_du_condensateur = courant / (propre * C)
    qualite = facteur_de_qualite(R, L, C)
    applique = sp.Symbol("U", positive=True)
    assert sp.simplify(aux_bornes_de_la_bobine / applique - qualite) == 0
    assert sp.simplify(aux_bornes_du_condensateur / applique - qualite) == 0


def test_le_facteur_de_qualite_compare_l_energie_emmagasinee_a_la_dissipee() -> None:
    """Page 16-41 : `2 pi` fois l'energie moyenne totale sur l'energie d'une periode."""
    courant = sp.Symbol("I", positive=True)
    propre = pulsation_de_resonance_serie(L, C)
    magnetique = L * courant**2 / 2
    electrique = (courant / propre) ** 2 / (2 * C)
    assert sp.simplify(magnetique - electrique) == 0
    dissipee = R * courant**2 * 2 * sp.pi / propre
    rapport = 2 * sp.pi * (magnetique + electrique) / dissipee
    assert sp.simplify(rapport - facteur_de_qualite(R, L, C)) == 0


@pytest.mark.parametrize("qualite", [1, 3, 20])
def test_la_largeur_de_bande_est_l_inverse_du_facteur_de_qualite(qualite: int) -> None:
    """(16.21.11) a (16.21.13) : `z = racine(2)` aux coupures, distantes de `1 / Q`."""
    basse, haute = pulsations_de_coupure(qualite)
    assert basse < 1 < haute
    for reduite in (basse, haute):
        module = sp.Abs(impedance_reduite_serie(qualite, reduite))
        assert sp.simplify(module - sp.sqrt(2)) == 0
    assert sp.simplify(haute - basse - sp.Rational(1, qualite)) == 0
    # La resonance est la moyenne geometrique des deux coupures.
    assert sp.simplify(basse * haute - 1) == 0


def test_un_facteur_de_qualite_non_positif_n_a_pas_de_bande() -> None:
    with pytest.raises(ValueError, match="facteur de qualite nul ou negatif"):
        pulsations_de_coupure(0)


def test_l_impedance_reduite_du_circuit_antiresonnant() -> None:
    """(16.21.15) et (16.21.19) : `C` en parallele avec `R` et `L` en serie."""
    propre = pulsation_de_resonance_serie(L, C)
    qualite = facteur_de_qualite(R, L, C)
    pulsation = X * propre
    serie = R + sp.I * pulsation * L
    physique = en_parallele(serie, 1 / (sp.I * pulsation * C)) / R
    reduite = impedance_reduite_parallele(qualite, X)
    assert sp.simplify(physique - reduite) == 0
    # Page 16-43 : en courant continu, l'impedance se reduit a `R`.
    assert impedance_reduite_parallele(3, 0) == 1


def test_les_pulsations_d_antiresonance_de_la_page_16_43() -> None:
    """(16.21.20) : `0,856 omega_r` pour `Q = 1`, `0,997 omega_r` pour `Q = 3`."""
    assert float(pulsation_d_antiresonance(1)) == pytest.approx(0.856, abs=5e-4)
    assert float(pulsation_d_antiresonance(3)) == pytest.approx(0.997, abs=5e-4)


@pytest.mark.parametrize("qualite", [1, 3])
def test_le_maximum_de_l_impedance_est_bien_la_ou_le_syllabus_le_place(
    qualite: int,
) -> None:
    """(16.21.21), contre un balayage du module de (16.21.19).

    La figure 16.21.5 le montre vers 1,5 pour `Q = 1` et vers 9,5 pour
    `Q = 3`.
    """
    module = sp.lambdify(X, sp.Abs(impedance_reduite_parallele(qualite, X)))
    balayage = [(module(k / 2000), k / 2000) for k in range(1, 4000)]
    sommet, abscisse = max(balayage)
    assert abscisse == pytest.approx(
        float(pulsation_d_antiresonance(qualite)), abs=1e-3
    )
    assert sommet == pytest.approx(float(maximum_a_l_antiresonance(qualite)), rel=1e-5)
    assert sommet == pytest.approx({1: 1.5, 3: 9.5}[qualite], abs=0.05)


def test_un_circuit_trop_amorti_ne_devient_pas_antiresonnant() -> None:
    """Note de la page 16-42 : le second membre de (16.21.18) devient negatif."""
    with pytest.raises(ValueError, match="pas d'antiresonance"):
        pulsation_d_antiresonance(sp.Rational(1, 2))
    # Le seuil est `Q^2 = racine(2) - 1`, soit `Q = 0,64` : 0,6 est en dessous.
    with pytest.raises(ValueError, match="pas d'antiresonance"):
        pulsation_d_antiresonance(sp.Rational(3, 5))
    assert pulsation_d_antiresonance(sp.Rational(13, 20)).is_positive


def test_a_grand_facteur_de_qualite_le_bouchon_vaut_q_carre_fois_r() -> None:
    """(16.21.23) exacte, (16.21.24) approchee : `z_p ~ Q^2 / z_s`."""
    qualite = sp.Symbol("Q", positive=True)
    serie = impedance_reduite_serie(qualite, X)
    parallele = impedance_reduite_parallele(qualite, X)
    exacte = (qualite + sp.I * qualite**2 * X) / (sp.I * X * serie)
    assert sp.simplify(parallele - exacte) == 0
    assert float(maximum_a_l_antiresonance(100)) == pytest.approx(100**2, rel=1e-3)
    assert float(pulsation_d_antiresonance(100)) == pytest.approx(1.0, abs=1e-4)


def test_les_profondeurs_de_penetration_du_cuivre() -> None:
    """(16.22.1) et page 16-46 : 9,3 mm a 50 Hz, 2,1 mm a 1 kHz, puis par decades."""
    conductivite = 1 / CUIVRE
    assert epaisseur_de_peau(50.0, conductivite) == pytest.approx(9.3e-3, rel=5e-3)
    assert epaisseur_de_peau(1e3, conductivite) == pytest.approx(2.1e-3, rel=2e-2)
    for frequence, lue in ((1e5, 0.2e-3), (1e7, 0.02e-3), (1e9, 0.002e-3)):
        assert epaisseur_de_peau(frequence, conductivite) == pytest.approx(
            lue, rel=0.05
        )


def test_la_resistance_effective_est_celle_d_une_coque() -> None:
    """(16.22.2) : `R_CA = rho l / (perimetre . delta)`."""
    assert resistance_effective(CUIVRE, 10.0, 0.05, 1e-4) == pytest.approx(
        CUIVRE * 10.0 / (0.05 * 1e-4)
    )
    with pytest.raises(ValueError, match="coque sans section"):
        resistance_effective(CUIVRE, 10.0, 0.05, 0.0)


@pytest.mark.parametrize(
    ("rayon_sur_peau", "lu", "tolerance"),
    [(0.4, 1.001, 1e-3), (1.0, 1.02, 2e-3), (10.0, 5.25, 0.02), (200.0, 100.0, 0.5)],
)
def test_les_quatre_rapports_de_resistance_de_la_page_16_46(
    rayon_sur_peau: float, lu: float, tolerance: float
) -> None:
    """La resistance en sinusoidal rapportee a celle en continu, fil circulaire."""
    rapport = rapport_des_resistances_d_un_fil(rayon_sur_peau)
    assert rapport == pytest.approx(lu, abs=tolerance)


def test_la_coque_au_perimetre_moyen_approche_le_rapport_exact() -> None:
    """(16.22.2) « ou mieux, le perimetre moyen de la coque fictive »."""
    rayon, peau = 10.0, 1.0
    coque = resistance_effective(1.0, 1.0, 2 * math.pi * (rayon - peau / 2), peau)
    continu = 1.0 / (math.pi * rayon**2)
    assert coque / continu == pytest.approx(
        rapport_des_resistances_d_un_fil(rayon / peau), rel=2e-3
    )


def test_le_profil_de_bessel_satisfait_la_diffusion_du_chapitre_15() -> None:
    """(15.14.3) en sinusoidal : `J_0(k r) exp(j omega t)`, `k^2 = -j omega mu / rho`.

    C'est de ce profil que sort le rapport des resistances.
    """
    mu, resistivite = sp.symbols("mu rho_e", positive=True)
    nombre = sp.sqrt(-sp.I * OMEGA * mu / resistivite)
    courant = sp.besselj(0, nombre * RAYON) * sp.exp(sp.I * OMEGA * TEMPS)
    assert residu_de_diffusion_cylindrique(courant, mu, resistivite) == 0


def test_un_rapport_de_rayon_non_positif_est_refuse() -> None:
    with pytest.raises(ValueError, match="rayon nul ou negatif"):
        rapport_des_resistances_d_un_fil(0.0)


def test_le_condensateur_reel_en_serie_et_en_parallele() -> None:
    """(16.23.1) a (16.23.5) : memes impedances, `d = 1 / (omega C_p R_p)`."""
    serie, capacite = sp.symbols("R_s C_s", positive=True)
    pertes = facteur_de_dissipation(OMEGA, capacite, serie)
    assert pertes == OMEGA * capacite * serie
    parallele, equivalente = condensateur_en_parallele(serie, capacite, pertes)
    en_serie_ = serie - sp.I / (OMEGA * capacite)
    en_parallele_ = en_parallele(parallele, 1 / (sp.I * OMEGA * equivalente))
    assert sp.simplify(en_serie_ - en_parallele_) == 0
    assert sp.simplify(pertes - 1 / (OMEGA * equivalente * parallele)) == 0


def test_un_bon_condensateur_garde_sa_capacite() -> None:
    """(16.23.6) : pour `d` sous le millieme, `R_p ~ R_s / d^2` et `C_p ~ C_s`."""
    parallele, equivalente = condensateur_en_parallele(2.0, 1e-6, 1e-3)
    assert parallele == pytest.approx(2.0 / 1e-6, rel=1e-5)
    assert equivalente == pytest.approx(1e-6, rel=1e-5)


def test_le_facteur_de_dissipation_est_p_sur_la_valeur_absolue_de_q() -> None:
    """Page 16-49 : `d = cotg |phi|`, soit `P / |Q|` ; CEI : `P / racine(S^2 - P^2)`."""
    serie, capacite, pulsation, courant = 2.0, 1e-4, 100.0, 3.0
    active = serie * courant**2
    reactive = -(courant**2) / (pulsation * capacite)
    apparente = math.hypot(active, reactive)
    pertes = facteur_de_dissipation(pulsation, capacite, serie)
    assert pertes == pytest.approx(active / abs(reactive))
    assert pertes == pytest.approx(active / math.sqrt(apparente**2 - active**2))


def test_la_bobine_reelle_en_serie_et_en_parallele() -> None:
    """(16.24.1) et (16.24.2) : `Q = omega L_s / R_s = R_p / (omega L_p)`."""
    serie, inductance = sp.symbols("R_s L_s", positive=True)
    qualite = OMEGA * inductance / serie
    parallele, equivalente = bobine_en_parallele(serie, inductance, qualite)
    en_serie_ = serie + sp.I * OMEGA * inductance
    en_parallele_ = en_parallele(parallele, sp.I * OMEGA * equivalente)
    assert sp.simplify(en_serie_ - en_parallele_) == 0
    assert sp.simplify(qualite - parallele / (OMEGA * equivalente)) == 0
    # (16.24.3) : a grand `Q`, `R_p = Q^2 R_s` et `L_p ~ L_s`.
    forte, voisine = bobine_en_parallele(2.0, 1e-3, 1e3)
    assert forte == pytest.approx(1e6 * 2.0, rel=1e-5)
    assert voisine == pytest.approx(1e-3, rel=1e-5)


def test_un_facteur_nul_n_a_pas_de_schema_parallele() -> None:
    with pytest.raises(ValueError, match="facteur nul ou negatif"):
        condensateur_en_parallele(2.0, 1e-6, 0.0)
    with pytest.raises(ValueError, match="facteur nul ou negatif"):
        bobine_en_parallele(2.0, 1e-3, 0.0)


def test_l_impedance_d_une_resistance_reelle() -> None:
    """(16.25.1), le circuit antiresonnant, et son approximation (16.25.2)."""
    exacte = impedance_d_une_resistance_reelle(R, L, C, OMEGA)
    association = en_parallele(R + sp.I * OMEGA * L, 1 / (sp.I * OMEGA * C))
    assert sp.simplify(exacte - association) == 0
    approchee = R + sp.I * OMEGA * (L - C * R**2)
    ecart = sp.series(exacte - approchee, OMEGA, 0, 2).removeO()
    assert sp.simplify(ecart) == 0


def test_une_petite_resistance_est_inductive_une_grande_capacitive() -> None:
    """Page 16-52 : le signe de `L - C R^2`."""
    petite = impedance_d_une_resistance_reelle(1.0, 1e-6, 1e-12, 1e3)
    grande = impedance_d_une_resistance_reelle(1e6, 1e-6, 1e-12, 1e3)
    assert sp.im(petite) > 0
    assert sp.im(grande) < 0


def test_les_schemas_serie_et_parallele_d_une_resistance() -> None:
    """(16.25.8) : `R_s = R_p / (1 + omega^2 tau'^2)` et `tau = -tau'`."""
    parallele, capacite = sp.symbols("R_p C_p", positive=True)
    serie, constante = resistance_en_serie_equivalente(parallele, capacite, OMEGA)
    admittance = 1 / parallele + sp.I * OMEGA * capacite
    assert sp.simplify(serie * (1 + sp.I * OMEGA * constante) - 1 / admittance) == 0
    assert constante == -parallele * capacite


def test_la_note_de_la_page_16_53() -> None:
    """`tau' = 100 ns`, `f = 1 000 Hz` : `(omega tau')^2 = 0,4 . 10^-6`."""
    serie, _ = resistance_en_serie_equivalente(1e3, 1e-7 / 1e3, 2 * math.pi * 1e3)
    assert 1e3 / serie - 1 == pytest.approx(0.4e-6, rel=0.02)


PARFAIT = TransformateurParfait(primaire=400, secondaire=100)


def test_le_transformateur_parfait_change_tension_et_courant() -> None:
    """(16.26.2) `u_2 = (N_2 / N_1) u_1`, (16.26.4) `i_2 = -(N_1 / N_2) i_1`."""
    assert PARFAIT.rapport() == sp.Rational(1, 4)
    assert PARFAIT.tension_secondaire(230) == sp.Rational(230, 4)
    assert PARFAIT.courant_secondaire(2) == -8
    # (16.26.5) : `p_2 = -p_1`, la puissance se conserve.
    recue = PARFAIT.tension_secondaire(230) * PARFAIT.courant_secondaire(2)
    assert recue == -230 * 2


def test_le_transformateur_parfait_change_une_impedance() -> None:
    """(16.26.6) : `u_2 / i_2 = -(N_2 / N_1)^2 u_1 / i_1`, ici une charge de 5 ohms."""
    assert PARFAIT.impedance_vue_du_primaire(5) == 80
    tension = PARFAIT.tension_secondaire(160)
    courant = PARFAIT.courant_secondaire(2)
    assert tension / courant == -(PARFAIT.rapport() ** 2) * sp.Rational(160, 2)
    assert -tension / courant == 5


def test_un_enroulement_sans_spire_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre de spires nul ou negatif"):
        TransformateurParfait(primaire=0, secondaire=100)


def test_le_transformateur_parfait_est_la_limite_des_bobines_couplees() -> None:
    """Section 10.15 : sans dispersion, `u_2 / u_1 = L_12 / L_1 = N_2 / N_1`."""
    bobines = BobinesSurNoyau((400, 100), 5e5, (math.inf, math.inf))
    assert bobines.mutuelle() / bobines.propre(0) == pytest.approx(
        float(PARFAIT.rapport())
    )


def test_le_transformateur_reel_ajoute_resistance_et_dispersion() -> None:
    """(16.26.8) contre les inductances du chapitre 10 : `l_1 = N_1^2 / R'_1`.

    `u_1 = r_1 i_1 + L_1 di_1/dt + L_12 di_2/dt`, et le flux commun vaut
    `(N_1 i_1 + N_2 i_2) / R_12` (16.26.3).
    """
    bobines = BobinesSurNoyau((400, 100), 5e5, (2e7, 5e6))
    premier, second = sp.sin(3 * TEMPS), sp.cos(3 * TEMPS) / 2
    commun = (400 * premier + 100 * second) / 5e5
    resistance = 0.7
    par_le_flux = tension_d_un_enroulement(
        resistance, bobines.de_dispersion(0), 400, premier, commun
    )
    par_les_inductances = resistance * premier
    par_les_inductances += bobines.propre(0) * sp.diff(premier, TEMPS)
    par_les_inductances += bobines.mutuelle() * sp.diff(second, TEMPS)
    ecart = sp.lambdify(TEMPS, par_le_flux - par_les_inductances)
    assert all(abs(ecart(t)) < 1e-9 for t in (0.0, 0.4, 1.3))
