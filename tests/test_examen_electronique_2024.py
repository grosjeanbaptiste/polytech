"""Les reponses du corrige manuscrit de janvier 2023-2024, recalculees.

La transcription de
`ElectroniqueInstrumentation/EXAMENS/ExamElecInstru202324S1Corrigev1.pdf`
porte des resultats chiffres et deux lapsus. Ces tests refont chaque calcul
depuis les donnees de l'enonce : ils valident ce que le corrige affirme, et
mesurent ce qu'il ecrit de travers.
"""

import math

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    densite_de_bruit_maximale,
    gain_de_chaine,
    nyquist,
    puissance_de_bruit_en_courant,
)
from polytech.electronique.domain.amplification import gain_maximal_par_etage
from polytech.electronique.domain.filtre import PrototypeButterworth
from polytech.electronique.domain.induction import BoucleDeMesure
from polytech.electronique.domain.laplace import est_stable, reponse_a_un_echelon
from polytech.electronique.domain.nodale import (
    systeme_en_echelle_lc,
    transfert_de_sortie,
)
from polytech.electronique.domain.quadripole import parametres_g_en_l
from polytech.electronique.domain.sallen_key import CelluleSallenKey

NANOHENRY = 1e-9
PICOFARAD = 1e-12
MICROVOLT = 1e-6
NANOVOLT = 1e-9
FEMTOVOLT = 1e-15

# --- Question 1 : induction dans la boucle de mesure -----------------------


def test_le_flux_traverse_la_boucle_selon_le_logarithme_de_a_plus_b_sur_a() -> None:
    """Question 1.1 : l'integrale entre `a` et `a+b` de `dx/x`."""
    mu0, i0, omega, t, ell, a, b = sp.symbols("mu_0 I_0 omega t ell a b", positive=True)
    boucle = BoucleDeMesure(hauteur=ell, distance=a, largeur=b)
    attendu = ell * mu0 * i0 * sp.sin(omega * t) / (2 * sp.pi) * sp.log((a + b) / a)
    assert sp.simplify(boucle.flux(mu0, i0, omega, t) - attendu) == 0


def test_la_tension_mesuree_est_celle_du_cadre_encadre() -> None:
    """Question 1.1 : le cadre encadre du corrige, terme a terme."""
    mu0, i0, omega, t, ell, a, b, v = sp.symbols(
        "mu_0 I_0 omega t ell a b V",
        positive=True,
    )
    boucle = BoucleDeMesure(hauteur=ell, distance=a, largeur=b)
    attendu = v + ell * mu0 * i0 * omega * sp.cos(omega * t) / (2 * sp.pi) * sp.log(
        (a + b) / a,
    )
    assert sp.simplify(boucle.tension_mesuree(v, mu0, i0, omega, t) - attendu) == 0


def test_le_logarithme_en_b_de_l_avant_derniere_ligne_est_un_lapsus() -> None:
    """Question 1.1 : `log((a+b)/b)` ne peut pas sortir de cette integrale."""
    a, b, x = sp.symbols("a b x", positive=True)
    integrale = sp.integrate(1 / x, (x, a, a + b))
    assert sp.simplify(integrale - sp.log((a + b) / a)) == 0
    assert sp.simplify(integrale - sp.log((a + b) / b)) != 0


# --- Question 2 : filtre en echelle LC de Butterworth ----------------------


def test_la_matrice_nodale_est_celle_que_le_corrige_ecrit() -> None:
    """Question 2.1 : quatre inconnues, dont le courant de la source."""
    s, r, bobine, c = sp.symbols("s R L C", positive=True)
    matrice, second_membre = systeme_en_echelle_lc(r, bobine, c, s)
    attendue = sp.Matrix(
        [
            [-1 / r, 1 / r, 0, 1],
            [1 / r, -(1 / r + 1 / (s * bobine)), 1 / (s * bobine), 0],
            [0, 1 / (s * bobine), -(1 / (s * bobine) + s * c + 1 / r), 0],
            [1, 0, 0, 0],
        ],
    )
    assert sp.simplify(matrice - attendue) == sp.zeros(4, 4)
    assert second_membre == sp.Matrix([0, 0, 0, sp.Symbol("V_S")])


def test_le_systeme_nodal_redonne_le_diviseur_d_impedances() -> None:
    """Question 2.1 : deux chemins independants vers la meme transmittance.

    Le systeme matriciel se resout par elimination ; le diviseur, lui, ne
    voit qu'une impedance serie et une impedance de charge. Qu'ils
    coincident est la seule preuve que la matrice n'a pas de faute de
    signe.
    """
    s, r, bobine, c = sp.symbols("s R L C", positive=True)
    charge = 1 / (1 / r + s * c)
    diviseur = sp.cancel(charge / (r + s * bobine + charge))
    assert sp.simplify(transfert_de_sortie(r, bobine, c, s) - diviseur) == 0


def test_le_filtre_en_echelle_perd_la_moitie_de_la_tension_en_continu() -> None:
    """Question 2.1 : source et charge egales forment un pont diviseur."""
    s, r, bobine, c = sp.symbols("s R L C", positive=True)
    transfert = transfert_de_sortie(r, bobine, c, s)
    assert sp.limit(transfert, s, 0) == sp.Rational(1, 2)


def test_le_passe_bas_de_butterworth_donne_41_nanohenrys_et_7_71_picofarads() -> None:
    """Question 2.2 : prototype `g = (sqrt 2, sqrt 2)`, 73 ohms, 400 MHz."""
    prototype = PrototypeButterworth(
        elements=(math.sqrt(2), math.sqrt(2)),
        impedance=73.0,
    )
    inductance, capacite = prototype.en_passe_bas(2 * math.pi * 400e6)
    assert inductance == pytest.approx(41.08 * NANOHENRY, rel=1e-3)
    assert capacite == pytest.approx(7.71 * PICOFARAD, rel=1e-3, abs=0)


def test_les_41_08_millihenrys_du_corrige_sont_un_million_de_fois_trop() -> None:
    """Question 2.2 : le `m` ecrit vaut `n`, comme deux fois deja dans ce corpus."""
    prototype = PrototypeButterworth(
        elements=(math.sqrt(2), math.sqrt(2)),
        impedance=73.0,
    )
    inductance, _ = prototype.en_passe_bas(2 * math.pi * 400e6)
    assert inductance / (41.08e-3) == pytest.approx(1e-6, rel=1e-3)


# --- Question 5 : parametres g du quadripole en L --------------------------


def test_les_deux_parametres_g_donnes_par_l_enonce_se_retrouvent() -> None:
    """Question 5.1 : `g_11` et `g_22` sont la preuve que la matrice est la bonne."""
    z1, z2 = sp.symbols("Z_1 Z_2", positive=True)
    g = parametres_g_en_l(z1, z2)
    assert sp.simplify(g[0, 0] - 1 / (z1 + z2)) == 0
    assert sp.simplify(g[1, 1] - z1 * z2 / (z1 + z2)) == 0


def test_les_deux_parametres_g_manquants_sont_ceux_du_corrige() -> None:
    """Question 5.1 : `g_12 = -Z_2/(Z_1+Z_2)` et `g_21 = Z_2/(Z_1+Z_2)`."""
    z1, z2 = sp.symbols("Z_1 Z_2", positive=True)
    g = parametres_g_en_l(z1, z2)
    assert sp.simplify(g[0, 1] + z2 / (z1 + z2)) == 0
    assert sp.simplify(g[1, 0] - z2 / (z1 + z2)) == 0


def test_la_mise_en_parallele_serie_donne_un_gain_a_vide_unitaire() -> None:
    """Question 5.2 : `G = G1 + G2`, et les deux cellules RC se completent.

    Le corrige s'arrete a la methode. La mener au bout donne un resultat que
    rien n'annoncait : les deux `g_21` se somment exactement a 1.
    """
    s, r, c = sp.symbols("s R C", positive=True)
    haute = parametres_g_en_l(r, 1 / (s * c))
    basse = parametres_g_en_l(1 / (s * c), r)
    assert sp.simplify((haute + basse)[1, 0] - 1) == 0


# --- Question 6 : cellule Sallen-Key --------------------------------------


def test_la_cellule_sallen_key_a_la_pulsation_du_corrige() -> None:
    """Question 6.1 : `omega_0^2 = 1/(R_1 R_2 C_1 C_2)`."""
    cellule = CelluleSallenKey(r1=1131.0, r2=223948.0, c1=10e-9, c2=100e-12)
    assert cellule.pulsation_propre() ** 2 == pytest.approx(3948118751.0, rel=1e-5)
    assert cellule.pulsation_propre() == pytest.approx(62834.06, rel=1e-6)


def test_la_frequence_propre_est_le_nombre_rond_promis_par_l_enonce() -> None:
    """Question 6.1 : l'astuce de l'enonce annonce un nombre rond."""
    cellule = CelluleSallenKey(r1=1131.0, r2=223948.0, c1=10e-9, c2=100e-12)
    assert cellule.frequence_propre() == pytest.approx(10e3, rel=1e-4)


def test_le_facteur_de_qualite_vaut_un_sur_racine_de_deux() -> None:
    """Question 6.1 : `omega_0/Q = (1/C_1)(1/R_1 + 1/R_2)`, d'ou `Q = 0,707`."""
    cellule = CelluleSallenKey(r1=1131.0, r2=223948.0, c1=10e-9, c2=100e-12)
    assert cellule.facteur_de_qualite() == pytest.approx(0.707, rel=1e-3)
    assert cellule.facteur_de_qualite() == pytest.approx(1 / math.sqrt(2), rel=1e-3)


def test_le_gain_continu_de_la_cellule_vaut_un() -> None:
    """Question 6.1 : l'amplificateur est monte en suiveur, donc `mu = 1`."""
    cellule = CelluleSallenKey(r1=1131.0, r2=223948.0, c1=10e-9, c2=100e-12)
    assert cellule.gain() == pytest.approx(1.0)


def test_la_fonction_de_transfert_se_met_sous_la_forme_canonique() -> None:
    """Question 6.1 : les coefficients lus doivent venir du circuit, pas d'une table."""
    s, r1, r2, c1, c2 = sp.symbols("s R_1 R_2 C_1 C_2", positive=True)
    transfert = CelluleSallenKey(r1, r2, c1, c2).fonction_de_transfert(s)
    attendue = 1 / (s**2 * r1 * r2 * c1 * c2 + s * c2 * (r1 + r2) + 1)
    assert sp.simplify(transfert - attendue) == 0


# --- Question 7 : reponse indicielle --------------------------------------


def test_le_systeme_est_stable_si_et_seulement_si_alpha_est_positif() -> None:
    """Question 7.1 : le pole `-alpha` doit etre a partie reelle negative."""
    s, g = sp.symbols("s G", positive=True)
    assert est_stable(g / (s + 1), s)
    assert not est_stable(g / (s - 1), s)
    assert not est_stable(g / s, s)


def test_la_reponse_indicielle_est_l_exponentielle_du_corrige() -> None:
    """Question 7.2 : `y(t) = (G beta/alpha)(1 - exp(-alpha t))`."""
    s, t, g, alpha, beta = sp.symbols("s t G alpha beta", positive=True)
    obtenue = reponse_a_un_echelon(g / (s + alpha), beta, s, t)
    attendue = g * beta / alpha * (1 - sp.exp(-alpha * t))
    assert sp.simplify(obtenue - attendue) == 0


def test_le_pole_integrateur_ajoute_la_rampe_du_corrige() -> None:
    """Question 7.3 : `y(t) = (G beta/alpha)[t + (exp(-alpha t) - 1)/alpha]`."""
    s, t, g, alpha, beta = sp.symbols("s t G alpha beta", positive=True)
    obtenue = reponse_a_un_echelon(g / ((s + alpha) * s), beta, s, t)
    attendue = g * beta / alpha * (t + (sp.exp(-alpha * t) - 1) / alpha)
    assert sp.simplify(obtenue - attendue) == 0


# --- Question 8 : chaine d'acquisition des seize microphones --------------


def test_le_gain_global_de_la_chaine_vaut_16500() -> None:
    """Question 8.1 : porter 0,2 mV a 3,3 V."""
    assert gain_de_chaine(3.3, 0.2e-3) == pytest.approx(16500.0)


def test_seul_un_convertisseur_douze_bits_tient_la_resolution_demandee() -> None:
    """Question 8.2 : le LSB rapporte a l'entree doit rester sous 0,15 microvolt."""
    etendue_rapportee = 6.6 / 16500
    assert bits_necessaires(etendue_rapportee, 0.15 * MICROVOLT) == 12


def test_le_multiplexeur_seize_voies_tient_sous_la_cadence_de_l_adc3() -> None:
    """Question 8.2 : 2 x 400 Hz par voie, seize voies, contre 20 kS/s."""
    cadence = cadence_du_multiplexeur(nyquist(400.0), voies=16)
    assert cadence == pytest.approx(12.8e3)
    assert cadence < 20e3


def test_seul_amp1_passe_sous_la_densite_de_bruit_autorisee() -> None:
    """Question 8.3 : 3 microvolts efficaces sur 400 Hz font 150 nV par racine Hz."""
    plafond = densite_de_bruit_maximale(3 * MICROVOLT, 400.0)
    assert plafond == pytest.approx(150e-9, rel=1e-9)
    assert plafond > 75e-9
    assert plafond < 200e-9
    assert plafond < 0.9e-6
    assert plafond < 15e-6


def test_le_bruit_en_courant_d_amp1_est_negligeable() -> None:
    """Question 8.3 : 50 fA par racine Hz dans 40 ohms, sur 400 Hz."""
    puissance = puissance_de_bruit_en_courant(50e-15, 40.0, 400.0)
    assert math.sqrt(puissance) == pytest.approx(40e3 * FEMTOVOLT, rel=1e-9, abs=0)
    assert math.sqrt(puissance) == pytest.approx(0.04 * NANOVOLT, rel=1e-9, abs=0)
    assert math.sqrt(puissance) < 3 * MICROVOLT


def test_le_premier_etage_consomme_un_quart_du_budget_de_bruit() -> None:
    """Question 8.4 : 2,25 microvolts carres sur les 9 disponibles."""
    consomme = (75e-9) ** 2 * 400.0
    assert consomme == pytest.approx(2.25 * MICROVOLT**2, rel=1e-9, abs=0)
    assert (3 * MICROVOLT) ** 2 - consomme == pytest.approx(
        6.75 * MICROVOLT**2, rel=1e-9, abs=0
    )


def test_un_seul_etage_suffit_a_porter_le_gain_de_16500() -> None:
    """Question 8.4 : AMP1 tient 20 000 sur 400 Hz, donc plus que 16 500."""
    assert gain_maximal_par_etage(8e6, 400.0) == pytest.approx(20000.0)
    assert gain_maximal_par_etage(8e6, 400.0) > 16500.0


def test_le_second_etage_a_les_coudees_franches() -> None:
    """Question 8.4 : le gain amont divise le bruit rapporte par 16 500 au carre."""
    restant = 6.75 * MICROVOLT**2
    densite = restant * 16500.0**2 / 400.0
    assert densite == pytest.approx(4.594e-6, rel=1e-3)
    assert math.sqrt(densite) == pytest.approx(2143e-6, rel=1e-3)
