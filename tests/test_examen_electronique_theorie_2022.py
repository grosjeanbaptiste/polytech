"""Les reponses du corrige manuscrit de janvier 2021-2022 (theorie), recalculees.

La transcription de `ElectroniqueInstrumentation/EXAMENS/CorrTheorie.pdf`
porte des resultats chiffres et deux ecarts. Ces tests refont chaque calcul
depuis les donnees de l'enonce, par un chemin different de celui du
manuscrit -- condition pour qu'ils puissent le contredire.
"""

import pytest
import sympy as sp

from polytech.electronique.domain.bruit import densite_spectrale_unilaterale
from polytech.electronique.domain.friis import (
    facteur_du_premier_etage,
    facteur_en_cascade,
    facteur_maximal_du_second_etage,
)
from polytech.electronique.domain.pole_zero import pente_asymptotique
from polytech.electronique.domain.retroaction import (
    gain_en_boucle_fermee,
    impedance_de_sortie_en_boucle_fermee,
    parametres_h_amplificateur,
    parametres_h_reseau_resistif,
)

MEGOHM = 1e6
MILLIOHM = 1e-3

# --- Question 1 : parametres hybrides et retroaction serie-parallele ------


def test_les_parametres_h_de_l_amplificateur_sont_ceux_du_corrige() -> None:
    """Question 1.2 : `h_11 = R_in`, `h_12 = 0`, `h_21 = -A R_in / R_o`."""
    r_in, r_o, gain = sp.symbols("R_in R_o A_v", positive=True)
    h = parametres_h_amplificateur(r_in, r_o, gain)
    attendue = sp.Matrix([[r_in, 0], [-gain * r_in / r_o, 1 / r_o]])
    assert sp.simplify(h - attendue) == sp.zeros(2, 2)


def test_les_parametres_h_du_reseau_resistif_sont_ceux_de_l_enonce() -> None:
    """Question 1.3 : l'enonce donne `H^(2)`, et le circuit doit le redonner."""
    r1, r2 = sp.symbols("R_1 R_2", positive=True)
    h = parametres_h_reseau_resistif(r1, r2)
    attendue = sp.Matrix(
        [
            [r1 * r2 / (r1 + r2), r1 / (r1 + r2)],
            [-r1 / (r1 + r2), 1 / (r1 + r2)],
        ],
    )
    assert sp.simplify(h - attendue) == sp.zeros(2, 2)


def test_les_valeurs_numeriques_du_reseau_resistif_sont_celles_de_l_enonce() -> None:
    """Question 1.3 : 66,67 MOhm, 1/3, -1/3 et 3,333 nS."""
    h = parametres_h_reseau_resistif(100 * MEGOHM, 200 * MEGOHM)
    assert float(h[0, 0]) == pytest.approx(66.67 * MEGOHM, rel=1e-4)
    assert float(h[0, 1]) == pytest.approx(1 / 3)
    assert float(h[1, 0]) == pytest.approx(-1 / 3)
    assert float(h[1, 1]) == pytest.approx(3.333e-9, rel=1e-3)


def test_la_matrice_globale_est_la_somme_des_deux() -> None:
    """Question 1.3 : la rétroaction serie-parallele additionne les matrices H."""
    amplificateur = parametres_h_amplificateur(1 * MEGOHM, 100.0, 1e5)
    reseau = parametres_h_reseau_resistif(100 * MEGOHM, 200 * MEGOHM)
    globale = amplificateur + reseau
    assert float(globale[0, 0]) == pytest.approx(67.67 * MEGOHM, rel=1e-4)
    assert float(globale[0, 1]) == pytest.approx(1 / 3)
    assert float(globale[1, 0]) == pytest.approx(-1e9, rel=1e-6)
    assert float(globale[1, 1]) == pytest.approx(10e-3, rel=1e-6)


def test_le_rapport_de_boucle_vaut_moins_1477_76() -> None:
    """Question 1.3 : `h_21 / (h_11 h_22)`, le nombre pivot du corrige."""
    globale = parametres_h_amplificateur(
        1 * MEGOHM,
        100.0,
        1e5,
    ) + parametres_h_reseau_resistif(100 * MEGOHM, 200 * MEGOHM)
    boucle = globale[1, 0] / (globale[0, 0] * globale[1, 1])
    assert float(boucle) == pytest.approx(-1477.76, rel=1e-4)


def test_le_gain_en_boucle_fermee_vaut_plus_trois_et_non_moins_trois() -> None:
    """Question 1.3 : le montage est non-inverseur, son gain est positif.

    Le corrige conclut `-3`. Le montage etant un non-inverseur de pont
    `R_1 = 100 M`, `R_2 = 200 M`, son gain vaut `1 + R_2/R_1 = +3` : le
    signe ne peut pas etre negatif. Le calcul le confirme.
    """
    globale = parametres_h_amplificateur(
        1 * MEGOHM,
        100.0,
        1e5,
    ) + parametres_h_reseau_resistif(100 * MEGOHM, 200 * MEGOHM)
    obtenu = float(gain_en_boucle_fermee(globale))
    assert obtenu == pytest.approx(+3.0, rel=1e-2)
    assert obtenu == pytest.approx(1 + 200 / 100, rel=1e-2)


def test_le_rapport_de_boucle_est_negatif_et_non_positif() -> None:
    """Question 1.3 : l'annotation « = +1477,76 » contredit `h_21 = -10^9`.

    `h_11 h_22` est positif ; `h_21` vaut `-10^9`, l'enonce le donne. Le
    rapport est donc negatif, et c'est bien ce signe-la que le denominateur
    du corrige utilise deux lignes plus bas.
    """
    globale = parametres_h_amplificateur(
        1 * MEGOHM,
        100.0,
        1e5,
    ) + parametres_h_reseau_resistif(100 * MEGOHM, 200 * MEGOHM)
    assert float(globale[1, 0]) < 0
    assert float(globale[0, 0] * globale[1, 1]) > 0
    assert float(globale[1, 0] / (globale[0, 0] * globale[1, 1])) < 0


def test_l_impedance_de_sortie_fermee_vaut_200_milliohms() -> None:
    """Question 1.3 : c'est le chiffre que le corrige annonce."""
    globale = parametres_h_amplificateur(
        1 * MEGOHM,
        100.0,
        1e5,
    ) + parametres_h_reseau_resistif(100 * MEGOHM, 200 * MEGOHM)
    obtenue = float(impedance_de_sortie_en_boucle_fermee(globale))
    assert obtenue == pytest.approx(202.6 * MILLIOHM, rel=1e-3)
    assert round(obtenue, 1) == pytest.approx(200 * MILLIOHM)


def test_la_formule_du_cours_annonce_trois_milliohms_et_non_six() -> None:
    """Question 1.3 : `R_o A_cl / A_ol` part de `R_o = 100`, pas de 200.

    Le corrige ecrit « 200 * 3/10^5 » et conclut a 3 mOhm. Le produit est
    6 mOhm ; c'est le facteur qui est faux, pas le resultat.
    """
    assert pytest.approx(3 * MILLIOHM) == 100.0 * 3 / 1e5
    assert pytest.approx(6 * MILLIOHM) == 200.0 * 3 / 1e5


def test_la_retroaction_degrade_l_impedance_de_sortie() -> None:
    """Question 1.3 : 200 mOhm mesures contre 3 mOhm predits par le cours."""
    globale = parametres_h_amplificateur(
        1 * MEGOHM,
        100.0,
        1e5,
    ) + parametres_h_reseau_resistif(100 * MEGOHM, 200 * MEGOHM)
    mesuree = float(impedance_de_sortie_en_boucle_fermee(globale))
    assert mesuree > 100.0 * 3 / 1e5


# --- Question 2 : le derivateur et son bruit ------------------------------


def test_le_montage_a_pour_transfert_un_plus_s_r_c() -> None:
    """Question 2.1 : zero virtuel, `I_f = sC V_i`, `V_out = V_i + R I_f`."""
    s, r, c, tension = sp.symbols("s R C V_i", positive=True)
    courant = s * c * tension
    sortie = tension + r * courant
    assert sp.simplify(sortie / tension - (1 + s * r * c)) == 0


def test_le_bruit_de_la_resistance_arrive_intact_en_sortie() -> None:
    """Question 2.2 : `V_- = 0` annule `I_f`, donc la PSD ne subit rien."""
    r, t, k = sp.symbols("R T k_B", positive=True)
    psd = densite_spectrale_unilaterale(r, t, k)
    assert sp.simplify(psd - 4 * k * r * t) == 0


def test_le_bruit_en_tension_est_filtre_par_le_module_au_carre() -> None:
    """Question 2.3 : `(1 + (2 pi f R C)^2) S_NiNi(f)`."""
    f, r, c = sp.symbols("f R C", positive=True)
    transfert = 1 + sp.I * 2 * sp.pi * f * r * c
    module_au_carre = sp.simplify(sp.Abs(transfert) ** 2)
    attendu = 1 + (2 * sp.pi) ** 2 * f**2 * r**2 * c**2
    assert sp.simplify(sp.expand(module_au_carre - attendu)) == 0


# --- Questions 3 et 4 : lecture des poles et zeros ------------------------


def test_cinq_poles_et_deux_zeros_donnent_moins_soixante_decibels() -> None:
    """Question 3.3 : `(5 - 2) x (-20 dB/decade)`."""
    assert pente_asymptotique(zeros=2, poles=5) == pytest.approx(-60.0)


def test_autant_de_poles_que_de_zeros_donne_une_pente_nulle() -> None:
    """Question 4.1 : deux de chaque, donc 0 dB/decade."""
    assert pente_asymptotique(zeros=2, poles=2) == pytest.approx(0.0)


def test_le_filtre_de_la_question_4_laisse_tout_passer() -> None:
    """Question 4.1 : zeros en `a +- jb`, poles en `-a +- jb`, module 1.

    Le corrige l'affirme par la geometrie -- les distances se compensent.
    Le calcul symbolique le refait sans rien voir.
    """
    a, b, omega = sp.symbols("a b omega", positive=True)
    s = sp.I * omega
    transfert = ((s - (a + sp.I * b)) * (s - (a - sp.I * b))) / (
        (s - (-a + sp.I * b)) * (s - (-a - sp.I * b))
    )
    assert sp.simplify(sp.Abs(transfert) - 1) == 0


def test_le_module_asymptotique_de_la_question_4_vaut_un() -> None:
    """Question 4.1 : la limite en l'infini, calculee et non devinee."""
    a, b, omega = sp.symbols("a b omega", positive=True)
    s = sp.I * omega
    transfert = ((s - (a + sp.I * b)) * (s - (a - sp.I * b))) / (
        (s - (-a + sp.I * b)) * (s - (-a - sp.I * b))
    )
    assert sp.limit(sp.Abs(transfert), omega, sp.oo) == 1


# --- Question 5 : facteur de bruit d'une cascade --------------------------


def test_le_facteur_du_premier_etage_vaut_1_4375() -> None:
    """Question 5.1 : Friis, `F^(1) = 2 - 9/16`."""
    assert facteur_du_premier_etage(2.0, 10.0, 4.0) == pytest.approx(1.4375)


def test_friis_boucle_la_boucle() -> None:
    """Question 5.1 : remettre `F^(1)` dans la formule redonne `F`."""
    premier = facteur_du_premier_etage(2.0, 10.0, 4.0)
    assert facteur_en_cascade(premier, 10.0, 4.0) == pytest.approx(2.0)


def test_le_second_etage_peut_monter_jusqu_a_dix_sept() -> None:
    """Question 5.2 : `F^(1) >= 1` donne `F^(2) <= 1 + (F - 1) A_1^2 = 17`.

    Le corrige ecrit correctement `F^(2) - 1 <= 16` puis conclut 15 : sa
    conclusion ne suit pas de sa propre ligne precedente.
    """
    assert facteur_maximal_du_second_etage(2.0, 4.0) == pytest.approx(17.0)


def test_a_dix_sept_le_premier_etage_est_exactement_a_la_limite() -> None:
    """Question 5.2 : c'est ce qui rend 17 maximal, et 15 trop prudent."""
    assert facteur_du_premier_etage(2.0, 17.0, 4.0) == pytest.approx(1.0)
    assert facteur_du_premier_etage(2.0, 15.0, 4.0) == pytest.approx(1.125)
