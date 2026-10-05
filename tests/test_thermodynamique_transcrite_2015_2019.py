"""Recalcule les corriges transcrits de thermodynamique, de 2015 a 2019.

Quatre transcriptions de `transcriptions/Thermodynamique/EXAMENS/` sont
relues par le calcul : decembre 2015, mai 2016, mai 2017 et mai 2019. Une
transcription est une lecture, et une lecture se trompe en silence : chaque
resultat est donc refait depuis les operandes que le corrige ecrit.

Trois de ces corriges ne donnent que des reponses. Les donnees qui leur
manquent (cp, k, rendements, puissances imposees) viennent alors de l'enonce
du meme examen, dans `cours-markdown/Thermodynamique/EXAMENS/`, et chaque
docstring nomme celui qu'elle emploie. Aucune propriete de fluide ne vient
d'ailleurs : le facteur kcal -> kJ lui-meme est deduit des corriges et
seulement encadre.
"""

import math

import pytest

ZERO_CELSIUS = 273.15
EXPOSANT = 0.4 / 1.4
CP_AIR = 1.005
ARRONDI = 1e-3
LECTURE = 3e-3
KCAL_MIN, KCAL_MAX = 4.18, 4.19

# --- Decembre 2015 : Exam_2015_Decembre_Exercice_cote_Reponses -------------

T_2015 = (298.15, 701.72, 877.13, 877.13)
W_2015 = (598.9, 0.0, -1114.9, 0.0)
Q_2015 = (0.0, 260.3, 1114.9, -859.2)
DU_2015 = (598.9, 260.3, 0.0, -859.2)
# Enonce Exam_2015_Decembre_Exercice_cote_Enonce :
# 2 kg de N2, cV = 742 et r = 297 J/(kg.K).
M_CV_2015 = 2 * 0.742
M_R_2015 = 2 * 0.297


def test_decembre_2015_premier_principe_par_transformation() -> None:
    """Decembre 2015, question 1 : deltaU = w + q sur chaque transformation."""
    for travail, chaleur, energie in zip(W_2015, Q_2015, DU_2015, strict=True):
        assert travail + chaleur == pytest.approx(energie, abs=0.05)
    assert sum(DU_2015) == pytest.approx(0.0, abs=0.05)


def test_decembre_2015_travail_net_et_rendement() -> None:
    """Decembre 2015, question 1 : Wnet = somme des w, rendement = |Wnet| / Q recue.

    Le corrige ecrit -515.99 la ou la somme des w du tableau, arrondis au
    dixieme, donne -516.0 ; le rendement 0.3752134 est celui de 515.99.
    """
    assert sum(W_2015) == pytest.approx(-515.99, abs=0.05)
    assert sum(W_2015) == pytest.approx(-sum(Q_2015), abs=0.05)
    recue = Q_2015[1] + Q_2015[2]
    assert 515.99 / recue == pytest.approx(0.3752134, rel=1e-5)


def test_decembre_2015_etats_depuis_l_enonce() -> None:
    """Decembre 2015, question 1 : T, p et v des etats 1 a 4.

    W12 = 598.9 kJ et Q23 = 260.3 kJ sont des donnees de l'enonce ; les
    volumes sont arrondis au centieme, d'ou une tolerance de 1 %.
    """
    t1, t2, t3, t4 = T_2015
    assert t1 + W_2015[0] / M_CV_2015 == pytest.approx(t2, rel=1e-4)
    assert t2 + Q_2015[1] / M_CV_2015 == pytest.approx(t3, rel=1e-4)
    assert t1 * 20.00 ** (0.4 / 1.4) == pytest.approx(t2, rel=1e-4)
    assert 20.00 * t3 / t2 == pytest.approx(25.00, rel=ARRONDI)
    assert 1.00 * t4 / t1 == pytest.approx(2.94, rel=ARRONDI)
    assert M_R_2015 * t1 / 100 == pytest.approx(1.77, rel=ARRONDI)
    assert M_R_2015 * t2 / 2000 == pytest.approx(0.21, rel=1e-2)
    assert M_CV_2015 * (t1 - t4) == pytest.approx(Q_2015[3], rel=ARRONDI)


def test_decembre_2015_detente_isotherme() -> None:
    """Decembre 2015, question 1 : w34 = -m r T ln(p3 / p4)."""
    travail = -M_R_2015 * T_2015[2] * math.log(25.00 / 2.94)
    assert travail == pytest.approx(W_2015[2], rel=ARRONDI)


def test_decembre_2015_chaleur_de_reaction_a_1500_k() -> None:
    """Decembre 2015, question 2 : deltaH(1500 K) par Kirchhoff, cP = A + B T.

    Enthalpies de formation et coefficients A, B : tableau de l'enonce, pour
    CO + 1/2 O2 -> CO2.
    """
    delta_h0 = -94052 - (-26416)
    delta_a = 10.442 - 6.73 - 0.5 * 6.94
    delta_b = 0.0022 - 0.0010 - 0.5 * 0.0011
    integrale = delta_a * (1500 - 298.15) + delta_b / 2 * (1500**2 - 298.15**2)
    assert delta_h0 + integrale == pytest.approx(-66642.8, rel=1e-5)


def test_decembre_2015_loi_de_la_constante_d_equilibre() -> None:
    """Decembre 2015, question 2 : la loi repasse par K(1500 K) = 2,05e5.

    Le rapport deltaH / pente vaut 1.987, soit R en cal/(mol.K) : le corrige
    ne l'ecrit pas, on verifie seulement qu'il est encadre.
    """
    assert pytest.approx(math.log(2.05e5), rel=ARRONDI) == 33539 / 1500 - 10.129
    assert 1.98 < 66642.8 / 33539 < 1.99


def test_decembre_2015_temperature_pour_99_pour_cent() -> None:
    """Decembre 2015, question 2 : T = 2035 K suppose un air a 20 % d'O2.

    L'enonce donne 1 mole de CO, 3 moles d'air et 99 % de conversion, mais
    pas la composition de l'air. La valeur transcrite se retrouve avec 20 %
    d'oxygene ; avec 21 %, on trouverait 2051 K. L'hypothese est donc celle
    du corrige, et le test la nomme au lieu de la taire.
    """

    def temperature(fraction_o2: float) -> float:
        oxygene = 3 * fraction_o2 - 0.99 / 2
        total = 0.01 + 0.99 + oxygene + 3 * (1 - fraction_o2)
        constante = (0.99 / 0.01) / math.sqrt(oxygene / total)
        return 33539 / (math.log(constante) + 10.129)

    assert temperature(0.20) == pytest.approx(2035, rel=ARRONDI)
    assert temperature(0.21) == pytest.approx(2051, rel=ARRONDI)


# --- Mai 2016 : Exam_2016_Mai_Exercice_cote_Reponses -----------------------

T_2016 = (20.0, 191.15, 326.27, 800.0, 443.98, 308.86)
DEBIT_AIR_2016 = 27.95
H_PAC = {"7": 355.0, "8isos": 387.5, "8": 401.5, "9": 265.0, "10": 265.0}
DEBIT_R12 = 90.15


def test_mai_2016_temperatures_du_cycle_de_joule() -> None:
    """Mai 2016, question 1 : T2, T5 et T6, qui confirment 443,98 et 308,86.

    Enonce Exam_2016_Mai_Exercice_cote_Enonce :
    compression isentropique de taux 5, turbine de
    rendement isentropique 0.9, k = 1.4, echangeur E adiabatique.
    """
    t1, t2, t3, t4, t5, t6 = T_2016
    assert (t1 + ZERO_CELSIUS) * 5**EXPOSANT - ZERO_CELSIUS == pytest.approx(
        t2, abs=0.01
    )
    t5_isos = (t4 + ZERO_CELSIUS) * (1 / 5) ** EXPOSANT - ZERO_CELSIUS
    assert t4 - 0.9 * (t4 - t5_isos) == pytest.approx(t5, abs=0.01)
    assert t5 - (t3 - t2) == pytest.approx(t6, abs=0.01)


def test_mai_2016_debit_flux_puissance_et_rendement() -> None:
    """Mai 2016, question 1 : debit d'air, flux chaud, puissance nette, rendement.

    Enonce Exam_2016_Mai_Exercice_cote_Enonce :
    cP = 1005 J/(kg.K), turbine de 10 MW, 1 MW de
    chaleur enleve au compresseur.
    """
    t1, t2, t3, t4, t5, _ = T_2016
    assert 10_000 / (CP_AIR * (t4 - t5)) == pytest.approx(DEBIT_AIR_2016, rel=ARRONDI)
    flux_chaud = DEBIT_AIR_2016 * CP_AIR * (t4 - t3) / 1000
    assert flux_chaud == pytest.approx(13.307, rel=ARRONDI)
    compresseur = DEBIT_AIR_2016 * CP_AIR * (t2 - t1) / 1000 + 1
    assert compresseur - 10 == pytest.approx(-4.192, rel=ARRONDI)
    assert pytest.approx(0.315, rel=ARRONDI) == 4.192 / 13.307


def test_mai_2016_pompe_a_chaleur() -> None:
    """Mai 2016, PAC : debit de R12, flux a la source moyenne et COP.

    Enonce Exam_2016_Mai_Exercice_cote_Enonce :
    la puissance nette du cycle de Joule entraine le
    compresseur, et son rejet a la source froide alimente l'evaporateur.
    """
    travail = H_PAC["8"] - H_PAC["7"]
    assert 4192 / travail == pytest.approx(DEBIT_R12, rel=ARRONDI)
    flux = DEBIT_R12 * (H_PAC["9"] - H_PAC["8"]) / 1000
    assert flux == pytest.approx(-12.305, rel=ARRONDI)
    assert pytest.approx(2.94, rel=2e-3) == 12.305 / 4.192
    rejet_joule = DEBIT_AIR_2016 * CP_AIR * (T_2016[5] - T_2016[0])
    evaporateur = DEBIT_R12 * (H_PAC["7"] - H_PAC["10"])
    assert evaporateur == pytest.approx(rejet_joule, rel=ARRONDI)


def test_mai_2016_l_enthalpie_8_isos_reste_douteuse() -> None:
    """Mai 2016, PAC : 387,5 ou 387,15, le calcul ne tranche pas la lecture.

    Le rendement isentropique 0.7 de l'enonce donne 387,55 : plus pres de
    387,5 que de 387,15, mais h8 = 401,5 est lui-meme lu sur un diagramme.
    Le marqueur de doute reste donc dans la transcription.
    """
    attendu = H_PAC["7"] + 0.7 * (H_PAC["8"] - H_PAC["7"])
    assert attendu == pytest.approx(387.55)
    assert abs(attendu - H_PAC["8isos"]) < abs(attendu - 387.15)


def test_mai_2016_rendements_des_turbines_a_vapeur() -> None:
    """Mai 2016, question 2 : rendements 0,85 (THP) et egaux (TBP1, TBP).

    Enonce Exam_2016_Mai_Exercice_cote_Enonce :
    THP de rendement 0.85, et TBP1 de meme rendement
    que la TBP complete. Les enthalpies sont lues sur un diagramme.
    """
    assert pytest.approx(0.85, rel=ARRONDI) == (819 - 681.3) / (819 - 657)
    premier_etage = (842 - 717.8) / (842 - 704)
    assert pytest.approx(premier_etage, rel=ARRONDI) == (842 - 588) / (842 - 560)


def test_mai_2016_le_manuscrit_lit_h4_a_une_kilocalorie_pres() -> None:
    """Mai 2016, question 2 : h4 = 588 contre 589,1 par les titres de l'enonce.

    L'enonce donne un titre en vapeur de 0.92 en 4 isos et de 0.97 en 4. La
    chaleur latente deduite de h4isos = 560 conduit a 589,1 : l'ecart est une
    lecture de diagramme du manuscrit, la transcription est fidele.
    """
    latente = (560 - 25) / 0.92
    assert 25 + 0.97 * latente == pytest.approx(589.1, rel=1e-4)
    assert 25 + 0.97 * latente == pytest.approx(588, rel=LECTURE)


def test_mai_2016_le_manuscrit_rechauffe_neuf_dixiemes_du_debit() -> None:
    """Mai 2016, question 2 : h7 = 96,06 vient d'un bilan a 0,9 qm cote eau.

    Enonce : qm'/qm = 0.1, et le soutirage retourne au condenseur, donc la
    pompe refoule tout le debit qm. Le manuscrit n'en rechauffe que 0,9 qm ;
    avec qm on trouverait 89,20. La transcription est fidele (relue a
    150 dpi : « 96,06 » deux fois), c'est la source qui s'ecarte de l'enonce.
    """
    cede = 0.1 * (717.8 - 100)
    assert 27.42 + cede / 0.9 == pytest.approx(96.06, abs=0.01)
    assert 27.42 + cede == pytest.approx(89.20, abs=0.01)


# --- Mai 2017 : Exam_2017_Mai_Exercice_cote_Reponses --------------------------------

H_2017 = (379.0, 387.0, 414.0, 410.8, 439.0, 256.0, 256.0, 220.5, 212.5, 212.5, 398.0)
QM1_2017, QM2_2017 = 0.015, 0.06
CP_AIR_2017 = 1.0053


def h17(point: int) -> float:
    """Rend l'enthalpie du point numerote de la question 1 de mai 2017."""
    return H_2017[point - 1]


def test_mai_2017_debits_de_la_pompe_a_chaleur() -> None:
    """Mai 2017, question 1 : qM2 depuis le flux froid, qM1 = 20 % du total.

    Enonce Exam_2017_Mai_Exercice_cote_Enonce : 10 kW preleves a la source froide,
    20 % du debit du condenseur detendus dans V1. Le corrige arrondit qM2 a
    0.06, d'ou la tolerance de 0,2 %.
    """
    assert 10 / (h17(1) - h17(10)) == pytest.approx(QM2_2017, rel=2e-3)
    assert pytest.approx(0.2) == QM1_2017 / (QM1_2017 + QM2_2017)


def test_mai_2017_bilans_des_echangeurs_et_du_melangeur() -> None:
    """Mai 2017, question 1 : E1, E2 et le ballon M sont adiabatiques."""
    total = QM1_2017 + QM2_2017
    assert QM2_2017 * (h17(6) - h17(8)) == pytest.approx(
        QM1_2017 * (h17(11) - h17(7)), rel=ARRONDI
    )
    assert h17(8) - h17(9) == pytest.approx(h17(2) - h17(1))
    melange = (QM2_2017 * h17(3) + QM1_2017 * h17(11)) / total
    assert melange == pytest.approx(h17(4), rel=ARRONDI)


def test_mai_2017_flux_puissances_et_cop() -> None:
    """Mai 2017, question 1 : flux chaud, PC1, PC2, COP et bilan global."""
    total = QM1_2017 + QM2_2017
    assert total * (h17(6) - h17(5)) * 1000 == pytest.approx(-13725)
    assert QM2_2017 * (h17(3) - h17(2)) * 1000 == pytest.approx(1620)
    assert total * (h17(5) - h17(4)) * 1000 == pytest.approx(2115)
    assert pytest.approx(3.67, rel=2e-3) == 13725 / (1620 + 2115)
    froid = QM2_2017 * (h17(1) - h17(10)) * 1000
    assert froid + 1620 + 2115 == pytest.approx(13725)


def test_mai_2017_compresseur_du_cycle_de_joule() -> None:
    """Mai 2017, question 2 : T2isos puis T2, rendement isentropique de 90 %.

    Enonce Exam_2017_Mai_Exercice_cote_Enonce : k = 1.4, taux de compression 15.
    """
    t1 = 25 + ZERO_CELSIUS
    t2_isos = t1 * 15**EXPOSANT
    assert t2_isos - ZERO_CELSIUS == pytest.approx(373.19, abs=0.01)
    assert pytest.approx(411.88, abs=0.01) == 25 + (373.19 - 25) / 0.9


def test_mai_2017_debit_d_air_et_flux_du_cycle_de_joule() -> None:
    """Mai 2017, question 2 : qMAIR, PT1, flux recupere et flux chaud.

    Enonce Exam_2017_Mai_Exercice_cote_Enonce : cP = 1005.3 J/(kg.K) et puissance
    nette du cycle de Joule de 265 MW.
    """
    net_massique = CP_AIR_2017 * ((1400 - 620) - (411.88 - 25))
    debit = 265_000 / net_massique
    assert debit == pytest.approx(670.54, rel=1e-4)
    capacite = 670.54 * CP_AIR_2017 / 1000
    assert capacite * (620 - 1400) == pytest.approx(-525.79, rel=1e-4)
    assert capacite * (320 - 620) == pytest.approx(-202.22, rel=1e-4)
    assert capacite * (1400 - 411.88) == pytest.approx(666.09, rel=1e-4)


def test_mai_2017_cycle_de_hirn() -> None:
    """Mai 2017, question 2 : qMEAU, PT2, pompe et rendement de la turbine T2.

    Les enthalpies sont en kcal/kg et les puissances en MW : le facteur de
    conversion que le corrige emploie, deduit de qMEAU, est seulement encadre.
    La puissance de la pompe n'a que deux chiffres, d'ou sa tolerance de 2 %.
    Enonce : rendement isentropique de T2 de 90 %.
    """
    recu = 845 - 31.2
    assert KCAL_MIN < 202.22e3 / (59.36 * recu) < KCAL_MAX
    assert 202.22 * (544.4 - 845) / recu == pytest.approx(-74.69, rel=ARRONDI)
    assert 202.22 * (31.2 - 30) / recu == pytest.approx(0.30, rel=2e-2)
    assert pytest.approx(0.9, rel=ARRONDI) == (845 - 544.4) / (845 - 511)


def test_mai_2017_rendement_du_cycle_combine() -> None:
    """Mai 2017, question 2 : (265 + |PT2| - Ppompe) / flux chaud = 0.51.

    Les 265 MW nets du cycle de Joule viennent de l'enonce ; le corrige ne
    donne que deux chiffres, d'ou la tolerance de 1 %.
    """
    assert pytest.approx(0.51, rel=1e-2) == (265 + 74.69 - 0.30) / 666.09


# --- Mai 2019 : Exam_2019_Mai_Exercice_cote_Solution ---------------------

T_2019 = {
    "1": 20.0,
    "2isos": 103.54,
    "2": 124.43,
    "3": 621.60,
    "4": 1000.0,
    "5isos": 717.66,
    "5": 745.89,
    "6": 248.72,
    "7": 20.0,
    "8": -45.0,
    "9": -35.0,
}
QM1, QM2, QM3 = 7.5427, 1.5727, 5.97
H_ETHANE = {
    "1": 161.2,
    "2isos": 181.2,
    "2": 189.2,
    "3": 165.0,
    "4isos": 178.3,
    "4": 183.62,
    "5": 77.9,
    "6": 77.9,
    "7": 59.6,
    "8": 59.6,
}
REF1, REF2 = 0.0911, 0.1356


def kelvin(point: str) -> float:
    """Rend la temperature absolue d'un point de la question 1 de mai 2019."""
    return T_2019[point] + ZERO_CELSIUS


def test_mai_2019_colonne_des_kelvins() -> None:
    """Mai 2019, question 1, tableau : T(K) = T(degres C) + 273,15.

    Confirme la lecture 521,87 du point 6, longtemps marquee douteuse.
    """
    haut = (293.15, 376.69, 397.58, 894.75, 1273.15, 990.81)
    bas = (1019.04, 521.87, 293.15, 228.15, 238.15)
    lus = haut + bas
    for point, lu in zip(T_2019, lus, strict=True):
        assert kelvin(point) == pytest.approx(lu, abs=0.005)


def test_mai_2019_pression_de_la_detente_isentropique() -> None:
    """Mai 2019, question 1, 2) : p7 = (T7 / T8)^(k / (k - 1)) = 2,405 bar."""
    assert pytest.approx(2.405, rel=ARRONDI) == (293.15 / 228.15) ** (1 / EXPOSANT)


def test_mai_2019_compresseur_et_turbine() -> None:
    """Mai 2019, question 1, 2) : points 2isos, 2, 5isos et 5."""
    t2_isos = 293.15 * 2.405**EXPOSANT
    assert t2_isos == pytest.approx(376.69, abs=0.01)
    assert pytest.approx(397.58, abs=0.01) == 293.15 + (376.69 - 293.15) / 0.8
    t5_isos = 1273.15 * (1 / 2.405) ** EXPOSANT
    assert t5_isos == pytest.approx(990.81, abs=0.01)
    assert pytest.approx(1019.04, abs=0.01) == 1273.15 - 0.9 * (1273.15 - 990.81)


def test_mai_2019_echangeur_d_efficacite_0_8() -> None:
    """Mai 2019, question 1, 2) : points 6 et 3 par l'efficacite de l'echangeur."""
    ecart = T_2019["5"] - T_2019["2"]
    assert T_2019["5"] - 0.8 * ecart == pytest.approx(T_2019["6"], abs=0.01)
    assert T_2019["2"] + 0.8 * ecart == pytest.approx(T_2019["3"], abs=0.01)


def test_mai_2019_debits_d_air() -> None:
    """Mai 2019, question 1, 3) : qM3, qM2 par la puissance nette nulle, qM1.

    cP = 1005 J/(kg.K) vient de l'enonce Exam_2019_Mai_Exercice_cote_Enonce. Le
    troisieme terme du bilan porte bien T8 - T7 : c'est la seule lecture de
    l'indice mal forme qui redonne 1,5727.
    """
    assert 60 / (CP_AIR * (T_2019["9"] - T_2019["8"])) == pytest.approx(
        QM3, rel=ARRONDI
    )
    compression = T_2019["2"] - T_2019["1"]
    turbine_1 = T_2019["5"] - T_2019["4"]
    turbine_2 = T_2019["8"] - T_2019["7"]
    debit = -QM3 * (compression + turbine_2) / (compression + turbine_1)
    assert debit == pytest.approx(QM2, rel=1e-4)
    assert pytest.approx(QM1) == QM2 + QM3


def test_mai_2019_puissances_et_flux() -> None:
    """Mai 2019, question 1, 4) et 5) : trois puissances, deux flux, bilan nul."""
    ecart = {
        "compresseur": T_2019["2"] - T_2019["1"],
        "turbine_1": T_2019["5"] - T_2019["4"],
        "turbine_2": T_2019["8"] - T_2019["7"],
        "chaud": T_2019["4"] - T_2019["3"],
        "moyen": T_2019["7"] - T_2019["2"],
    }
    assert QM1 * CP_AIR * ecart["compresseur"] == pytest.approx(791.62, rel=1e-4)
    assert QM2 * CP_AIR * ecart["turbine_1"] == pytest.approx(-401.64, rel=1e-4)
    assert QM3 * CP_AIR * ecart["turbine_2"] == pytest.approx(-389.99, rel=1e-4)
    assert QM2 * CP_AIR * ecart["chaud"] == pytest.approx(598.09, rel=1e-4)
    assert QM3 * CP_AIR * ecart["moyen"] == pytest.approx(-626.56, rel=1e-4)
    assert pytest.approx(0.0, abs=0.02) == 791.62 - 401.64 - 389.99


def test_mai_2019_pressions_et_temperatures_de_l_ethane() -> None:
    """Mai 2019, question 2, 1) et 2) : T_EVAP = -80 - 10 et T1' = -90 + 10."""
    assert pytest.approx(-90) == -80 - 10
    assert pytest.approx(-80) == -90 + 10


def test_mai_2019_rendement_commun_des_deux_compresseurs() -> None:
    """Mai 2019, question 2, 2) : rendement 0,714 puis h4' = 183,62 kcal/kg."""
    h = H_ETHANE
    rendement = (h["2isos"] - h["1"]) / (h["2"] - h["1"])
    assert rendement == pytest.approx(0.714, rel=ARRONDI)
    h4 = h["3"] + (h["4isos"] - h["3"]) / 0.714
    assert h4 == pytest.approx(h["4"], rel=1e-4)


def test_mai_2019_debits_d_ethane() -> None:
    """Mai 2019, question 2, 3) : qMREF2 par le condenseur, qMREF1 par le ballon.

    Le facteur kcal -> kJ que suppose 0,1356 kg/s est encadre, non impose. Le
    bilan du separateur, lui, n'en depend pas : il confirme la lecture 0,0911.
    """
    h = H_ETHANE
    assert KCAL_MIN < 60 / (REF2 * (h["4"] - h["5"])) < KCAL_MAX
    debit = REF2 * (h["3"] - h["6"]) / (h["2"] - h["7"])
    assert debit == pytest.approx(REF1, rel=ARRONDI)


def test_mai_2019_flux_froid_puissances_et_cop() -> None:
    """Mai 2019, question 2, 4) : flux froid, PC1, PC2, COP et bilan a 60 kW.

    Chaque puissance est rapportee aux 60 kW du condenseur, ce qui elimine le
    facteur de conversion des kilocalories.
    """
    h = H_ETHANE
    par_kcal = 60 / (REF2 * (h["4"] - h["5"]))
    assert REF1 * (h["1"] - h["8"]) * par_kcal == pytest.approx(38.74, rel=ARRONDI)
    assert REF1 * (h["2"] - h["1"]) * par_kcal == pytest.approx(10.68, rel=ARRONDI)
    assert REF2 * (h["4"] - h["3"]) * par_kcal == pytest.approx(10.57, rel=ARRONDI)
    assert pytest.approx(1.82, rel=2e-3) == 38.74 / (10.68 + 10.57)
    assert pytest.approx(60, rel=ARRONDI) == 38.74 + 10.68 + 10.57


def test_mai_2019_bareme_de_la_question_1() -> None:
    """Mai 2019, question 1 : notes partielles, sous-totaux et total entoure 35."""
    assert pytest.approx(3) == 0.5 + 0.5 + 2
    assert pytest.approx(13) == 2 + 2 + 0.5 + 2 + 2 + 2 + 2 + 0.5
    assert pytest.approx(5) == 2 + 2 + 1
    assert pytest.approx(35) == 4 + 3 + 13 + 5 + (2 + 2 + 2) + (2 + 2)


def test_mai_2019_bareme_de_la_question_2() -> None:
    """Mai 2019, question 2 : notes partielles, sous-totaux et total entoure 28.

    Le premier sous-total confirme la note surchargee de p_EVAP : 1 point.
    """
    assert pytest.approx(4) == 1 + 1 + 1 + 1
    par_point = (1.5, 0.5, 1, 1, 0.5, 2.5, 1, 1.5, 1, 1.5)
    assert sum(par_point) == pytest.approx(12)
    assert pytest.approx(28) == 4 + 12 + (2 + 2) + (2 + 2 + 2 + 2)
