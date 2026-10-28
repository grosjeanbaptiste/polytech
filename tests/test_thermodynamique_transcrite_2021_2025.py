"""Recalcule les corriges transcrits de thermodynamique, de 2021 a 2025.

Cinq transcriptions de `transcriptions/Thermodynamique/EXAMENS/` sont
couvertes : novembre 2021, 2022, 2023 et 2024, puis mai 2025. Une
transcription est une lecture, et une lecture se trompe en silence : chaque
resultat est donc refait ici depuis les operandes que le corrige ecrit.

Trois corriges (novembre 2022, novembre 2023, mai 2025) ne donnent que des
reponses finales. Leurs donnees viennent alors de l'enonce homonyme du depot,
`cours-markdown/Thermodynamique/EXAMENS/`, nomme dans chaque docstring.

Aucune constante ne vient d'ailleurs que des documents : `R`, le joule par
calorie, le pascal par atmosphere et le zero Celsius sont ceux que les
corriges de novembre 2021 et de novembre 2024 ecrivent en toutes lettres.

Le fichier depasse la centaine de lignes de la regle du depot : il porte un
lot de cinq corriges sous un nom impose, et chaque fonction reste courte.
"""

import math

import pytest

from polytech.chemistry.domain.transformations_de_gaz import (
    capacite_isochore,
    chaleur_isochore,
    rapport_des_pressions_a_volume_constant,
    temperature_apres_detente_adiabatique,
)

R = 8.314
JOULES_PAR_CALORIE = 4.186
PASCALS_PAR_ATMOSPHERE = 101325
ZERO_CELSIUS = 273.15


def energie_interne(enthalpie: float, pression: float, volume: float) -> float:
    """Rend u = h - p v, les trois grandeurs etant deja en unites SI."""
    return enthalpie - pression * volume


def temperature_de_liquidus(latente: float, fusion: float, titre: float) -> float:
    """Inverse ln x = -(L/R) (1/T - 1/T_f) pour rendre T en kelvins."""
    return 1 / (1 / fusion - R / latente * math.log(titre))


def titre_de_liquidus(latente: float, fusion: float, temperature: float) -> float:
    """Rend x = exp(-(L/R) (1/T - 1/T_f))."""
    return math.exp(-latente / R * (1 / temperature - 1 / fusion))


# --- Novembre 2021 -------------------------------------------------------

ETATS_2021 = (
    (122, 39, 1.10, 506.3, 510.7),
    (147, 39, 5.0, 595.6, 615.3),
    (191, 100, 5.0, 748.9, 799.5),
    (170, 12, 30, 675.1, 711.6),
)


@pytest.mark.parametrize(("h", "p", "v", "u", "h_kj"), ETATS_2021)
def test_2021_question_1_energie_interne_de_chaque_etat(
    h: float, p: float, v: float, u: float, h_kj: float
) -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, question 1 : u = h - p v.

    Le manuscrit multiplie par 4186 et 101325, donc calcule des J/kg, et
    ecrit le resultat en kJ/kg : la division par mille est faite ici.
    """
    enthalpie = h * 1000 * JOULES_PAR_CALORIE
    assert enthalpie / 1000 == pytest.approx(h_kj, rel=1e-3)
    interne = energie_interne(enthalpie, p * PASCALS_PAR_ATMOSPHERE, v * 1e-3)
    assert interne / 1000 == pytest.approx(u, rel=1e-3)


def test_2021_question_1_transformation_isobare() -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, question 1 : W12, dU12, Q12."""
    travail = -39 * PASCALS_PAR_ATMOSPHERE * 2 * (5.0 - 1.10) * 1e-3 / 1000
    assert travail == pytest.approx(-30.82, rel=1e-3)
    variation = 2 * (595.6 - 506.3)
    assert variation == pytest.approx(178.6, rel=1e-3)
    assert variation - travail == pytest.approx(209.42, rel=1e-3)


def test_2021_question_1_isochore_puis_isentropique() -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, question 1 : dU23 et dU34."""
    assert pytest.approx(306.6, rel=1e-3) == 2 * (748.9 - 595.6)
    assert pytest.approx(-147.6, rel=1e-3) == 2 * (675.1 - 748.9)


def test_2021_question_2_temperature_finale() -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, question 2 : bilan d'enthalpie."""
    assert 5 * 4180 == 20900
    assert 5 * 334400 == 1672000
    assert 8 * 449 == 3592
    assert 8 * 449 * 600 == 2155200
    pente = 20900 + 20900 + 3592
    assert pente == 45392
    second_membre = 250000 - 1672000 + 2155200
    assert second_membre == 733200
    celsius = second_membre / pente
    assert celsius == pytest.approx(16.15, rel=1e-3)
    assert celsius + ZERO_CELSIUS == pytest.approx(289.30, rel=1e-4)


def test_2021_question_2_bilan_d_entropie() -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, question 2 : dS et dS_i."""
    eau = 5 * 4180 * math.log(289.3 / 273.15)
    assert eau == pytest.approx(1200.56, rel=1e-4)
    fusion = 5 * 334400 / 273.15
    assert fusion == pytest.approx(6121.2, rel=1e-4)
    fer = 8 * 449 * math.log(289.3 / 873.15)
    assert fer == pytest.approx(-3967.9, rel=1e-4)
    assert pytest.approx(873.15) == 600 + ZERO_CELSIUS
    systeme = eau + fusion + eau + fer
    assert systeme == pytest.approx(4554.42, rel=1e-4)
    assert pytest.approx(3690.3, rel=1e-4) == 4554.42 - 250000 / 289.3


def test_2021_question_3_titres_a_moins_dix_huit_degres() -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, question 3 a) : x1 puis tau.

    Le manuscrit arrondit les titres a trois decimales (0,830 ; 0,914 ;
    0,086) : la tolerance est donc absolue, d'une demi-unite du dernier
    chiffre ecrit.
    """
    assert pytest.approx(255.15) == -18 + ZERO_CELSIUS
    cote_eau = titre_de_liquidus(6002, 273.15, 255.15)
    assert cote_eau == pytest.approx(0.830, abs=5e-4)
    massique = 0.83 * 18.015 / (0.83 * 18.015 + (1 - 0.83) * 62.068)
    assert massique == pytest.approx(0.586, abs=5e-4)
    assert pytest.approx(0.414) == 1 - 0.586
    cote_glycol = titre_de_liquidus(9958, 260.15, 255.15)
    assert cote_glycol == pytest.approx(0.914, abs=5e-4)
    assert pytest.approx(0.086) == 1 - 0.914
    massique = 0.086 * 18.015 / (0.086 * 18.015 + (1 - 0.086) * 62.068)
    assert massique == pytest.approx(0.0266, abs=5e-5)
    assert pytest.approx(0.9734) == 1 - 0.0266


def test_2021_question_3_eutectique_par_la_branche_du_glycol() -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, page 5 : T = 222,15 K."""
    temperature = temperature_de_liquidus(9958, 260.15, 1 - 0.545)
    assert temperature == pytest.approx(222.15, abs=0.01)


def test_2021_question_3_la_branche_de_l_eau_garde_son_marqueur() -> None:
    """Exam_2021_Novembre_Exercice_cote_Solution, page 5 : « [? 222,37] » K.

    Le recalcul donne 222,13 K, a deux centiemes de l'autre branche -- ce
    que le « OK » du manuscrit constate. Relue a 600 dpi, la valeur ecrite
    porte bien un 3 en premiere decimale ; la seconde est un 7 ou un 1.
    Aucune des deux lectures ne retombe sur le recalcul : le doute est
    dans le manuscrit, et le marqueur reste.
    """
    temperature = temperature_de_liquidus(6002, 273.15, 0.545)
    assert temperature == pytest.approx(222.13, abs=0.01)
    assert abs(temperature - 222.15) < 0.03
    assert min(abs(222.37 - temperature), abs(222.31 - temperature)) > 0.15


# --- Novembre 2022 -------------------------------------------------------

MASSE_2022 = 3
CV_2022 = 742
R_MASSIQUE_2022 = 297


def test_2022_question_1_etats_du_cycle() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 1 a).

    Donnees de l'enonce `Exam_2022_Novembre_Exercice_cote_Enonce.md` : 3 kg d'azote a
    2 bar, volume divise par 8 a temperature constante, 618,45 degres C a
    l'etat 3, r = 297 J/(kg K) et k = 1,4.
    """
    volume = MASSE_2022 * R_MASSIQUE_2022 * 300 / 2e5
    assert volume == pytest.approx(1.3365, rel=1e-4)
    assert volume / 8 == pytest.approx(0.16706, rel=1e-4)
    assert 2 * 8 == 16
    chaud = 618.45 + ZERO_CELSIUS
    assert chaud == pytest.approx(891.6)
    assert MASSE_2022 * R_MASSIQUE_2022 * chaud / 16e5 == pytest.approx(
        0.49651, rel=1e-4
    )
    detendu = chaud * (0.49651 / 1.3365) ** 0.4
    assert detendu == pytest.approx(600, rel=1e-4)
    assert pytest.approx(4, rel=1e-4) == 16 * (0.49651 / 1.3365) ** 1.4


def test_2022_question_1_travaux_et_chaleurs() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 1 b).

    L'enonce donne la chaleur cedee a la compression (555835 J) et
    cV = 742 J/(kg K) ; le travail isotherme est refait par m r T ln 8.
    """
    isotherme = MASSE_2022 * R_MASSIQUE_2022 * 300 * math.log(8)
    assert isotherme == pytest.approx(555835, rel=1e-5)
    isobare = -16e5 * (0.49651 - 0.16706)
    assert isobare == pytest.approx(-527120, rel=1e-5)
    interne = MASSE_2022 * CV_2022 * (891.6 - 300)
    assert interne - isobare == pytest.approx(1844022, rel=1e-5)
    assert pytest.approx(-649102, rel=1e-5) == MASSE_2022 * CV_2022 * (600 - 891.6)
    assert pytest.approx(-667800, rel=1e-5) == MASSE_2022 * CV_2022 * (300 - 600)


def test_2022_question_1_le_bilan_du_cycle_est_nul() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 1 b) : W + Q = 0."""
    travaux = 555835 - 527120 - 649102 + 0
    chaleurs = -555835 + 1844022 + 0 - 667800
    assert travaux + chaleurs == 0


def test_2022_question_2_le_metal_est_le_cuivre() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 2.

    Enonce : 10 kg de metal de 550 a 100 degres C, 1 kg d'eau chauffee de
    20 a 100 degres C dont 600 g s'evaporent (2281 J/g), 29500 J perdus,
    cp eau = 4,18 J/(g K) et cp cuivre = 0,385 J/(g K).
    """
    recu_par_l_eau = 1000 * 4.18 * (100 - 20) + 600 * 2281
    chaleur_massique = (recu_par_l_eau + 29500) / (10000 * (550 - 100))
    assert chaleur_massique == pytest.approx(0.385, rel=1e-3)


def pression_d_antoine(a: float, b: float, c: float, temperature: float) -> float:
    """Loi de l'enonce de novembre 2022 : log10 p (bar) = A - B / (T + C)."""
    return float(10 ** (a - b / (temperature + c)))


DECANE = (4.24, 1611.68, -66.87)
BENZENE = (4.17, 1295.35, -42.43)


def test_2022_question_3_temperatures_de_condensation() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 3 b) : Antoine a 3 bar."""
    decane = 1611.68 / (4.24 - math.log10(3)) + 66.87
    assert decane == pytest.approx(495.18, abs=0.01)
    benzene = 1295.35 / (4.17 - math.log10(3)) + 42.43
    assert benzene == pytest.approx(393.2, abs=0.05)


def test_2022_question_3_fin_de_condensation_au_point_a() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 3 c) : T_A = 413,2 K.

    A la fin de la condensation le liquide a le titre global, 0,4 en
    decane : la somme de Raoult doit rendre les 3 bar a 413,2 K, et non a
    461 K, l'autre valeur que l'enonce propose.
    """
    a_413 = 0.4 * pression_d_antoine(*DECANE, 413.2) + 0.6 * pression_d_antoine(
        *BENZENE, 413.2
    )
    assert a_413 == pytest.approx(3, rel=1e-3)
    a_461 = 0.4 * pression_d_antoine(*DECANE, 461) + 0.6 * pression_d_antoine(
        *BENZENE, 461
    )
    assert a_461 > 2 * 3


def test_2022_question_3_titres_a_430_kelvins() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 3 d) : x et y."""
    decane = pression_d_antoine(*DECANE, 430)
    benzene = pression_d_antoine(*BENZENE, 430)
    liquide = (3 - benzene) / (decane - benzene)
    assert liquide == pytest.approx(0.6116, rel=1e-3)
    assert liquide * decane / 3 == pytest.approx(0.1291, rel=1e-3)


def test_2022_question_3_debut_de_solidification_au_point_b() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, question 3 e) : T = 255,49 K.

    Enonce : benzene, T_fusion = 278,6 K et L = 13080 J/mol ; decane,
    243,5 K et 28700 J/mol. A 0,4 en decane, la branche du benzene
    (titre 0,6) est atteinte bien avant celle du decane.
    """
    benzene = temperature_de_liquidus(13080, 278.6, 0.6)
    assert benzene == pytest.approx(255.49, abs=0.01)
    assert temperature_de_liquidus(28700, 243.5, 0.4) < benzene


def test_2022_question_3_l_eutectique_se_lit_0_6388() -> None:
    """Exam_2022_Novembre_Exercice_cote_Reponses, figure 3 a) : x_EUT = 0,6388.

    Marqueur « [? 0,6398] » leve a 800 dpi : le troisieme chiffre est un 8
    ferme, pas le 9 a queue ouverte de la meme page. Deux recoupements le
    confirment : l'enonce donne 0,6388, et les deux branches du liquidus
    ne se rejoignent qu'a cette abscisse -- a 0,6398 elles s'ecartent de
    plus d'un dixieme de kelvin.
    """
    ecart_lu = temperature_de_liquidus(13080, 278.6, 1 - 0.6388)
    ecart_lu -= temperature_de_liquidus(28700, 243.5, 0.6388)
    assert abs(ecart_lu) < 0.01
    ecart_ancien = temperature_de_liquidus(13080, 278.6, 1 - 0.6398)
    ecart_ancien -= temperature_de_liquidus(28700, 243.5, 0.6398)
    assert abs(ecart_ancien) > 0.1


# --- Novembre 2023 -------------------------------------------------------

MASSE_2023 = 2
CV_2023 = 718
R_MASSIQUE_2023 = 287.2
FROID_2023 = 20 + ZERO_CELSIUS


def test_2023_question_1_etats_1_a_3() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, question 1 a).

    Enonce `Exam_2023_Novembre_Exercice_cote_Enonce.md` : 2 kg d'air a 1 bar et 20
    degres C, compression adiabatique reversible jusqu'a 500 degres C,
    isobare jusqu'a 1000 degres C, r = 287,2 J/(kg K) et k = 1,4.
    """
    produit = MASSE_2023 * R_MASSIQUE_2023
    assert produit * FROID_2023 / 1e5 == pytest.approx(1.6839, rel=1e-4)
    pression = ((500 + ZERO_CELSIUS) / FROID_2023) ** (1.4 / 0.4)
    assert pression == pytest.approx(29.792, rel=1e-4)
    assert produit * (500 + ZERO_CELSIUS) / 29.792e5 == pytest.approx(0.1491, rel=1e-3)
    assert produit * (1000 + ZERO_CELSIUS) / 29.792e5 == pytest.approx(0.2454, rel=1e-3)


def test_2023_question_1_travaux_et_chaleurs() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, question 1 b).

    L'enonce impose le travail de la detente (800 kJ) et cV = 718 J/(kg K).
    La chaleur de l'isochore est verifiee a part, avec l'etat 4.
    """
    assert pytest.approx(689280) == MASSE_2023 * CV_2023 * (500 - 20)
    assert pytest.approx(-287200) == -MASSE_2023 * R_MASSIQUE_2023 * (1000 - 500)
    assert pytest.approx(-287200, rel=2e-3) == -29.792e5 * (0.2454 - 0.1491)
    chaleur = MASSE_2023 * (CV_2023 + R_MASSIQUE_2023) * (1000 - 500)
    assert chaleur == pytest.approx(1005200)


def test_2023_question_1_le_bilan_du_cycle_est_nul() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, question 1 b) : W + Q = 0."""
    assert 689280 - 287200 - 800000 + 1005200 - 607280 == 0


def test_2023_le_manuscrit_ecrit_443_90_pour_une_temperature_de_442_90() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, etat 4 : ecart du manuscrit.

    Relu a 800 dpi, le scan porte bien « 443,90 » : la transcription est
    fidele. Mais le premier principe sur la detente adiabatique (W = -800
    kJ, donnee de l'enonce) rend 442,90 degres C, et c'est cette valeur,
    non 443,90, que les deux autres reponses du manuscrit supposent :
    p4 = 2,443 bar et Q41 = -607280 J. Un chiffre des unites a glisse a la
    recopie de la reponse.
    """
    assert capacite_isochore(2, 718) == 1436
    exacte = temperature_apres_detente_adiabatique(1000.0, 2, 718, 800000.0)
    assert exacte == pytest.approx(442.90, abs=0.005)
    arrondie = 442.90
    assert chaleur_isochore(2, 718, exacte, 20.0) == pytest.approx(-607280.0, abs=0.05)
    assert chaleur_isochore(2, 718, arrondie, 20.0) == pytest.approx(
        -607284.4, abs=0.05
    )
    assert rapport_des_pressions_a_volume_constant(20.0, exacte) == pytest.approx(
        2.4426, abs=5e-5
    )
    assert rapport_des_pressions_a_volume_constant(20.0, arrondie) == pytest.approx(
        2.4426, abs=5e-5
    )
    assert rapport_des_pressions_a_volume_constant(20.0, 443.90) == pytest.approx(
        2.4460, abs=5e-5
    )
    assert chaleur_isochore(2, 718, 443.90, 20.0) == pytest.approx(-608720.4, abs=0.05)


def test_2023_question_2_masse_de_methanol_evapore() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, question 2 : 38,308 g.

    Enonce : 200 g de cuivre a 0,09 cal/(g K), 800 g de methanol a 0,6
    cal/(g K), refroidis de 25 a 5 degres C ; L = 260 cal/g.
    """
    cedee = (200 * 0.09 + 800 * 0.6) * (25 - 5)
    assert cedee / 260 == pytest.approx(38.308, rel=1e-4)


FUSION_1_2023 = (6750 * JOULES_PAR_CALORIE, 875 + ZERO_CELSIUS)
FUSION_2_2023 = (6400 * JOULES_PAR_CALORIE, 848 + ZERO_CELSIUS)
EUTECTIQUE_2023 = 643.37 + ZERO_CELSIUS


def test_2023_question_3_la_temperature_d_eutectique_ferme_la_somme() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, question 3 a) : somme = 1.

    Enonce : T_fusion de 875 et 848 degres C, L de 6750 et 6400 cal/mol.
    Les calories sont converties par le 4,186 des corriges voisins.
    """
    somme = titre_de_liquidus(*FUSION_1_2023, EUTECTIQUE_2023)
    somme += titre_de_liquidus(*FUSION_2_2023, EUTECTIQUE_2023)
    assert somme == pytest.approx(1, rel=1e-3)


def test_2023_question_3_le_solide_1_se_forme_a_751_48_degres() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, question 3 b).

    Marqueur « [? 751,48] » leve a 800 dpi : le troisieme chiffre a la
    forme du 1 de « 1000 », sans la barre du 7 initial. Le recalcul au
    titre 0,7 confirme 751,5 et ecarte 757,48 ; la branche du compose 2
    (titre 0,3) est bien plus basse, c'est donc le solide 1 qui se forme.
    """
    branche_1 = temperature_de_liquidus(*FUSION_1_2023, 0.7) - ZERO_CELSIUS
    assert branche_1 == pytest.approx(751.48, abs=0.1)
    branche_2 = temperature_de_liquidus(*FUSION_2_2023, 0.3) - ZERO_CELSIUS
    assert branche_2 < branche_1


def test_2023_question_3_moles_de_compose_2_a_ajouter() -> None:
    """Exam_2023_Novembre_Exercice_cote_Reponses, question 3 c) : 2,393 moles.

    Il faut amener le titre en 1 a celui de l'eutectique : 3,5 / (5 + n).
    La tolerance est de 2 pour mille : la reponse est une petite
    difference de deux grands termes, et le manuscrit ne dit pas quelle
    valeur de R en calories il a prise.
    """
    titre = titre_de_liquidus(*FUSION_1_2023, EUTECTIQUE_2023)
    assert 0.7 * 5 / titre - 5 == pytest.approx(2.393, rel=2e-3)


# --- Novembre 2024 -------------------------------------------------------

MOLES_2024 = 117.426
ETATS_2024 = (
    (10000, 5, 6000, 41860, 38820.25),
    (6400, 5, 2000, 26790.4, 25777.15),
    (6200, 2, 5000, 25953.2, 24939.95),
    (6875, 2, 6000, 28778.75, 27562.85),
)


def test_2024_question_1_moles_et_volumes() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, question 1 : n, v2, V1, v1."""
    moles = 2000 / 17.032
    assert moles == pytest.approx(117.426, rel=1e-5)
    assert 234.85 / moles == pytest.approx(2, rel=1e-4)
    volume = 0.23485 + 237960 / (5 * PASCALS_PAR_ATMOSPHERE)
    assert volume == pytest.approx(0.704546, rel=1e-5)
    assert volume / moles * 1e6 == pytest.approx(6000, rel=1e-4)


@pytest.mark.parametrize(("h_cal", "p", "v", "h", "u"), ETATS_2024)
def test_2024_question_1_energie_interne_de_chaque_etat(
    h_cal: float, p: float, v: float, h: float, u: float
) -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, question 1 : u = h - p v."""
    enthalpie = h_cal * JOULES_PAR_CALORIE
    assert enthalpie == pytest.approx(h, rel=1e-6)
    interne = energie_interne(enthalpie, p * PASCALS_PAR_ATMOSPHERE, v * 1e-6)
    assert interne == pytest.approx(u, rel=1e-6)


def test_2024_question_1_isobare_puis_isentropique() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, transformations 1-2 et 2-3."""
    variation = MOLES_2024 * (25777.15 - 38820.25)
    assert variation == pytest.approx(-1531599, rel=1e-6)
    assert -1531599 - 237960 == -1769559
    assert pytest.approx(-98309, rel=1e-5) == MOLES_2024 * (24939.95 - 25777.15)


def test_2024_question_1_evaporation_puis_isochore() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, transformations 3-4 et 4-1.

    « [? 307996,6] » garde son marqueur : le dernier chiffre est une
    surcharge que le scan ne departage pas. Le recalcul rend 307996,66 et
    la ligne suivante, Q34 = 331793 J, suppose bien 307996,6 -- ce qui
    confirme la lecture sans la remplacer.
    """
    travail = -2 * PASCALS_PAR_ATMOSPHERE * (0.006 - 0.005) * MOLES_2024
    assert travail == pytest.approx(-23796.4, rel=1e-5)
    variation = MOLES_2024 * (27562.85 - 24939.95)
    assert variation == pytest.approx(307996.6, abs=0.1)
    assert pytest.approx(331793, abs=0.01) == 307996.6 + 23796.4
    assert pytest.approx(1321911.5, abs=0.1) == MOLES_2024 * (38820.25 - 27562.85)


def test_2024_question_1_le_bilan_du_cycle_est_nul() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, question 1 : W + Q = 0.

    Le residu, 0,1 J sur plus d'un megajoule, est l'arrondi des reponses.
    """
    travaux = 237960 - 98309 - 23796.4 + 0
    chaleurs = -1769559 + 0 + 331793 + 1321911.5
    assert travaux + chaleurs == pytest.approx(0, abs=0.5)


def test_2024_les_baremes_retombent_sur_les_totaux_ecrits() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution : /35, /25, /60 et /36.

    Les points entoures de la question 1 font bien 35, mais ils sont 24 :
    le manuscrit annonce « 25 cotes ». Les « 8 cotes » des transformations
    et les « 10 cotes » de la question 2 sont exacts.
    """
    etats_1_et_2 = (1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2)
    etat_3 = (1, 2, 1, 1, 1, 2)
    etat_4 = (2, 2, 2, 1, 1, 2)
    etats = etats_1_et_2 + etat_3 + etat_4
    assert sum(etats) == 35
    assert len(etats) == 24
    transformations = (1, 4, 2, 4, 4, 4, 2, 4)
    assert sum(transformations) == 25
    assert len(transformations) == 8
    assert sum(etats) + sum(transformations) == 60
    question_2 = (2, 4, 4, 4, 4, 4, 4, 4, 3, 3)
    assert sum(question_2) == 36
    assert len(question_2) == 10


def test_2024_question_2_coefficients_des_pressions_de_saturation() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, question 2 b)."""
    assert pytest.approx(3401.49, rel=1e-5) == 28280 / R
    assert pytest.approx(10.10454, rel=1e-5) == 28280 / R / 336.63
    assert pytest.approx(3614.38, rel=1e-5) == 30050 / R
    assert pytest.approx(10.21359, rel=1e-5) == 30050 / R / 353.88


def test_2024_question_2_pression_de_rosee() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, question 2 c)."""
    temperature = 71.5 + ZERO_CELSIUS
    assert math.exp(-3401.49 / temperature + 10.10454) == pytest.approx(
        1.26508, rel=1e-4
    )
    assert math.exp(-3614.38 / temperature + 10.21359) == pytest.approx(
        0.76070, rel=1e-4
    )
    assert pytest.approx(1, rel=1e-3) == 1 / (0.6 / 1.26508 + 0.4 / 0.7607)


def test_2024_question_2_titre_de_bulle_et_ajout() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, questions 2 a) et 2 d)."""
    assert pytest.approx(0.6) == 600 / (600 + 400)
    assert pytest.approx(0.47428, rel=1e-5) == 1 * 0.6 / 1.26508
    assert pytest.approx(265.1, rel=1e-3) == 600 / 0.47428 - 600 - 400


def test_2024_question_2_solidification_et_eutectique() -> None:
    """Exam_2024_Novembre_Exercice_cote_Solution, questions 2 e), f) et h)."""
    assert pytest.approx(0.741) == 1 - 0.259
    assert pytest.approx(0.52572) == 1 - 0.47428
    debut = temperature_de_liquidus(2677.3, 279.87, 0.52572)
    assert debut == pytest.approx(179.54, abs=0.01)
    assert pytest.approx(-93.61) == 179.54 - ZERO_CELSIUS
    eutectique = temperature_de_liquidus(2677.3, 279.87, 0.259)
    assert eutectique == pytest.approx(128.73, abs=0.01)
    assert pytest.approx(-144.42) == 128.73 - ZERO_CELSIUS
    assert pytest.approx(6.72) == 279.87 - ZERO_CELSIUS


# --- Mai 2025 ------------------------------------------------------------

H_VAPEUR_2025 = {
    "1": 787.2,
    "2ISOS": 613.2,
    "2": 633.4,
    "3": 654.2,
    "4": 817.2,
    "5ISOS": 583.5,
    "5": 610.6,
}
LIQUIDES_2025 = {"6": 30.0, "7": 32.42, "8": 36.02, "9": 145.0, "10": 54.94}
TITRE_2025 = (H_VAPEUR_2025["2"] - LIQUIDES_2025["9"]) / (
    H_VAPEUR_2025["3"] - LIQUIDES_2025["9"]
)


def test_2025_question_1_les_deux_turbines_ont_le_meme_rendement() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, question 1, tableau.

    L'enonce `Exam_2025_Mai_Exercice_cote_Enonce.md` impose un meme rendement
    isentropique aux turbines T1 et T2 : les six enthalpies doivent le
    rendre.
    """
    h = H_VAPEUR_2025
    premiere = (h["1"] - h["2"]) / (h["1"] - h["2ISOS"])
    seconde = (h["4"] - h["5"]) / (h["4"] - h["5ISOS"])
    assert premiere == pytest.approx(seconde, rel=1e-3)


def test_2025_question_1_echangeur_et_vanne() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, question 1, etats 9 a 11.

    Enonce : efficacite de 0,8 pour l'echangeur E, vanne adiabatique. Le
    tableau donne aux liquides une enthalpie egale a leur temperature.
    """
    t = LIQUIDES_2025
    assert t["9"] - 0.8 * (t["9"] - t["7"]) == pytest.approx(t["10"], abs=0.01)
    enthalpie_apres_la_vanne = 54.94
    assert enthalpie_apres_la_vanne == t["10"]


def test_2025_le_bilan_de_l_echangeur_ne_ferme_qu_a_deux_pour_cent() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, question 1, etat 8 : ecart du document.

    Le titre tire du tableau laisse 4,08 % de liquide au ballon, d'ou un
    echauffement de 3,68 kcal/kg ; le document ecrit h8 - h7 = 3,60, soit
    exactement 4 % de liquide. Le texte extrait du PDF porte bien 36.02 :
    la transcription est fidele, c'est la source qui arrondit le titre.
    """
    t = LIQUIDES_2025
    cede = (1 - TITRE_2025) * (t["9"] - t["10"])
    assert cede == pytest.approx(3.68, abs=0.005)
    assert t["8"] - t["7"] == pytest.approx(3.60)
    assert (t["8"] - t["7"]) / (t["9"] - t["10"]) == pytest.approx(0.04, rel=1e-3)
    assert cede == pytest.approx(t["8"] - t["7"], rel=0.03)


def test_2025_question_1_debit_flux_et_rendement() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, questions 1.3 et 1.4.

    Enonce : 146 MW nets. Le debit recalcule vaut 99,78 kg/s contre 99,91
    ecrits : 0,13 % d'ecart, que le document, sans developpement, ne
    permet pas d'attribuer ; la tolerance du debit est donc de 2 pour
    mille. Le flux et le rendement, eux, retombent sur les reponses.
    """
    h = H_VAPEUR_2025
    t = LIQUIDES_2025
    travail = h["1"] - h["2"] + TITRE_2025 * (h["4"] - h["5"]) - (t["7"] - t["6"])
    assert 146e3 / (travail * JOULES_PAR_CALORIE) == pytest.approx(99.91, rel=2e-3)
    chaleur = h["1"] - t["8"] + TITRE_2025 * (h["4"] - h["3"])
    flux = 99.91 * chaleur * JOULES_PAR_CALORIE / 1000
    assert flux == pytest.approx(379.6, rel=1e-3)
    assert pytest.approx(0.385, rel=1e-3) == 146 / 379.6


H_2025 = {
    "1": 1660,
    "2": 1773,
    "3": 1762.5,
    "4ISOS": 1990,
    "4": 2015.3,
    "5": 610,
    "6": 610,
    "7": 1682,
    "8": 470,
    "9": 470,
}


def test_2025_question_2_temperatures_imposees_par_l_enonce() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, question 2, tableau.

    Enonce : sources a -20 et 30 degres C, 10 degres d'ecart aux deux
    echangeurs, 5 degres de surchauffe, 30 degres de sous-refroidissement.
    """
    evaporation = -20 - 10
    assert evaporation == -30
    assert evaporation + 5 == -25
    condensation = 30 + 10
    assert condensation == 40
    assert condensation - 30 == 10


def test_2025_question_2_rendement_isentropique_et_vannes() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, question 2 : eta_ISOS C2 = 0,9."""
    h = H_2025
    assert (h["4ISOS"] - h["3"]) / (h["4"] - h["3"]) == pytest.approx(0.9, rel=1e-3)
    assert h["6"] == h["5"]
    assert h["9"] == h["8"]


def test_2025_question_2_debits_des_deux_compresseurs() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, question 2.3.

    Enonce : C1 consomme 22 kW et cede 8 kW. Le debit derive suit du bilan
    de l'echangeur E, et l'etat 3 du bilan de la chambre de melange.
    """
    h = H_2025
    premier = (22 - 8) / (h["2"] - h["1"])
    assert premier == pytest.approx(0.1239, rel=1e-3)
    derive = 0.1239 * (h["5"] - h["8"]) / (h["7"] - h["6"])
    assert 0.1239 + derive == pytest.approx(0.14008, rel=1e-4)
    melange = (0.1239 * h["2"] + derive * h["7"]) / 0.14008
    assert melange == pytest.approx(h["3"], rel=1e-4)


def test_2025_question_2_puissance_flux_et_psn() -> None:
    """Exam_2025_Mai_Exercice_cote_Solution, question 2.4."""
    h = H_2025
    assert 0.14008 * (h["4"] - h["3"]) == pytest.approx(35.41, rel=1e-3)
    assert 0.1239 * (h["1"] - h["9"]) == pytest.approx(147.44, rel=1e-4)
    assert pytest.approx(2.57, rel=1e-3) == 147.44 / (22 + 35.41)
