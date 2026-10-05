"""Recalcule les corriges manuscrits de thermodynamique transcrits, 2018 a 2023.

Quatre transcriptions de `transcriptions/Thermodynamique/EXAMENS/` sont
relues ici par le calcul : decembre 2018, decembre 2019, mai 2022 (fichier
nomme Mai2002) et mai 2023. Chaque test repart des operandes ecrits dans le
corrige et doit retomber sur le resultat ecrit ; aucune propriete de fluide
ne vient d'ailleurs que du document.

Trois ecarts appartiennent aux manuscrits et non a la lecture ; chacun a son
test dedie, qui dit ce que le document ecrit et ce que le calcul donne.
"""

import math

import pytest

CAL = 4186.0
ATM = 101325.0
R_CAL = 1.987
ARRONDI = 1e-3


def antoine(a: float, b: float, c: float, celsius: float) -> float:
    """Pression de saturation en mm Hg, loi ecrite dans le corrige de 2018."""
    return float(10 ** (a - b / (celsius + c)))


def benzene(celsius: float) -> float:
    """Benzene, constantes du corrige de decembre 2018, page 3."""
    return antoine(6.89742, 1206.53, 220.91, celsius)


def paraxylene(celsius: float) -> float:
    """Paraxylene, constantes du corrige de decembre 2018, page 3."""
    return antoine(6.98650, 1451.39, 215.148, celsius)


def test_2018_question_1_etats_du_cycle_de_l_azote() -> None:
    """Exam_2018_Decembre_Exercice_cote_Solution, question 1 : T, p, V des etats."""
    masse, r, k = 3.0, 297.0, 1.4
    assert masse * r * (20 + 273.15) / 1e5 == pytest.approx(2.612, rel=ARRONDI)
    assert 1 * 10**k == pytest.approx(25.12, rel=ARRONDI)
    assert 25.12e5 * 0.2612 / (masse * r) == pytest.approx(736.36, rel=ARRONDI)
    assert pytest.approx(463.21) == 736.36 - 273.15
    assert pytest.approx(0.4576, rel=ARRONDI) == 0.2612 + 493.29e3 / 25.12e5
    assert 25.12e5 * 0.4576 / (masse * r) == pytest.approx(1290.11, rel=ARRONDI)
    assert pytest.approx(1016.96) == 1290.11 - 273.15
    assert masse * r * 1290.11 / 2.612 / 1e5 == pytest.approx(4.40, rel=ARRONDI)


def test_2018_question_1_travaux_et_chaleurs() -> None:
    """Exam_2018_Decembre_Exercice_cote_Solution, question 1 : W et Q, en kJ."""
    m_cv, m_r = 3 * 742e-3, 3 * 297e-3
    w12 = m_cv * (736.36 - 293.15)
    assert w12 == pytest.approx(986.59, rel=ARRONDI)
    assert w12 / 2 == pytest.approx(493.29, rel=ARRONDI)
    du23 = m_cv * (1290.11 - 736.36)
    assert du23 == pytest.approx(1232.65, rel=ARRONDI)
    assert du23 + 493.29 == pytest.approx(1725.94, rel=ARRONDI)
    w34 = -m_r * 1290.11 * math.log(2.612 / 0.4576)
    assert w34 == pytest.approx(-2002.27, rel=ARRONDI)
    assert m_cv * (293.15 - 1290.11) == pytest.approx(-2219.23, rel=ARRONDI)


def test_2018_question_1_le_cycle_se_referme() -> None:
    """Exam_2018_Decembre_Exercice_cote_Solution, question 1 : somme Q = -somme W."""
    chaleurs = 0 + 1725.94 + 2002.27 - 2219.23
    travaux = 986.59 - 493.29 - 2002.27 + 0
    assert chaleurs == pytest.approx(-travaux, abs=0.05)


def test_2018_question_2_ebullition_et_equilibre_a_380_k() -> None:
    """Exam_2018_Decembre_Exercice_cote_Solution, question 2 : Antoine et Raoult."""
    bulle = 0.4 * benzene(103.26) + 0.6 * paraxylene(103.26)
    assert bulle == pytest.approx(760, rel=ARRONDI)
    assert pytest.approx(376.41) == 103.26 + 273.15
    celsius = 380 - 273.15
    assert benzene(celsius) == pytest.approx(1645.44, rel=1e-5)
    assert paraxylene(celsius) == pytest.approx(301.34, rel=1e-4)
    liquide = (760 - 301.34) / (1645.44 - 301.34)
    assert liquide == pytest.approx(0.3412, rel=ARRONDI)
    assert liquide * 1645.44 / 760 == pytest.approx(0.7388, rel=ARRONDI)
    assert pytest.approx(1 / 760, rel=ARRONDI) == 0.7388 / 1645.44 + 0.2612 / 301.34
    assert pytest.approx(0.6588) == 1 - 0.3412
    assert pytest.approx(0.2612) == 1 - 0.7388


def test_2018_question_2_solidification_et_benzene_a_ajouter() -> None:
    """Exam_2018_Decembre_Exercice_cote_Solution, question 2 : etat 4 et ajout."""
    eutectique = 0.6312
    assert eutectique < 0.7388, "c'est bien le benzene qui se depose"
    inverse = 1 / (273.15 + 5.53) - math.log(0.7388) * R_CAL / 2358
    assert 1 / inverse == pytest.approx(260.18, rel=1e-4)
    ajout = (0.7388 * 10 - 4) / (1 - 0.7388)
    assert ajout == pytest.approx(12.97, rel=ARRONDI)


def test_2019_question_1_energies_internes() -> None:
    """Exam_2019_Decembre_Exercice_cote_Solution, question 1 : u = h - p v."""
    assert pytest.approx(2) == 600 / 300
    assert pytest.approx(765940, rel=1e-5) == 197.5 * CAL - 2 * ATM * 0.3
    assert pytest.approx(574147, rel=1e-5) == 142 * CAL - 20 * ATM * 0.01
    assert pytest.approx(496266, rel=1e-5) == 124 * CAL - 0.75 * ATM * 0.3
    assert pytest.approx(723007, rel=1e-5) == 184 * CAL - 2 * ATM * 0.233


def test_2019_le_manuscrit_ecrit_101325_pour_4186_dans_u4() -> None:
    """Exam_2019_Decembre_Exercice_cote_Solution, question 1, etat 4.

    Le document ecrit `184 x 101325 - 2 x 101325 x 0,233 = 723007` : relu sur
    l'image, c'est bien ce qu'il porte. L'expression ecrite vaut 18,6 MJ/kg ;
    le resultat ecrit est celui du facteur 4186. Faute de plume de la source.
    """
    assert pytest.approx(18_596_583, rel=1e-6) == 184 * ATM - 2 * ATM * 0.233
    assert pytest.approx(723007, rel=0.5) != 184 * ATM - 2 * ATM * 0.233


def test_2019_question_1_travaux_et_chaleurs() -> None:
    """Exam_2019_Decembre_Exercice_cote_Solution, question 1 : W et Q, en joules."""
    assert 2 * (496266 - 765940) == -539348
    assert 2 * (574147 - 496266) == 155762
    q34 = 254.15 * 2 * (1.095 - 0.835) * CAL
    assert pytest.approx(254.15) == -19 + 273.15
    assert q34 == pytest.approx(553213, rel=1e-5)
    assert 2 * (723007 - 574147) == 297720
    assert 297720 - 553213 == -255493
    assert pytest.approx(-27155, rel=1e-4) == -2 * ATM * 2 * (0.3 - 0.233)
    assert 2 * (765940 - 723007) == 85866
    assert 85866 + 27155 == 113021


def test_2019_question_1_le_cycle_se_referme() -> None:
    """Exam_2019_Decembre_Exercice_cote_Solution, question 1 : somme Q = -somme W."""
    assert -539348 + 0 + 553213 + 113021 == -(0 + 155762 - 255493 - 27155)


def test_2019_question_2_affinite_et_constante_a_298_k() -> None:
    """Exam_2019_Decembre_Exercice_cote_Solution, question 2 : mu, a et ln K."""
    temperature = 298.15
    assert -94052 - temperature * 51.061 == pytest.approx(-109275.8, rel=1e-6)
    assert -temperature * 6.49 == pytest.approx(-1935, rel=1e-4)
    assert -26416 - temperature * 47.301 == pytest.approx(-40518.8, rel=1e-6)
    assert -63700 - temperature * 14.2 == pytest.approx(-67933.7, rel=1e-6)
    affinite = -(-109275.8 - 1935 + 40518.8 + 67933.7)
    assert affinite == pytest.approx(2758.3)
    assert affinite / (R_CAL * temperature) == pytest.approx(4.656, rel=1e-4)
    assert -(-94052 + 0 + 63700 + 26416) == 3936
    assert pytest.approx(1980.9, rel=1e-4) == 3936 / R_CAL


def test_2019_le_manuscrit_arrondit_298_dans_van_t_hoff() -> None:
    """Exam_2019_Decembre_Exercice_cote_Solution, question 2 : ln K(1273).

    Le document ecrit -0,435, relu sur l'image. Sa propre formule, avec le
    298,15 qu'elle affiche, donne -0,432 ; -0,435 est ce que donne 298 tout
    court. Ecart de 0,7 % sur ln K, imputable a la source, et K = 0,647 en
    decoule sans autre ecart.
    """
    pente = 1980.9
    assert 4.656 + pente * (1 / 1273 - 1 / 298.15) == pytest.approx(-0.432, abs=5e-4)
    assert 4.656 + pente * (1 / 1273 - 1 / 298) == pytest.approx(-0.435, abs=5e-4)
    assert math.exp(-0.435) == pytest.approx(0.647, rel=ARRONDI)


def test_2019_question_2_composition_et_volume_de_gaz() -> None:
    """Exam_2019_Decembre_Exercice_cote_Solution, question 2 : xi, titres, V."""
    avancement = 0.647 / (1 + 0.647)
    assert avancement == pytest.approx(0.3928, rel=ARRONDI)
    assert pytest.approx(0.1309, rel=ARRONDI) == 0.3928 / 3
    assert pytest.approx(0.2024, rel=ARRONDI) == (1 - 0.3928) / 3
    assert pytest.approx(0.6667, rel=ARRONDI) == 2 / 3
    assert pytest.approx(1) == 0.1309 + 0.2024 + 0.6667
    moles = 3 * 1000 / (0.3928 * 55.85)
    assert moles == pytest.approx(136.75, rel=1e-4)
    assert pytest.approx(14.28, rel=ARRONDI) == 136.75 * 8.314 * 1273 / ATM


def test_2022_question_1_bilans_de_la_centrale() -> None:
    """Exam_2022_Mai_Exercice_cote_Reponses, question 1 : recoupements.

    Le document ne donne que des resultats. Se recoupent : l'efficacite 0,8
    de l'echangeur (enonce), son bilan d'enthalpie, la puissance nette de
    100 MW (enonce), le flux chaud et le rendement.
    """
    haute, basse = 124.45, 110.76
    h1, h2, h2l, h2v, h3, h4, h5, h6, h7 = (
        669.8,
        588.5,
        160.0,
        658.7,
        533.34,
        40.0,
        40.92,
        51.4,
        64.74,
    )
    assert (h2l - h7) / (h2l - h5) == pytest.approx(0.8, rel=ARRONDI)
    assert (haute - basse) * (h2l - h7) == pytest.approx(haute * (h6 - h5), rel=ARRONDI)
    nette = haute * (h2 - h1) + basse * (h3 - h2v) + haute * (h5 - h4)
    assert nette * CAL / 1e6 == pytest.approx(-100, rel=ARRONDI)
    assert haute * (h1 - h6) * CAL / 1e6 == pytest.approx(322.15, rel=1e-4)
    assert pytest.approx(0.31, abs=5e-3) == 100 / 322.15


def test_2022_le_titre_implicite_ne_suit_pas_la_regle_des_moments() -> None:
    """Exam_2022_Mai_Exercice_cote_Reponses, question 1 : debits.

    Le rapport des debits ecrits vaut 0,890 ; la regle des moments sur les
    enthalpies du tableau (h2, h2L, h2V, relues sur l'image) donne 0,859.
    Le document n'ecrit pas le titre : l'ecart de 3,5 % est celui de la
    source, dont le calcul intermediaire n'est pas porte.
    """
    assert pytest.approx(0.890, abs=5e-4) == 110.76 / 124.45
    assert pytest.approx(0.859, abs=5e-4) == (588.5 - 160) / (658.7 - 160)


def test_2022_question_2_machine_frigorifique() -> None:
    """Exam_2022_Mai_Exercice_cote_Reponses, question 2 : recoupements."""
    premier, second = 0.0678, 0.0740
    h1, h2, h3, h4, h5, h6, h9 = 1615, 1910, 1720, 1902, 567, 465, 1680
    assert 20 / (h2 - h1) == pytest.approx(premier, rel=ARRONDI)
    assert (second - premier) * (h9 - h5) == pytest.approx(
        premier * (h5 - h6), rel=3e-3
    ), "bilan de l'echangeur, a l'arrondi de la difference des debits pres"
    assert second * (h4 - h3) * 1e3 == pytest.approx(13468, rel=1e-4)
    assert premier * (h1 - h6) * 1e3 == pytest.approx(77970, rel=1e-4)
    assert pytest.approx(2.33, abs=5e-3) == 77970 / (20000 + 13468)
    melange = second * h3 - premier * h2 - (second - premier) * h9
    assert melange * 1e3 == pytest.approx(-12634, rel=1e-4)


def test_2023_question_1_cycle_de_joule() -> None:
    """Exam_2023_Mai_Exercice_cote_Solution, question 1 : etats de l'air."""
    isentropique = 298.15 * 12 ** (0.4 / 1.4)
    assert isentropique == pytest.approx(606.42, rel=1e-5)
    assert pytest.approx(333.27) == 606.42 - 273.15
    reelle = 298.15 + (606.42 - 298.15) / 0.9
    assert reelle == pytest.approx(640.67, rel=1e-5)
    assert pytest.approx(367.52) == 640.67 - 273.15
    assert pytest.approx(120.02, rel=1e-4) == -37.4e6 / (1005.2 * (215 - 525))
    assert pytest.approx(185.2, rel=ARRONDI) == 215 - 3.6e6 / (120.02 * 1005.2)


def test_2023_question_1_cycle_de_hirn() -> None:
    """Exam_2023_Mai_Exercice_cote_Solution, question 1 : etats de l'eau."""
    pompe = 1e-3 * (100 - 0.05) * ATM
    assert pompe == pytest.approx(10127, rel=1e-4)
    assert pytest.approx(2.42, rel=ARRONDI) == 10127 / CAL
    assert pytest.approx(37.42) == 35 + 2.42
    assert pytest.approx(11.431, rel=1e-4) == 37.4e6 / ((819 - 37.42) * CAL)
    assert pytest.approx(615.8, rel=1e-4) == 691 - 3.6e6 / (11.431 * CAL)


def test_2023_question_1_les_flux_des_echangeurs_changent_de_signe() -> None:
    """Exam_2023_Mai_Exercice_cote_Solution, pages 2 et 3.

    Negatifs pour l'air qui cede, positifs pour l'eau qui recoit : les deux
    pages decrivent le meme echange, vu de chaque fluide.
    """
    air = 120.02 * 1005.2
    eau = 11.431 * CAL
    assert air * (215 - 525) == pytest.approx(-eau * (819 - 37.42), rel=ARRONDI)
    assert air * (185.2 - 215) == pytest.approx(-eau * (691 - 615.8), rel=ARRONDI)


def test_2023_question_1_puissances_et_rendement() -> None:
    """Exam_2023_Mai_Exercice_cote_Solution, question 1, point 4, en MW."""
    air = 120.02 * 1005.2 / 1e6
    eau = 11.431 * CAL / 1e6
    assert air * (1200 - 367.52) == pytest.approx(100.43, rel=1e-4)
    assert air * (367.52 - 25) == pytest.approx(41.32, rel=ARRONDI)
    assert air * (525 - 1200) == pytest.approx(-81.43, rel=ARRONDI)
    assert eau * (37.42 - 35) == pytest.approx(0.12, abs=5e-3)
    assert eau * (615.8 - 819) == pytest.approx(-9.72, rel=ARRONDI)
    assert eau * (565 - 691) == pytest.approx(-6.03, rel=ARRONDI)
    assert pytest.approx(-55.74) == 41.32 + 0.12 - 81.43 - 9.72 - 6.03
    assert pytest.approx(0.555, rel=ARRONDI) == 55.74 / 100.43


def test_2023_question_2_pompe_a_chaleur_a_15_degres() -> None:
    """Exam_2023_Mai_Exercice_cote_Solution, question 2, point 1."""
    assert (0 - 10, 15 + 7, -10 + 5) == (-10, 22, -5)
    assert pytest.approx(426) == 396 + (420 - 396) / 0.8
    debit = -5 / (231 - 426)
    assert debit == pytest.approx(0.02564, rel=ARRONDI)
    assert pytest.approx(0.769, rel=ARRONDI) == 0.02564 * (426 - 396)
    assert pytest.approx(6.50, rel=ARRONDI) == 5 / 0.769


def test_2023_question_2_pompe_a_chaleur_a_60_degres() -> None:
    """Exam_2023_Mai_Exercice_cote_Solution, question 2, point 2."""
    assert 60 + 7 == 67
    assert pytest.approx(457.3, rel=1e-3) == 396 + (445 - 396) / 0.8
    assert pytest.approx(0.01795, rel=ARRONDI) == 0.7 * 0.02564
    assert pytest.approx(-2.859, rel=ARRONDI) == 0.01795 * (298 - 457.3)
    assert pytest.approx(1.100, rel=ARRONDI) == 0.01795 * (457.3 - 396)
    assert pytest.approx(2.60, rel=ARRONDI) == 2.859 / 1.100


def test_2023_bareme_des_pages_et_des_questions() -> None:
    """Exam_2023_Mai_Exercice_cote_Solution : totaux entoures.

    Chaque total de page (6, 13, 18, 8, 22, 23) est la somme des points
    portes en marge, le nombre de cotes ecrit a cote est leur nombre, et
    chaque question fait le 45 entoure en regard de son titre.
    """
    pages: dict[float, list[float]] = {
        6: [3, 3],
        13: [0.5, 0.5, 0.5, 2, 2, 0.5, 0.5, 0.5, 3, 3],
        18: [0.5, 0.5, 1, 0.5, 1, 0.5, 1, 1, 1, 1, 0.5, 2, 0.5, 3, 3, 1],
        8: [3, 3, 2],
        22: [2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 2, 2],
        23: [2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    }
    for total, points in pages.items():
        assert sum(points) == total
    cotes = {13: 10, 18: 16, 8: 3, 22: 14, 23: 15}
    for total, nombre in cotes.items():
        assert len(pages[total]) == nombre
    assert 6 + 13 + 18 + 8 == 22 + 23 == 45
