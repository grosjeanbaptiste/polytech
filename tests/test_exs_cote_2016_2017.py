"""Electromagnetisme : l'exercice cote du 13 mai 2017, quatre questions.

Source : `Electromagnetisme/EXS/Exs_Cote_2016_2017_Solution.pdf`, dix pages
-- enonce imprime, solutions manuscrites --, transcrit dans
`transcriptions/Electromagnetisme/EXS/Exs_Cote_2016_2017_Solution.md`.

Chaque resultat du manuscrit est recopie comme oracle, et recoupe par un
chemin qu'il n'emprunte pas : le court-circuit pour le courant de Norton,
le bilan des puissances par element, l'integration directe de `L di/dt`,
la reseau infini de fils dont la somme est connue.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2017_fils as fi
from polytech.electromagnetisme.application import cote_2017_norton as no
from polytech.electromagnetisme.application import cote_2017_ouverture as ou
from polytech.electromagnetisme.application import cote_2017_puissance as pu

# --- Question 1 : Norton -----------------------------------------------------


def test_q1_le_generateur_de_norton_du_manuscrit() -> None:
    """Page 2 : `R_N = 7/4` ohm, `E_TH = 15/2` V, `I_N = 30/7` A."""
    courant, resistance = no.norton()
    assert resistance == sp.Rational(7, 4)
    assert no.tension_a_vide() == sp.Rational(15, 2)
    assert courant == sp.Rational(30, 7)


def test_q1_le_courant_de_norton_est_celui_du_court_circuit() -> None:
    """Second chemin : le manuscrit divise `E_TH` par `R_N` ; ici on court-circuite."""
    assert no.courant_de_court_circuit() == no.norton()[0]


def test_q1_le_courant_dans_le_dipole() -> None:
    """Page 2 : `R' = 14/15` ohm, `I = 60/29` A, soit 2,07 A."""
    assert no.resistance_en_parallele() == sp.Rational(14, 15)
    assert no.courant_du_dipole() == sp.Rational(60, 29)
    assert float(no.courant_du_dipole()) == pytest.approx(2.07, abs=5e-3)


def test_q1_le_meme_courant_sur_le_circuit_entier() -> None:
    """Troisieme chemin : le dipole branche sur le reseau d'origine, sans Norton."""
    assert no.courant_sur_le_reseau_complet() == no.courant_du_dipole()


def test_q1_le_diviseur_de_courant() -> None:
    """`I = I_N G / (G_N + G + G/2)` : une ligne, sans maille."""
    assert no.courant_par_le_diviseur() == no.courant_du_dipole()


def test_q1_les_trois_chemins_concordent_pour_une_autre_resistance() -> None:
    """L'enonce prend `R = 1` ohm, qui efface toute puissance de `R` : on le change."""
    autre = sp.Integer(3)
    attendu = no.courant_sur_le_reseau_complet(autre)
    assert no.courant_du_dipole(autre) == attendu
    assert no.courant_par_le_diviseur(autre) == attendu
    assert attendu != no.courant_du_dipole()


# --- Question 2 : trois mailles, puis une puissance -------------------------


def test_q2_la_matrice_du_manuscrit() -> None:
    """Page 4 : la matrice 3 x 3 des mailles, symetrique."""
    matrice, second_membre = pu.systeme()
    z2 = pu.R2 + 1 / (sp.I * pu.W * pu.C2)
    attendue = sp.Matrix(
        [
            [pu.R1 + sp.I * pu.W * pu.L1 + z2, -z2, 0],
            [-z2, z2 + sp.I * pu.W * pu.L3, -sp.I * pu.W * pu.L34],
            [0, -sp.I * pu.W * pu.L34, pu.R4 + sp.I * pu.W * pu.L4],
        ]
    )
    assert sp.simplify(matrice - attendue) == sp.zeros(3, 3)
    assert matrice == matrice.T
    assert second_membre == sp.Matrix([pu.E1 - pu.E2, pu.E2, pu.E4])


def test_q2_le_courant_et_la_tension_de_la_branche() -> None:
    """Page 4 : `I12 = -2 - 2j`, `U12 = -440,0 + 578,4j`."""
    assert pu.courant_de_la_branche() == complex(-2, -2)
    tension = pu.tension_de_la_branche()
    assert tension.real == pytest.approx(-440.0, abs=0.05)
    assert tension.imag == pytest.approx(578.4, abs=0.05)


def test_q2_les_puissances_du_manuscrit() -> None:
    """Page 4 : `P = -276,9` W, `Q = -2036,8` var."""
    puissance = pu.puissance_de_la_branche()
    assert puissance.real == pytest.approx(-276.9, abs=0.05)
    assert puissance.imag == pytest.approx(-2036.8, abs=0.05)


def test_q2_le_bilan_par_element() -> None:
    """Second chemin : la source, la resistance, le condensateur, un par un."""
    bilan = pu.bilan_par_element()
    puissance = pu.puissance_de_la_branche()
    assert bilan.resistance == pytest.approx(80.0, rel=1e-12, abs=0)
    assert bilan.condensateur < 0
    assert bilan.source_active < 0
    total = complex(bilan.source_active + bilan.resistance, bilan.source_reactive)
    total += complex(0, bilan.condensateur)
    assert total.real == pytest.approx(puissance.real, rel=1e-12, abs=0)
    assert total.imag == pytest.approx(puissance.imag, rel=1e-12, abs=0)


# --- Question 3 : ouverture sur une bobine ----------------------------------


def test_q3_le_courant_avant_l_ouverture() -> None:
    """Page 6 : `i = E racine(2)/(omega L) cos(omega t + theta - pi/2)`."""
    avant = ou.courant_avant()
    attendu = (
        ou.E * sp.sqrt(2) / (ou.W * ou.L) * sp.cos(ou.W * ou.T + ou.THETA - sp.pi / 2)
    )
    assert sp.simplify(avant - attendu) == 0


def test_q3_ce_courant_verifie_la_loi_de_la_bobine() -> None:
    """Second chemin, sans phaseur : `L di/dt = e(t)`."""
    tension = ou.E * sp.sqrt(2) * sp.cos(ou.W * ou.T + ou.THETA)
    assert sp.simplify(ou.L * ou.courant_avant().diff(ou.T) - tension) == 0


def test_q3_le_courant_apres_l_ouverture() -> None:
    """Page 6 : `i(t) = i(0) e^(-R t / L)`."""
    apres = ou.courant_apres()
    depart = ou.E * sp.sqrt(2) / (ou.W * ou.L) * sp.sin(ou.THETA)
    assert sp.simplify(apres - depart * sp.exp(-ou.R * ou.T / ou.L)) == 0
    assert sp.simplify(ou.R * apres + ou.L * apres.diff(ou.T)) == 0


def test_q3_la_condition_d_extinction() -> None:
    """Page 6 : `theta = k pi`."""
    assert ou.phases_d_extinction() == (0, sp.pi)
    assert ou.courant_apres().subs(ou.THETA, sp.pi) == 0


def test_q3_l_energie_dissipee_est_celle_de_la_bobine() -> None:
    """`int R i^2 dt = L i(0)^2 / 2` : toute l'energie magnetique part en chaleur."""
    dissipee, emmagasinee = ou.energies()
    assert sp.simplify(dissipee - emmagasinee) == 0


# --- Question 4 : n fils au-dessus d'un plan --------------------------------


def test_q4_le_champ_du_manuscrit() -> None:
    """Page 9 : `E = -(h / pi eps0) (Q/L) somme 1/((i-3)^2 d^2 + h^2)` selon `y`."""
    for nombre in (3, 5, 8):
        _, ey = fi.champ_en_p(nombre)
        termes = [1 / ((i - 3) ** 2 * fi.D**2 + fi.H**2) for i in range(1, nombre + 1)]
        attendu = -fi.H / (sp.pi * fi.EPS) * fi.LINEIQUE * sp.Add(*termes)
        assert sp.simplify(ey - attendu) == 0


def test_q4_la_composante_tangentielle_est_nulle_sur_le_plan() -> None:
    """Quel que soit `n` : c'est ce que les images garantissent."""
    for nombre in (1, 4, 7):
        assert sp.simplify(fi.champ_en_p(nombre)[0]) == 0


def test_q4_la_charge_surfacique() -> None:
    """Page 9 : `sigma = eps0 E_y`."""
    nombre = 6
    sigma = fi.charge_surfacique(nombre)
    assert sp.simplify(sigma - fi.EPS * fi.champ_en_p(nombre)[1]) == 0
    assert sigma.subs({fi.D: 1, fi.H: 1, fi.LINEIQUE: 1}) < 0


def test_q4_un_fil_seul_induit_sa_charge_opposee() -> None:
    """L'integrale de `sigma` sur tout le plan vaut `-Q/L`."""
    assert sp.simplify(fi.charge_induite_par_un_fil() + fi.LINEIQUE) == 0


def test_q4_un_reseau_infini_tend_vers_une_nappe() -> None:
    """`sigma = -(Q/L)/d coth(pi h / d)` : la somme tronquee y tend, reste borne."""
    infinie = fi.charge_surfacique_du_reseau_infini()
    for pas, hauteur in ((1, sp.Rational(1, 2)), (1, 1), (1, 3), (2, 1), (3, 5)):
        valeurs = {fi.D: pas, fi.H: hauteur, fi.LINEIQUE: 1}
        tronquee = float(fi.charge_surfacique_centree(200).subs(valeurs))
        limite = float(infinie.subs(valeurs))
        reste = float(fi.reste_de_la_somme(200).subs(valeurs))
        assert 0 < tronquee - limite < reste
        # Le majorant est serre : il ne depasse l'ecart que de deux pour cent.
        assert reste < 1.02 * (tronquee - limite)
    loin = float(infinie.subs({fi.D: 1, fi.H: 3, fi.LINEIQUE: 1}))
    assert loin == pytest.approx(-1.0, rel=1e-6, abs=0)


def test_q4_sous_un_fil_seul() -> None:
    """Un seul fil, `P` a son pied : `sigma = -(Q/L) / (pi h)`."""
    attendue = -fi.LINEIQUE / (sp.pi * fi.H)
    assert sp.simplify(fi.charge_surfacique_sous_un_fil_seul() - attendue) == 0


def test_q2_les_ordres_de_grandeur_du_document() -> None:
    """La reactance du condensateur, et qui fournit quoi."""
    bilan = pu.bilan_par_element()
    assert pytest.approx(318.3, abs=0.05) == 1 / (pu.PULSATION * pu.CAPACITE)
    assert -bilan.source_active == pytest.approx(356.9, abs=0.05)
    assert -bilan.source_active - bilan.resistance == pytest.approx(276.9, abs=0.05)
