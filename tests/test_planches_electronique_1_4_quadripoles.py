"""Les planches d'electronique : quadripoles et analyse nodale modifiee (1, IV).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_1_part4.pdf`, 50
planches lues sur les pages rendues -- leurs formules sont des images.

Les oracles : les matrices Z, Y et ABCD de la cellule RC (planches 5 a
16), les gains et l'impedance d'entree d'un quadripole charge (planche
19), le tableau de conversion de la planche 30, relu au zoom, et les
trois exemples d'analyse nodale modifiee (planches 37 a 42).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.abcd import (
    cascade,
    derivation,
    gain_en_courant,
    gain_en_tension,
    impedance_d_entree,
    serie,
)
from polytech.electronique.domain.conversions_de_quadripole import (
    Jeu,
    abcd_depuis_z,
    associer,
    convertir,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    SourceCommandee,
    SourceDeCourant,
    SourceDeTension,
    SourceDeTensionCommandee,
    resoudre,
    systeme_nodal_modifie,
)
from polytech.electronique.domain.quadripole import matrice_y_en_l, matrice_z_en_l

S = sp.Symbol("s")
R, C = sp.symbols("R C", positive=True)
Z_RC = matrice_z_en_l(R, 1 / (S * C))
Y_RC = matrice_y_en_l(R, S * C)


def test_la_matrice_impedance_de_la_cellule_rc() -> None:
    """Planches 5 a 7 : `z_11 = R + 1/(sC)`, les trois autres valent `1/(sC)`."""
    attendue = sp.Matrix([[R + 1 / (S * C), 1 / (S * C)], [1 / (S * C), 1 / (S * C)]])
    assert sp.simplify(Z_RC - attendue) == sp.zeros(2, 2)


def test_la_matrice_admittance_de_la_cellule_rc() -> None:
    """Planches 11 et 12 : `y_11 = 1/R`, `y_12 = y_21 = -1/R`, `y_22 = 1/R + sC`."""
    attendue = sp.Matrix([[1 / R, -1 / R], [-1 / R, 1 / R + S * C]])
    assert sp.simplify(Y_RC - attendue) == sp.zeros(2, 2)
    assert sp.simplify(Y_RC - Z_RC.inv()) == sp.zeros(2, 2)
    assert sp.simplify(Z_RC * Y_RC) == sp.eye(2)


def test_la_matrice_abcd_de_la_cellule_rc() -> None:
    """Planche 16 : `A = 1 + sRC`, `B = R`, `C = sC`, `D = 1`."""
    attendue = sp.Matrix([[1 + S * R * C, R], [S * C, 1]])
    en_cascade = cascade(serie(R), derivation(1 / (S * C)))
    assert sp.simplify(en_cascade - attendue) == sp.zeros(2, 2)
    assert sp.simplify(abcd_depuis_z(Z_RC) - attendue) == sp.zeros(2, 2)
    with pytest.raises(ValueError, match="z_21 nul"):
        abcd_depuis_z(sp.Matrix([[R, 0], [0, R]]))


def test_la_cascade_multiplie_les_matrices_abcd() -> None:
    """Planche 17 : `A = A^(1) A^(2)`, dans l'ordre de traversee."""
    premiere, seconde = abcd_depuis_z(Z_RC), serie(sp.Integer(50))
    ensemble = cascade(premiere, seconde)
    assert sp.simplify(ensemble - premiere * seconde) == sp.zeros(2, 2)
    assert sp.simplify(ensemble - seconde * premiere) != sp.zeros(2, 2)


def test_le_quadripole_charge_se_reduit_a_un_dipole() -> None:
    """Planche 19 : `V_2/V_1 = 1/(A + B/Z_L)`, `I_2/I_1 = -1/(C Z_L + D)`, `Z_in`."""
    charge = sp.Symbol("Z_L", positive=True)
    matrice = abcd_depuis_z(Z_RC)
    a, b, c, d = matrice
    assert sp.simplify(gain_en_tension(matrice, charge) - 1 / (a + b / charge)) == 0
    assert sp.simplify(gain_en_courant(matrice, charge) + 1 / (c * charge + d)) == 0
    entree = impedance_d_entree(matrice, charge)
    assert sp.simplify(entree - (a * charge + b) / (c * charge + d)) == 0
    a_la_main = R + (1 / (S * C)) * charge / (1 / (S * C) + charge)
    assert sp.simplify(entree - a_la_main) == 0


def test_les_associations_ajoutent_les_matrices() -> None:
    """Planches 9, 13, 26 et 29 : serie `Z`, parallele `Y`, mixtes `H` et `G`."""
    autre = matrice_z_en_l(sp.Integer(10), sp.Integer(40))
    assert associer(Jeu.Z, Z_RC, autre) == Z_RC + autre
    en_parallele = associer(Jeu.Y, Y_RC, Y_RC)
    assert sp.simplify(en_parallele - 2 * Y_RC) == sp.zeros(2, 2)
    moitie = matrice_y_en_l(R / 2, 2 * S * C)
    assert sp.simplify(en_parallele - moitie) == sp.zeros(2, 2)


M11, M12, M21, M22 = sp.symbols("m_11 m_12 m_21 m_22")
M = sp.Matrix([[M11, M12], [M21, M22]])
DET = M.det()
TABLE = {
    (Jeu.Y, Jeu.Z): sp.Matrix([[M22, -M12], [-M21, M11]]) / DET,
    (Jeu.H, Jeu.Z): sp.Matrix([[DET, M12], [-M21, 1]]) / M22,
    (Jeu.G, Jeu.Z): sp.Matrix([[1, -M12], [M21, DET]]) / M11,
    (Jeu.Z, Jeu.Y): sp.Matrix([[M22, -M12], [-M21, M11]]) / DET,
    (Jeu.H, Jeu.Y): sp.Matrix([[1, -M12], [M21, DET]]) / M11,
    (Jeu.G, Jeu.Y): sp.Matrix([[DET, M12], [-M21, 1]]) / M22,
    (Jeu.Z, Jeu.H): sp.Matrix([[DET, M12], [-M21, 1]]) / M22,
    (Jeu.Y, Jeu.H): sp.Matrix([[1, -M12], [M21, DET]]) / M11,
    (Jeu.G, Jeu.H): sp.Matrix([[M22, -M12], [-M21, M11]]) / DET,
    (Jeu.Z, Jeu.G): sp.Matrix([[1, -M12], [M21, DET]]) / M11,
    (Jeu.Y, Jeu.G): sp.Matrix([[DET, M12], [-M21, 1]]) / M22,
    (Jeu.H, Jeu.G): sp.Matrix([[M22, -M12], [-M21, M11]]) / DET,
}
"""Planche 30, recopiee case par case : (connu, desire) -> matrice desiree."""


@pytest.mark.parametrize(("connu", "desire"), sorted(TABLE, key=str))
def test_le_tableau_de_conversion_de_la_planche_30(connu: Jeu, desire: Jeu) -> None:
    """Chaque case se retrouve par l'algebre, sans la table."""
    calculee = convertir(M, depuis=connu, vers=desire)
    assert sp.simplify(calculee - TABLE[connu, desire]) == sp.zeros(2, 2)


def test_une_conversion_aller_et_retour_ne_change_rien() -> None:
    """Et un jeu converti vers lui-meme est lui-meme."""
    assert convertir(M, depuis=Jeu.H, vers=Jeu.H) == M
    hybride = convertir(Z_RC, depuis=Jeu.Z, vers=Jeu.H)
    retour = convertir(hybride, depuis=Jeu.H, vers=Jeu.Z)
    assert sp.simplify(retour - Z_RC) == sp.zeros(2, 2)
    assert sp.simplify(convertir(Z_RC, depuis=Jeu.Z, vers=Jeu.Y) - Y_RC) == sp.zeros(
        2, 2
    )


def test_les_parametres_hybrides_de_la_cellule_rc() -> None:
    """Planche 24 : `h_11 = V_1/I_1` a sortie en court-circuit, soit `R`."""
    hybride = convertir(Z_RC, depuis=Jeu.Z, vers=Jeu.H)
    attendue = sp.Matrix([[R, 1], [-1, S * C]])
    assert sp.simplify(hybride - attendue) == sp.zeros(2, 2)
    inverse = convertir(Z_RC, depuis=Jeu.Z, vers=Jeu.G)
    assert sp.simplify(inverse - hybride.inv()) == sp.zeros(2, 2)


GM = sp.Symbol("g_m", positive=True)


def test_le_modele_hybride_pi_du_mosfet() -> None:
    """Planche 22 : grille ouverte, source `g_m V_GS` et `R` en sortie.

    Les planches posent le probleme sans le resoudre. La grille ne tire
    aucun courant : `y_11 = y_12 = 0`, `y_21 = g_m`, `y_22 = 1/R`, la
    matrice impedance n'existe pas, et le gain sous une charge `R_L`
    vaut `-g_m (R // R_L)`.
    """
    charge = sp.Symbol("R_L", positive=True)
    admittance = sp.Matrix([[0, 0], [GM, 1 / R]])
    with pytest.raises(ValueError, match="conversion impossible"):
        convertir(admittance, depuis=Jeu.Y, vers=Jeu.Z)
    elements = (
        SourceDeTension(1, 0, sp.Symbol("V_1"), sp.Symbol("i_1")),
        SourceCommandee(2, 0, GM, 1, 0),
        Admittance(2, 0, 1 / R),
        Admittance(2, 0, 1 / charge),
    )
    tensions = resoudre(elements, 2)
    gain = sp.simplify(tensions[1] / tensions[0])
    assert sp.simplify(gain + GM * R * charge / (R + charge)) == 0
    assert tensions[2] == 0


RS, IS, VS = sp.symbols("R_S I_S V_S", positive=True)


def test_l_exemple_1_une_source_de_courant() -> None:
    """Planches 37 et 38 : `V_2 = I_S R_S / (CRs + CR_S s + 1)`.

    La planche ecrit la loi des noeuds du noeud 1 avec les courants
    sortants positifs ; la bibliotheque les compte entrants. Les deux
    systemes ont la meme solution.
    """
    elements = (
        SourceDeCourant(0, 1, IS),
        Admittance(1, 0, 1 / RS),
        Admittance(1, 2, 1 / R),
        Admittance(2, 0, S * C),
    )
    matrice, membre = systeme_nodal_modifie(elements, 2)
    planche = sp.Matrix([[1 / RS + 1 / R, -1 / R], [1 / R, -(1 / R + S * C)]])
    assert sp.simplify(matrice - sp.diag(-1, 1) * planche) == sp.zeros(2, 2)
    assert membre == sp.Matrix([-IS, 0])
    tensions = resoudre(elements, 2)
    commun = C * R * S + C * RS * S + 1
    assert sp.simplify(tensions[0] - IS * RS * (C * R * S + 1) / commun) == 0
    assert sp.simplify(tensions[1] - IS * RS / commun) == 0
    assert sp.limit(tensions[1], RS, sp.oo) == IS / (C * S)
    assert (
        sp.simplify(sp.limit(tensions[0], RS, sp.oo) - IS * (C * R * S + 1) / (C * S))
        == 0
    )


def test_l_exemple_2_une_source_de_tension() -> None:
    """Planches 39 et 40 : le courant de la source devient une inconnue."""
    courant = sp.Symbol("I_S")
    elements = (
        SourceDeTension(1, 0, VS, courant),
        Admittance(1, 2, 1 / (RS + R)),
        Admittance(2, 0, S * C),
    )
    matrice, membre = systeme_nodal_modifie(elements, 2)
    serie_ = 1 / (RS + R)
    planche = sp.Matrix(
        [[-serie_, serie_, 1], [serie_, -(serie_ + S * C), 0], [1, 0, 0]]
    )
    assert sp.simplify(matrice - planche) == sp.zeros(3, 3)
    assert membre == sp.Matrix([0, 0, VS])
    v_1, v_2, i_s = resoudre(elements, 2)
    commun = C * R * S + C * RS * S + 1
    assert v_1 == VS
    assert sp.simplify(v_2 - VS / commun) == 0
    assert sp.simplify(i_s - C * VS * S / commun) == 0
    diviseur = (1 / (S * C)) / (RS + R + 1 / (S * C)) * VS
    assert sp.simplify(v_2 - diviseur) == 0


def test_la_forme_mathematique_de_l_analyse_nodale() -> None:
    """Planches 45 a 47 : `[[Y, B], [C, 0]]`, `B` et `C` disant ou sont les sources."""
    courant = sp.Symbol("I_S")
    elements = (
        SourceDeTension(1, 0, VS, courant),
        Admittance(1, 2, 1 / (RS + R)),
        Admittance(2, 0, S * C),
    )
    matrice, _ = systeme_nodal_modifie(elements, 2)
    assert matrice[:2, 2] == sp.Matrix([1, 0])
    assert matrice[2, :2] == sp.Matrix([[1, 0]])
    assert matrice[2, 2] == 0
    assert matrice[:2, :2].is_symmetric()


R1, R2, GAIN, VI = sp.symbols("R_1 R_2 A V_i", positive=True)


def test_l_exemple_3_une_source_de_tension_commandee() -> None:
    """Planches 41 et 42 : `V_1 = A V_i (R_1 + R_2) / (R_1 + R_2 + A R_1)`.

    A gain infini, `V_1 / V_i = 1 + R_2 / R_1` et `V_2 = V_i` : la masse
    virtuelle de l'amplificateur non inverseur.
    """
    elements = (
        SourceDeTension(3, 0, VI, sp.Symbol("i_e")),
        SourceDeTensionCommandee(1, 0, GAIN, (3, 2), sp.Symbol("I_1")),
        Admittance(1, 2, 1 / R2),
        Admittance(2, 0, 1 / R1),
    )
    v_1, v_2, v_3, _, i_1 = resoudre(elements, 3)
    commun = R1 + R2 + GAIN * R1
    assert v_3 == VI
    assert sp.simplify(v_1 - GAIN * VI * (R1 + R2) / commun) == 0
    assert sp.simplify(v_2 - GAIN * R1 * VI / commun) == 0
    assert sp.simplify(i_1 - GAIN * VI / commun) == 0
    ideal = sp.limit(v_1 / VI, GAIN, sp.oo)
    assert sp.simplify(ideal - (1 + R2 / R1)) == 0
    assert sp.limit(v_2, GAIN, sp.oo) == VI
    hors = SourceDeTensionCommandee(1, 0, GAIN, (9, 2), sp.Symbol("I_1"))
    with pytest.raises(ValueError, match="noeud 9 hors du circuit"):
        resoudre((*elements[:1], hors, *elements[2:]), 3)


def test_un_circuit_pathologique_n_a_pas_de_solution() -> None:
    """Planche 43 : deux sources de courant opposees, seules sur un noeud."""
    courant = sp.Symbol("I")
    elements = (SourceDeCourant(1, 0, courant), SourceDeCourant(0, 1, courant))
    with pytest.raises(ValueError, match="systeme nodal singulier"):
        resoudre(elements, 1)
