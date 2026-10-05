"""Syllabus d'electricite generale, chapitre 13 : reseaux mailles et theoremes.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 13-27 a 13-41, lues sur les pages rendues.

Deux graphes viennent du cours : celui de la figure 13.17.1, a six noeuds
et onze branches, avec ses arbres (c) et (d) ; et le tetraedre de la
figure 13.18.1, dont les courants de branche sont recopies. Les valeurs
des elements sont choisies ici, le cours n'en donnant pas ; l'oracle est
alors l'analyse nodale, qui ne choisit aucune maille.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.arbre import (
    est_un_arbre,
    maille_fondamentale,
    nombre_de_mailles_independantes,
)
from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    resistance_vue_des_bornes,
    tension_a_vide,
)
from polytech.electromagnetisme.domain.circuits.mailles import (
    Branche,
    boucle_par_les_noeuds,
    courants_des_branches,
    residu_d_une_maille,
    residus_des_noeuds,
    resoudre_par_les_mailles,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.circuits.stationnaires import Thevenin
from polytech.electromagnetisme.domain.circuits.theoremes import (
    conductance_de_transfert,
    par_superposition,
)
from polytech.electromagnetisme.domain.electrostatique.coefficients import (
    residu_de_l_identite_de_gauss,
)

GRAPHE = (
    ("3", "4"),
    ("3", "1"),
    ("4", "5"),
    ("1", "5"),
    ("3", "2"),
    ("2", "1"),
    ("2", "6"),
    ("6", "4"),
    ("6", "5"),
    ("3", "6"),
    ("1", "6"),
)  # fmt: skip -- les onze branches de la figure 13.17.1a, une par couple
ARBRE_C = (4, 5, 6, 7, 8)
ARBRE_D = (6, 7, 8, 9, 10)


def test_le_reseau_de_la_figure_13_17_1_a_six_mailles_independantes() -> None:
    """Page 13-31 : `b - (n - 1)`, soit `11 - 5`."""
    assert nombre_de_mailles_independantes(noeuds=6, branches=len(GRAPHE)) == 6


def test_un_reseau_qui_ne_tient_pas_ensemble_est_refuse() -> None:
    with pytest.raises(ValueError, match="trop peu de branches"):
        nombre_de_mailles_independantes(noeuds=6, branches=4)


def test_les_arbres_c_et_d_joignent_tous_les_noeuds_sans_maille() -> None:
    """Page 13-31 : « l'arbre comporte necessairement n - 1 branches »."""
    assert est_un_arbre(GRAPHE, ARBRE_C)
    assert est_un_arbre(GRAPHE, ARBRE_D)
    # Cinq branches, mais une maille 3-4-5-1 et les noeuds 2 et 6 oublies.
    assert not est_un_arbre(GRAPHE, (0, 1, 2, 3, 4))
    assert not est_un_arbre(GRAPHE, (4, 5, 6, 7))


def test_avec_l_arbre_d_toutes_les_mailles_sont_triangulaires() -> None:
    """Page 13-31 : l'etoile du noeud 6 donne des equations a trois termes."""
    co_arbre = [rang for rang in range(len(GRAPHE)) if rang not in ARBRE_D]
    for maillon in co_arbre:
        assert len(maille_fondamentale(GRAPHE, ARBRE_D, maillon)) == 3
    plus_longue = max(
        len(maille_fondamentale(GRAPHE, ARBRE_C, m))
        for m in range(len(GRAPHE))
        if m not in ARBRE_C
    )
    assert plus_longue > 3


def test_une_branche_de_l_arbre_ne_ferme_aucune_maille() -> None:
    with pytest.raises(ValueError, match="appartient a l'arbre"):
        maille_fondamentale(GRAPHE, ARBRE_D, 6)


R12, R14, R42, R23, R31, R34 = sp.symbols(
    "R_12 R_14 R_42 R_23 R_31 R_34", positive=True
)
E12, E14, E42, E23, E31, E34 = sp.symbols("E_12 E_14 E_42 E_23 E_31 E_34", real=True)
TETRAEDRE = (
    Branche("1", "2", R12, E12),
    Branche("1", "4", R14, E14),
    Branche("4", "2", R42, E42),
    Branche("2", "3", R23, E23),
    Branche("3", "1", R31, E31),
    Branche("3", "4", R34, E34),
)
"""Figure 13.18.1 : les trois premieres branches portent `I_a`, `I_b`, `I_c`."""
ETOILE = (3, 4, 5)
IA, IB, IC = sp.symbols("I_a I_b I_c", real=True)


def test_les_courants_de_maille_donnent_ceux_de_la_figure_13_18_1() -> None:
    """Exemple 13.18.1 : `I_23 = I_a + I_c`, `I_31 = I_a + I_b`, `I_34 = I_c - I_b`."""
    courants = courants_des_branches(TETRAEDRE, ETOILE, (IA, IB, IC))
    assert courants == (IA, IB, IC, IA + IC, IA + IB, IC - IB)


def test_les_courants_de_maille_satisfont_d_office_la_loi_des_noeuds() -> None:
    """(13.17.1) : « automatiquement », dit la page 13-32."""
    courants = courants_des_branches(TETRAEDRE, ETOILE, (IA, IB, IC))
    assert set(residus_des_noeuds(TETRAEDRE, courants).values()) == {0}


def test_l_equation_de_la_maille_3143_est_celle_du_syllabus() -> None:
    """Page 13-33 : la somme des forces electromotrices dans le sens 3-1-4-3.

    « est egale a `R_31 (I_a + I_b) + R_14 I_b - R_43 (I_c - I_b)` ».
    """
    courants = courants_des_branches(TETRAEDRE, ETOILE, (IA, IB, IC))
    boucle = boucle_par_les_noeuds(TETRAEDRE, ("3", "1", "4", "3"))
    assert boucle == ((4, 1), (1, 1), (5, -1))
    residu = residu_d_une_maille(TETRAEDRE, boucle, courants)
    chutes = R31 * (IA + IB) + R14 * IB - R34 * (IC - IB)
    assert sp.expand(residu - (E31 + E14 - E34 - chutes)) == 0


def test_un_parcours_sans_branche_est_refuse() -> None:
    carre = (Branche("1", "2", 1, 0), Branche("2", "3", 1, 0), Branche("3", "1", 1, 0))
    with pytest.raises(ValueError, match="aucune branche entre"):
        boucle_par_les_noeuds(carre, ("1", "2", "4"))


VALEURS = {
    R12: 2,
    R14: 3,
    R42: 1,
    R23: 4,
    R31: 5,
    R34: 6,
    E12: 10,
    E14: 0,
    E42: 0,
    E23: 0,
    E31: 7,
    E34: 0,
}  # fmt: skip -- deux forces electromotrices, six resistances


def _chiffre(branches: tuple[Branche, ...]) -> tuple[Branche, ...]:
    return tuple(
        Branche(
            b.de, b.vers, sp.sympify(b.resistance).subs(VALEURS), b.fem.subs(VALEURS)
        )
        for b in branches
    )


def _par_les_noeuds(branches: tuple[Branche, ...]) -> tuple[sp.Expr, ...]:
    """Les courants de branche par l'analyse nodale : `(V_de - V_vers + E) / R`."""
    elements: list[SourceDeTension | Resistance] = []
    for rang, branche in enumerate(branches):
        milieu = f"m{rang}"
        elements.append(SourceDeTension(milieu, branche.de, branche.fem))
        elements.append(Resistance(milieu, branche.vers, branche.resistance))
    potentiels = Reseau(tuple(elements), reference=branches[0].de).potentiels()
    return tuple(
        (potentiels[b.de] - potentiels[b.vers] + b.fem) / b.resistance for b in branches
    )


def test_la_methode_des_mailles_et_l_analyse_nodale_s_accordent() -> None:
    """Trois inconnues au lieu de six : `b - (n - 1)` equations seulement."""
    branches = _chiffre(TETRAEDRE)
    par_les_mailles = resoudre_par_les_mailles(branches, ETOILE)
    assert par_les_mailles == _par_les_noeuds(branches)
    assert set(residus_des_noeuds(branches, par_les_mailles).values()) == {0}


def test_le_choix_de_l_arbre_ne_change_pas_les_courants() -> None:
    """Page 13-31 : « le choix [...] de l'arbre [est] arbitraire »."""
    branches = _chiffre(TETRAEDRE)
    autre_arbre = (0, 1, 5)
    assert resoudre_par_les_mailles(branches, autre_arbre) == (
        resoudre_par_les_mailles(branches, ETOILE)
    )


def test_chaque_maille_du_reseau_resolu_verifie_la_deuxieme_espece() -> None:
    """(13.17.3) sur les mailles 1241 et 2342 du syllabus, hors de tout arbre."""
    branches = _chiffre(TETRAEDRE)
    courants = resoudre_par_les_mailles(branches, ETOILE)
    for parcours in (("1", "2", "4", "1"), ("2", "3", "4", "2")):
        boucle = boucle_par_les_noeuds(branches, parcours)
        assert residu_d_une_maille(branches, boucle, courants) == 0


def test_un_arbre_qui_n_en_est_pas_un_est_refuse() -> None:
    with pytest.raises(ValueError, match="pas un arbre"):
        resoudre_par_les_mailles(_chiffre(TETRAEDRE), (0, 1, 2))


def test_le_theoreme_de_superposition() -> None:
    """13.19 : chaque force electromotrice seule, les autres annulees."""
    branches = _chiffre(TETRAEDRE)
    assert par_superposition(branches, ETOILE) == (
        resoudre_par_les_mailles(branches, ETOILE)
    )


def test_le_theoreme_de_reciprocite() -> None:
    """(13.22.2) : `I_2 / E_1 = I'_1 / E'_2`, les conductances de transfert."""
    passif = tuple(Branche(b.de, b.vers, b.resistance, 0) for b in _chiffre(TETRAEDRE))
    aller = conductance_de_transfert(passif, ETOILE, source=0, mesure=4)
    retour = conductance_de_transfert(passif, ETOILE, source=4, mesure=0)
    assert aller == retour
    assert aller != 0


def test_la_reciprocite_generale_est_une_identite_de_gauss() -> None:
    """(13.22.1) : `somme E'_kj I_jk = somme E_kj I'_jk`, analogue de (9.5.1)."""
    premier = _chiffre(TETRAEDRE)
    second = tuple(
        Branche(b.de, b.vers, b.resistance, fem)
        for b, fem in zip(premier, (0, 3, -2, 0, 0, 9), strict=True)
    )
    courants = resoudre_par_les_mailles(premier, ETOILE)
    autres = resoudre_par_les_mailles(second, ETOILE)
    residu = residu_de_l_identite_de_gauss(
        [b.fem for b in premier], courants, [b.fem for b in second], autres
    )
    assert residu == 0


def _dipole() -> Reseau:
    """Le tetraedre chiffre, sans sa branche 3-4 : un bipole vu de 3 et 4."""
    elements: list[SourceDeTension | Resistance] = []
    for rang, branche in enumerate(_chiffre(TETRAEDRE)[:5]):
        elements.append(SourceDeTension(f"m{rang}", branche.de, branche.fem))
        elements.append(Resistance(f"m{rang}", branche.vers, branche.resistance))
    return Reseau(tuple(elements), reference="4")


def test_le_theoreme_de_thevenin() -> None:
    """(13.20.2) : `I_b = E_T / (R + R_T)`, contre la resolution du reseau entier."""
    tension = tension_a_vide(_dipole(), "3", "4")
    vue = resistance_vue_des_bornes(_dipole(), "3", "4")
    dans_la_charge = tension / (6 + vue)
    assert dans_la_charge == resoudre_par_les_mailles(_chiffre(TETRAEDRE), ETOILE)[5]


def test_le_theoreme_de_norton() -> None:
    """13.21 : `I_N` de court-circuit, `G_N = 1 / R_T` ; en charge, moins."""
    tension = tension_a_vide(_dipole(), "3", "4")
    vue = resistance_vue_des_bornes(_dipole(), "3", "4")
    norton = Thevenin(float(tension), float(vue)).en_norton()
    court_circuit = courant_de_court_circuit(_dipole(), "3", "4")
    assert norton.courant_de_court_circuit == pytest.approx(float(court_circuit))
    assert norton.conductance() == pytest.approx(float(1 / vue))
    en_charge = float(tension / (6 + vue))
    assert abs(en_charge) < abs(norton.courant_de_court_circuit)


def test_une_maille_sans_resistance_n_a_pas_de_courant_determine() -> None:
    """Deux sources ideales en opposition : aucune solution, ou une infinite."""
    contradictoire = (Branche("1", "2", 0, 5), Branche("2", "1", 0, 3))
    with pytest.raises(ValueError, match="pas de solution unique"):
        resoudre_par_les_mailles(contradictoire, (0,))
    indetermine = (Branche("1", "2", 0, 5), Branche("2", "1", 0, -5))
    with pytest.raises(ValueError, match="pas de solution unique"):
        resoudre_par_les_mailles(indetermine, (0,))


def test_la_conductance_de_transfert_d_une_maille_seule_est_celle_de_la_serie() -> None:
    """Deux resistances de 2 et 3 ohms en une maille : un cinquieme de siemens."""
    maille = (Branche("1", "2", 2, 0), Branche("2", "1", 3, 0))
    assert conductance_de_transfert(maille, (0,), source=0, mesure=1) == (
        sp.Rational(1, 5)
    )
