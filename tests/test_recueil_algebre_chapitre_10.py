"""Le chapitre X du recueil d'algebre : valeurs propres et vecteurs propres.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 40
(pages 157 et 174 a 183) ; l'oracle est « Solutions du 10eme chapitre »,
pages 184 a 188 du syllabus. Une matrice de passage ne s'ecrit pas d'une
seule facon : celle de la bibliotheque est eprouvee par ce qu'elle doit
faire -- `P^-1 A P` a la forme annoncee, les valeurs propres sur sa
diagonale.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch10_enonces as enonces
from polytech.maths.application import recueil_algebre_ch10_etude as etude
from polytech.maths.application import recueil_algebre_ch10_parametres as parametres
from polytech.maths.application import recueil_algebre_ch10_reflexions as reflexions
from polytech.maths.application.macros_recueil_algebre_ch10 import macros_des_etudes
from polytech.maths.application.macros_recueil_algebre_ch10_reflexions import (
    macros_du_chapitre_dix,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.reduction import Nature
from polytech.maths.domain.algebre.sous_espaces import engendrent_le_meme_sous_espace

M = sp.Matrix
R = sp.Rational
A, B, AL, BE, T = enonces.A, enonces.B, enonces.ALPHA, enonces.BETA, enonces.T
TR = etude.TROUVE

SPECTRES = {
    (1, ""): {6: (1, 1), -9: (2, 1)},
    (5, ""): {1: (1, 1), 4: (1, 1)},
    (6, ""): {2: (3, 1), 4: (1, 1)},
    (7, ""): {6: (1, 1), -9: (2, 1)},
    (8, ""): {-2: (2, 2), 4: (1, 1)},
    (9, ""): {3: (4, 2)},
    (10, ""): {1: (2, 1)},
    (11, ""): {1: (2, 2), 5: (1, 1)},
    (12, ""): {2: (2, 2), 8: (1, 1)},
    (14, ""): {1: (2, 2), 2: (1, 1), 3: (1, 1)},
    (16, ""): {1: (2, 2), -1: (2, 2)},
    (19, ""): {-2: (2, 1), 2: (3, 2)},
    (20, TR): {-1: (2, 1), 1: (1, 1)},
    (21, ""): {1: (3, 1)},
    (22, ""): {1: (2, 2), -1: (1, 1), -2: (1, 1)},
    (23, ""): {1: (2, 1), A: (1, 1)},
    (24, ""): {1: (1, 1), 2: (1, 1), A: (1, 1)},
    (26, ""): {1: (1, 1), 2: (1, 1), B: (1, 1)},
    (34, ""): {-1: (1, 1), 1: (1, 1), A: (1, 1)},
    (36, ""): {1: (1, 1)},
    (23, "AUn"): {1: (3, 2)},
    (23, "ADeux"): {1: (2, 2), 2: (1, 1)},
    (24, "AUn"): {1: (2, 2), 2: (1, 1)},
    (24, "ADeux"): {1: (1, 1), 2: (2, 2)},
    (25, TR): {-1: (2, 1), -3: (1, 1)},
    (26, "BUnAZero"): {1: (2, 2), 2: (1, 1)},
    (26, "BUn"): {1: (2, 1), 2: (1, 1)},
    (26, "BDeux"): {1: (1, 1), 2: (2, 1)},
    (27, ""): {AL: (3, 1)},
    (28, "AlphaUn"): {2: (3, 2)},
    (28, ""): {2: (2, 1), AL + 1: (1, 1)},
    (29, TR): {2: (3, 1)},
    (30, ""): {1: (2, 1), -1: (1, 1), -2: (1, 1)},
    (31, TR): {1: (2, 1), 2: (1, 1)},
    (32, ""): {-1: (2, 2), 2: (2, 1)},
    (33, TR): {1: (1, 1), 2: (2, 1)},
    (34, "AUn"): {1: (2, 1), -1: (1, 1)},
    (34, "AMoinsUn"): {-1: (2, 1), 1: (1, 1)},
    (35, TR): {2: (3, 1)},
    (36, "AMoinsDeux"): {-1: (2, 1), 1: (1, 1)},
    (36, "ADeux"): {1: (3, 2)},
    (37, TR): {1: (2, 1), 4: (1, 1)},
    (38, "AZero"): {1: (1, 1), 0: (2, 1)},
    (38, "AMoinsUn"): {1: (2, 2), -1: (1, 1)},
    (38, "APositif"): {1: (1, 1)},
    (39, "AUn"): {1: (2, 1), -1: (1, 1)},
    (39, "AZero"): {0: (2, 2), 1: (1, 1)},
    (40, TR): {2: (3, 1)},
}
"""Exercice et cas : chaque valeur propre reelle, `(m_a, m_g)`."""


@pytest.mark.parametrize("cas", sorted(SPECTRES, key=str))
def test_les_valeurs_propres_et_leurs_multiplicites(cas: tuple[int, str]) -> None:
    calcule = {v: (ma, len(base)) for v, ma, base in etude.spectre(*cas)}
    assert calcule == SPECTRES[cas]


@pytest.mark.parametrize("cas", sorted(SPECTRES, key=str))
def test_chaque_vecteur_annonce_est_propre(cas: tuple[int, str]) -> None:
    matrice = etude.matrice(*cas)
    for valeur, _, base in etude.spectre(*cas):
        for vecteur in base:
            colonne = M(list(vecteur))
            assert sp.simplify(matrice * colonne - valeur * colonne).is_zero_matrix
            assert all(sp.denom(sp.together(terme)) == 1 for terme in vecteur)


D, TG, NI = Nature.DIAGONALISABLE, Nature.TRIANGULARISABLE, Nature.NI_L_UN_NI_L_AUTRE
DIAGONALISABLES = [(5, ""), (8, ""), (11, ""), (12, ""), (14, ""), (16, ""), (22, "")]
DIAGONALISABLES += [(23, "ADeux"), (24, "AUn"), (24, "ADeux"), (26, "BUnAZero")]
DIAGONALISABLES += [(38, "AMoinsUn"), (39, "AZero")]
TRIANGULARISABLES = [(1, ""), (6, ""), (7, ""), (9, ""), (19, ""), (21, ""), (30, "")]
TRIANGULARISABLES += [(32, ""), (20, TR), (23, "AUn"), (25, TR), (26, "BUn")]
TRIANGULARISABLES += [(26, "BDeux"), (27, ""), (28, "AlphaUn"), (28, ""), (29, TR)]
TRIANGULARISABLES += [(31, TR), (33, TR), (34, "AUn"), (34, "AMoinsUn"), (35, TR)]
TRIANGULARISABLES += [(36, "AMoinsDeux"), (36, "ADeux"), (37, TR), (38, "AZero")]
TRIANGULARISABLES += [(39, "AUn"), (40, TR)]
NATURES = {
    D: DIAGONALISABLES,
    TG: TRIANGULARISABLES,
    NI: [(10, ""), (38, "APositif")],
}
"""Ce que le syllabus annonce, pages 184 a 188."""
CAS_ET_NATURE = [(cas, nature) for nature, liste in NATURES.items() for cas in liste]


@pytest.mark.parametrize(("cas", "nature"), CAS_ET_NATURE)
def test_la_nature_de_chaque_reduction(cas: tuple[int, str], nature: Nature) -> None:
    assert etude.nature(*cas) is nature


@pytest.mark.parametrize(("cas", "nature"), CAS_ET_NATURE)
def test_chaque_reduction_tient(cas: tuple[int, str], nature: Nature) -> None:
    matrice, reduction = etude.matrice(*cas), etude.reduction(*cas)
    passage, reduite = reduction.passage, reduction.reduite
    assert sp.simplify(passage.det()) != 0
    assert sp.simplify(matrice * passage - passage * reduite).is_zero_matrix
    acquis = sum(ma for _, ma, _ in etude.spectre(*cas))
    assert reduite[:, :acquis].is_upper
    assert (reduite.is_diagonal() is True) == (nature is D)
    assert all(sp.denom(sp.together(terme)) == 1 for terme in passage)


def test_le_passage_ne_degenere_pour_aucun_parametre() -> None:
    """Un determinant constant : la reduction vaut pour tout `a`."""
    for cas in [(26, "BUn"), (26, "BDeux"), (27, ""), (31, TR), (38, "APositif")]:
        assert etude.reduction(*cas).passage.det().is_number
    assert etude.reduction(28, "").passage.det().is_number


UNE_ETAPE = [(7, ""), (20, TR), (23, "AUn"), (25, TR), (26, "BUn"), (26, "BDeux")]
UNE_ETAPE += [(28, "AlphaUn"), (28, ""), (31, TR), (33, TR), (36, "AMoinsDeux")]
UNE_ETAPE += [(36, "ADeux"), (37, TR), (39, "AUn"), (38, "APositif")]
UNE_OU_DEUX = [(6, ""), (9, ""), (10, ""), (19, ""), (21, ""), (27, ""), (29, TR)]
UNE_OU_DEUX += [(35, TR)]


@pytest.mark.parametrize("cas", UNE_ETAPE)
def test_les_reductions_en_une_etape(cas: tuple[int, str]) -> None:
    assert etude.reduction(*cas).etapes == 1


@pytest.mark.parametrize("cas", UNE_OU_DEUX)
def test_les_reductions_en_une_ou_deux_etapes(cas: tuple[int, str]) -> None:
    assert etude.reduction(*cas).etapes in {1, 2}


def test_exercice_40_deux_etapes() -> None:
    assert etude.reduction(40, TR).etapes == 2


def test_la_premiere_etape_isole_les_valeurs_propres() -> None:
    passage, semblable = etude.premiere_etape(40, TR)
    assert semblable[1:, 0].is_zero_matrix
    assert semblable[0, 0] == 2
    matrice = etude.matrice(40, TR)
    assert (matrice * passage - passage * semblable).is_zero_matrix
    assert not semblable.is_upper


CARACTERISTIQUES = {
    (1, ""): -(T - 6) * (T + 9) ** 2,
    (10, ""): (T - 1) ** 2 * (T**2 + 1),
    (23, ""): -((T - 1) ** 2) * (T - A),
    (24, ""): -(T - 1) * (T - 2) * (T - A),
    (26, ""): -(T - 1) * (T - 2) * (T - B),
    (27, ""): -((T - AL) ** 3),
    (28, ""): -((T - 2) ** 2) * (T - AL - 1),
    (34, ""): -(T - 1) * (T + 1) * (T - A),
    (36, ""): -(T - 1) * (T**2 - A * T + 1),
    (37, ""): -(T - 1) * (T - A + 1) * (T - A - 2),
    (38, ""): -(T - 1) * (T**2 + A),
    (39, ""): -(T - 1) * (T - A) * (T + A),
}


@pytest.mark.parametrize("cas", sorted(CARACTERISTIQUES))
def test_les_polynomes_caracteristiques(cas: tuple[int, str]) -> None:
    assert sp.expand(etude.caracteristique(*cas) - CARACTERISTIQUES[cas]) == 0
    assert etude.caracteristique(*cas).is_Mul


RETENUS = {
    20: {AL: 0, BE: -1},
    25: {AL: 8, BE: -4},
    29: {A: 2, B: 3},
    31: {B: 2},
    33: {A: 2, B: 3},
    35: {A: 3},
    37: {A: 2},
    40: {A: 2, B: 3},
}


@pytest.mark.parametrize("numero", sorted(RETENUS))
def test_les_parametres_que_les_conditions_fixent(numero: int) -> None:
    assert parametres.retenus(numero) == RETENUS[numero]


def test_les_candidats_ecartes_sont_nommes() -> None:
    assert {AL: -9, BE: -R(13, 4)} in parametres.candidats(20)
    assert {AL: 4, BE: 0} in parametres.candidats(25)
    assert {B: 0} in parametres.candidats(31)
    assert {A: -1} in parametres.candidats(35)
    assert {A: -1} in parametres.candidats(37)
    assert all(len(parametres.candidats(n)) == 2 for n in (20, 25, 31, 35, 37))


def test_un_candidat_qu_un_autre_contient_n_est_pas_repete() -> None:
    """`b = 0` vaut pour tout `a` : `a = 2, b = 0` n'en est qu'un cas."""
    assert parametres.candidats(29) == ({B: 0}, {A: 2, B: 3})
    assert parametres.candidats(40) == ({B: 0}, {A: 2, B: 3})


def test_un_exercice_sans_parametre_a_fixer_est_refuse() -> None:
    with pytest.raises(KeyError, match="aucun parametre a fixer"):
        parametres.candidats(5)


def test_un_exercice_ou_un_cas_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="dans le chapitre X"):
        etude.matrice(4)
    with pytest.raises(KeyError, match=" : pas de cas "):
        etude.spectre(23, "Absent")


def test_exercices_2_a_4() -> None:
    assert reflexions.exercice_2() == ((5, 3, 0, 7), (1, 0, 0, 0))
    assert reflexions.exercice_3() == 6
    assert reflexions.exercice_4() == (3, False)


def test_un_vecteur_qui_n_est_pas_propre_est_refuse() -> None:
    with pytest.raises(ValueError, match="n'est pas un vecteur propre"):
        reflexions.valeur_associee(M([[1, 2], [3, 4]]), M([1, 0]))
    with pytest.raises(ValueError, match="n'est pas un vecteur propre"):
        reflexions.valeur_associee(M([[1, 2], [3, 4]]), M([0, 0]))


def test_exercice_12_une_matrice_orthogonale() -> None:
    passage, diagonale = reflexions.exercice_12()
    assert sp.simplify(passage.T * passage) == sp.eye(3)
    assert sp.simplify(passage.T * etude.matrice(12) * passage) == diagonale
    assert sorted(diagonale.diagonal()) == [2, 2, 8]


def test_exercice_13_la_puissance() -> None:
    k = reflexions.K
    q = (1 - A - B) ** k
    attendu = M([[B + A * q, A - A * q], [B - B * q, A + B * q]]) / (A + B)
    assert sp.simplify(reflexions.exercice_13() - attendu).is_zero_matrix
    cube = etude.matrice(13) ** 3
    assert sp.simplify(reflexions.exercice_13().subs(k, 3) - cube).is_zero_matrix


def test_exercice_14_une_diagonale_semblable() -> None:
    valeur, autres, diagonale = reflexions.exercice_14()
    assert valeur == 3
    assert autres == (1, 2)
    assert sorted(diagonale.diagonal()) == [1, 1, 2, 3]
    assert diagonale.is_diagonal()


def test_exercice_15_la_racine_carree() -> None:
    passage, diagonale, racine = reflexions.exercice_15()
    assert passage * diagonale * passage.inv() == enonces.CARRE_15
    assert diagonale.is_diagonal()
    assert racine == M([[1, 0, 0], [-1, 1, -1], [1, 0, 2]])
    assert racine**2 == M([[1, 0, 0], [-3, 1, -3], [3, 0, 4]])
    assert all(valeur > 0 for valeur in racine.eigenvals())


def test_exercice_16_sans_polynome_caracteristique() -> None:
    carre, trace, plus, moins = reflexions.exercice_16()
    assert carre == sp.eye(4)
    assert trace == 0
    assert engendrent_le_meme_sous_espace(plus, ((1, 0, -1, 0), (0, 1, -2, 1)))
    assert engendrent_le_meme_sous_espace(moins, ((1, 0, 1, 2), (0, 1, 0, -1)))


def test_exercice_17_la_derivation() -> None:
    matrice, couples = reflexions.exercice_17()
    assert matrice == M([[0, -1], [1, 0]])
    t = reflexions.TEMPS
    assert {valeur for valeur, _ in couples} == {sp.I, -sp.I}
    for valeur, fonction in couples:
        assert sp.simplify(sp.diff(fonction, t) - valeur * fonction) == 0
        assert sp.simplify(fonction - sp.exp(valeur * t)) == 0


def test_exercice_18_la_projection() -> None:
    matrice, droite, plan = reflexions.exercice_18()
    attendue = M(
        [[2, R(1, 2), -R(1, 2)], [-1, R(1, 2), R(1, 2)], [3, R(3, 2), -R(1, 2)]]
    )
    assert matrice == attendue
    assert matrice**2 == matrice
    assert engendrent_le_meme_sous_espace(droite, ((1, -1, 3),))
    assert len(plan) == 2
    assert all(M(list(v)).dot(M([2, 1, -1])) == 0 for v in plan)


def test_exercice_23_la_puissance_pour_a_egal_deux() -> None:
    n = reflexions.N
    attendu = M(
        [
            [1, 0, 0],
            [0, -3 + 2 ** (n + 2), 3 - 3 * 2**n],
            [0, -4 + 2 ** (n + 2), 4 - 3 * 2**n],
        ]
    )
    assert sp.simplify(reflexions.exercice_23() - attendu).is_zero_matrix


def test_le_jeu_de_macros_du_chapitre_dix() -> None:
    macros = macros_du_chapitre_dix()
    assert all(nom.startswith("AlgDix") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros_des_etudes().items() <= macros.items()
    assert macros["AlgDixUnPoly"] == r"- \left(t - 6\right) \left(t + 9\right)^{2}"
    assert r"\lambda = -9 : \; m_a = 2,\; m_g = 1" in macros["AlgDixUnSpectre"]
    assert (
        r"S(6) = \left\langle \left(0,\, 9,\, 1\right) \right\rangle"
        in (macros["AlgDixUnEspaces"])
    )
    assert macros["AlgDixVingtRetenus"] == r"\alpha = 0,\; \beta = -1"
    assert r"\text{ou}" in macros["AlgDixVingtCandidats"]
    assert macros["AlgDixTroisValeur"] == "6"
    assert "algebre-10-valeurs-propres" in JEUX_DE_MACROS


def test_la_premiere_etape_n_est_montree_que_s_il_en_faut_deux() -> None:
    macros = macros_des_etudes()
    assert "AlgDixQuaranteTrouvePremiere" in macros
    assert "AlgDixQuaranteTrouvePremierPassage" in macros
    assert "AlgDixUnPremiere" not in macros
    assert "AlgDixUnPremierPassage" not in macros


GENERIQUES = {
    (23, ""): sp.Integer(1),
    (24, ""): sp.Integer(1),
    (26, ""): (B - 1) * (B - 2),
    (34, ""): (A - 1) * (A + 1),
    (36, ""): sp.Integer(1),
}
"""Le facteur dont depend le determinant du passage : nul, le passage degenere."""


@pytest.mark.parametrize("cas", sorted(GENERIQUES))
def test_une_seule_reduction_pour_tout_parametre(cas: tuple[int, str]) -> None:
    matrice, reduction = etude.matrice(*cas), etude.reduction(*cas)
    passage, reduite = reduction.passage, reduction.reduite
    assert sp.simplify(matrice * passage - passage * reduite).is_zero_matrix
    assert (passage.det() / GENERIQUES[cas]).cancel().is_number
    assert passage.det() != 0
    assert all(sp.denom(sp.together(terme)) == 1 for terme in passage)


def test_les_reduites_generiques_disent_quand_la_matrice_se_diagonalise() -> None:
    assert etude.reduction(23, "").reduite.is_upper
    hors_diagonale = etude.reduction(23, "").reduite - sp.diag(1, A, 1)
    assert sp.solve(hors_diagonale[0, 2], A) == [2]
    assert hors_diagonale[0, 2] != 0
    assert etude.reduction(24, "").reduite == sp.diag(1, 2, A)
    assert etude.reduction(26, "").reduite == sp.diag(1, 2, B)
    assert etude.reduction(34, "").reduite == sp.diag(-1, 1, A)
    assert etude.reduction(36, "").reduite[1:, 0].is_zero_matrix


def test_exercice_40_ce_que_la_moyenne_impose() -> None:
    en_a, lie = parametres.exercice_40()
    assert sp.expand(en_a + B**2 * (A - B + 1)) == 0
    assert sp.expand(lie + (T - A) * (T**2 - 2 * A * T + A + 2)) == 0
    assert en_a.is_Mul
    assert lie.is_Mul


def test_les_cas_secondaires_n_ont_que_leur_spectre() -> None:
    macros = macros_des_etudes()
    assert "AlgDixVingtTroisADeuxSpectre" in macros
    assert "AlgDixVingtTroisADeuxPassage" not in macros
    assert "AlgDixVingtTroisPassage" in macros
    assert "AlgDixTrenteNeufAZeroSpectre" in macros
    assert "AlgDixTrenteNeufAZeroReduite" not in macros


def test_exercice_38_pour_a_positif_les_autres_racines_ne_sont_pas_reelles() -> None:
    valeurs = etude.matrice(38, "APositif").eigenvals()
    assert sorted(v.is_real for v in valeurs) == [False, False, True]
