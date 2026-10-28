"""Les figures calculees de la fiche de « Mathematique », relues par la lib.

Chaque nombre, droite, courbe ou racine que le TikZ de
`cursus/fiches/Mathematique.toml` affiche est lu dans le TOML puis
compare a `polytech.maths`.
"""

from __future__ import annotations

import re

import sympy as sp

from polytech.maths.domain.algebre.rang import rang, systeme_compatible
from polytech.maths.domain.algebre.solutions import Nature, nature_des_solutions
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.second_ordre import EquationDuSecondOrdre
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.symboles import CONSTANTE as C
from polytech.maths.domain.symboles import VARIABLE as X
from tests.fiches_toml import courbes, expression, figure, nombres

M = "Mathematique"
SYSTEMES = "discussion_d_un_systeme_a_parametres"
FRACTIONS = "decomposition_en_fractions_simples"
EDO1 = "edo_lineaires_du_premier_ordre"
ORDRE_N = "equations_lineaires_d_ordre_n_a_coefficients_constants"
COUVERTES = {SYSTEMES, FRACTIONS, EDO1, ORDRE_N}


def _droites(tikz: str) -> list[list[sp.Matrix]]:
    """Par `scope`, les droites `a x + b y = c` des deux `\\draw` tracants."""
    resultat = []
    for scope in re.findall(r"\\begin\{scope\}.*?\\end\{scope\}", tikz, re.S):
        lignes = []
        for p in re.findall(
            r"\\draw\[thick[^\]]*\] \(([^)]*)\) -- \(([^)]*)\);", scope
        ):
            (x1, y1), (x2, y2) = nombres(p[0]), nombres(p[1])
            a, b = y2 - y1, x1 - x2
            lignes.append(sp.Matrix([[a, b, a * x1 + b * y1]]))
        resultat.append(lignes)
    return resultat


def test_les_trois_panneaux_sont_unique_impossible_indetermine() -> None:
    attendues = [Nature.UNIQUE, Nature.IMPOSSIBLE, Nature.INDETERMINE]
    panneaux = _droites(figure(M, SYSTEMES))
    assert len(panneaux) == 3
    for lignes, nature in zip(panneaux, attendues, strict=True):
        complete = sp.Matrix.vstack(*lignes)
        systeme = SystemeLineaire(complete[:, :2], complete[:, 2])
        assert nature_des_solutions(systeme) is nature


def test_les_legendes_des_panneaux_disent_les_rangs_de_la_formule() -> None:
    tikz = figure(M, SYSTEMES)
    legendes = re.findall(r"\{\$(.*?)\$\\\\(.*?)\}", tikz)
    mots = {
        Nature.UNIQUE: "une solution",
        Nature.IMPOSSIBLE: "aucune",
        Nature.INDETERMINE: "infinit",
    }
    for (relation, texte), lignes in zip(legendes, _droites(tikz), strict=True):
        complete = sp.Matrix.vstack(*lignes)
        matrice = tuple(
            tuple(float(v) for v in ligne[:2]) for ligne in complete.tolist()
        )
        second = tuple(float(v) for v in complete[:, 2])
        nature = nature_des_solutions(SystemeLineaire(complete[:, :2], complete[:, 2]))
        assert mots[nature] in texte
        assert systeme_compatible(matrice, second) is (nature is not Nature.IMPOSSIBLE)
        assert (rang(matrice) == 2) is ("= n" in relation and "<" not in relation)


def test_les_deux_fractions_simples_tracees_somment_la_fraction() -> None:
    entiere, haut, bas = (expression(c) for c in courbes(figure(M, FRACTIONS)))
    assert egales(haut + bas, entiere)
    termes = FractionRationnelle(sp.Integer(1), X**2 - 4).termes()
    assert egales(sum(termes), entiere.subs(sp.Symbol("x", real=True), X))
    assert {sp.simplify(t) for t in termes} == {
        sp.simplify(haut.subs(sp.Symbol("x", real=True), X)),
        sp.simplify(bas.subs(sp.Symbol("x", real=True), X)),
    }


def test_les_trois_courbes_sont_les_solutions_c_1_0_moins_1() -> None:
    tikz = figure(M, EDO1)
    generale = EquationLineaire(-2 * X, X).solution_generale()
    constantes = nombres(" ".join(re.findall(r"\$C = (-?\d+)\$", tikz)))
    assert constantes == [1.0, 0.0, -1.0]
    for corps, k in zip(courbes(tikz), constantes, strict=True):
        assert egales(
            expression(corps), generale.subs(C, k).subs(X, sp.Symbol("x", real=True))
        )


def test_les_croix_sont_les_racines_des_deux_polynomes_caracteristiques() -> None:
    tikz = figure(M, ORDRE_N)
    croix: dict[str, set[float]] = {}
    for couleur, x in re.findall(
        r"\\node\[(\w+)\] at \((-?\d+),0\) \{\$\\times\$\}", tikz
    ):
        croix.setdefault(couleur, set()).add(float(x))
    polynomes = {
        c: (-float(b), float(k) * (1 if s == "+" else -1))
        for c, b, s, k in re.findall(
            r"(rouge|bleu) : \$r\^2 - (\d+)r ([+-]) (\d+)\$", tikz
        )
    }
    assert set(polynomes) == {"rouge", "bleu"}
    for couleur, anglais in (("rouge", "red"), ("bleu", "blue")):
        b, c = polynomes[couleur]
        racines = EquationDuSecondOrdre(1, b, c).racines_caracteristiques()
        assert croix[anglais] == {float(r) for r in racines}
