"""Les resultats ecrits dans le texte de la fiche de « Mathematique ».

Chaque fragment LaTeX est extrait du TOML, lu comme expression, puis
compare par `comparaison.egales` a ce que `polytech.maths` calcule : la
fiche ne tient plus par un resultat retape a cote.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.differentielles.ordre_n import solution_de_cauchy
from polytech.maths.domain.integration.definie import IntegraleDefinie
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X
from tests.fiches_toml import cherche, latex_vers_sympy, texte

M = "Mathematique"
A = sp.Symbol("a", real=True)


def _lu(fragment: str, **symboles: sp.Symbol) -> sp.Expr:
    expr = latex_vers_sympy(fragment)
    return expr.subs(
        {s: symboles[s.name] for s in expr.free_symbols if s.name in symboles}
    )


def _dans(notion: str, motif: str, groupe: int = 1) -> str:
    return cherche(motif, texte(M, notion), groupe)


def test_la_solution_de_cauchy_de_2022_ecrite_dans_la_fiche() -> None:
    brut = _dans(
        "equations_lineaires_d_ordre_n_a_coefficients_constants",
        r"y = (\\frac\{x\^2\}\{3\}[^$]*)\$",
    )
    lue = _lu(brut, x=T)
    assert egales(
        lue, solution_de_cauchy(OperateurDifferentiel(D**2 - 4 * D + 3), T**2, (0, 0))
    )


def test_la_solution_de_cauchy_de_2025_ecrite_dans_la_fiche() -> None:
    brut = _dans(
        "equations_lineaires_d_ordre_n_a_coefficients_constants",
        r"y = (t\^2 \+ 2 \+ [^$]*)\$",
    )
    operateur = OperateurDifferentiel(D**2 - 5 * D - 14)
    attendue = solution_de_cauchy(operateur, -14 * T**2 - 10 * T - 26, (0, 13))
    assert egales(_lu(brut, t=T), attendue)


def test_les_deux_integrales_definies_ecrites_dans_la_fiche() -> None:
    un, deux, trois = sp.Integer(1), sp.Integer(2), sp.Integer(3)
    log_2022 = _lu(_dans("integration_par_parties", r"vaut \$(\\ln[^$]*)\$"))
    valeur = IntegraleDefinie(sp.log(1 + X) / X**2, un, deux).valeur()
    assert egales(log_2022, valeur)
    texte_2025 = _dans(
        "decomposition_en_fractions_simples", r"\\mathrm\{d\}x = ([^$]*)\$"
    )
    assert egales(
        _lu(texte_2025), IntegraleDefinie(4 * X**2 / (X**4 - 1), deux, trois).valeur()
    )


def test_la_decomposition_ecrite_pour_2025_est_celle_de_la_lib() -> None:
    brut = _dans("decomposition_en_fractions_simples", r"\$(\\frac\{4x\^2\}[^$]*)\$")
    gauche, droite = (_lu(c, x=X) for c in brut.split(" = "))
    fraction = FractionRationnelle(4 * X**2, X**4 - 1)
    assert egales(gauche, fraction.numerateur / fraction.denominateur)
    assert egales(droite, sum(fraction.termes()))


def test_les_coefficients_a_et_b_ecrits_dans_la_formule() -> None:
    t = texte(M, "decomposition_en_fractions_simples")
    a = latex_vers_sympy(cherche(r"A = (\\tfrac\d+)", t, 1))
    b = latex_vers_sympy(cherche(r"B = (-\\tfrac\d+)", t, 1))
    fraction = FractionRationnelle(sp.Integer(1), X**2 - 4)
    assert egales(a / (X - 2) + b / (X + 2), sum(fraction.termes()))


def test_l_image_et_la_condition_de_compatibilite_ecrites_dans_la_fiche() -> None:
    laplace = texte(M, "resolution_d_une_edo_par_la_transformee_de_laplace")
    brut = cherche(r"= (13 - [^$]*)\$", laplace, 1).replace("\n", " ")
    droite = _lu(brut, p=P)
    probleme = ProblemeLineaire((1, -5, -14), -14 * T**2 - 10 * T - 26, (0, 13))
    assert egales(droite, probleme.image() * (P**2 - 5 * P - 14))
    fractions = _lu(
        cherche(r"donne \$(\\frac\{2\}\{p\^3\} \+ \\frac\{2\}\{p\})\$", laplace, 1), p=P
    )
    assert egales(fractions, 2 / P**3 + 2 / P)
    systeme = texte(M, "discussion_d_un_systeme_a_parametres")
    ligne = sp.Matrix(
        [[1, 1, 1, 1 - A], [A, 1 + A, 1 + A, A - A**2], [A, 1 - A, 1 - A, A**2]]
    )
    derniere = echelonner(ligne, 3).matrice[2, 3]
    assert egales(_lu(cherche(r"(a\\,\(2a - 1\)) = 0", systeme, 1), a=A), derniere)
