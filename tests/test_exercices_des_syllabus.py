"""Les exercices que les deux syllabus de theorie posent au fil des pages.

Sources : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf` (pages 13,
83, 90, 131 et 181 du PDF) et
`Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`
(pages 84, 90, 125, 146, 165 et 187).

Trois oracles viennent des pages : la solution partielle que les notes
donnent du systeme (page 13), la transformee `w / (p^2 + w^2)` (page 83),
et les trois mineurs `W_31, W_32, W_33` de l'exemple (page 181). Le reste
est eprouve par ce qu'il doit faire, et par un second chemin.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import exercices_des_syllabus as syllabus
from polytech.maths.application import primitives_des_documents_de_travail as travail
from polytech.maths.application import recueil_primitives_gestes as gestes
from polytech.maths.application.macros_exercices_des_syllabus import intervalle
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.solutions import Nature
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

M = sp.Matrix
R = sp.Rational


def test_page_13_le_systeme_devient_indetermine() -> None:
    """Les notes : `x_2 = -10 + 11 x_1`, `x_3 = -8 + 8 x_1`."""
    nature, solution = syllabus.systeme_modifie()
    assert nature is Nature.INDETERMINE
    x1 = solution[0]
    assert sp.expand(solution[1] - (-10 + 11 * x1)) == 0
    assert sp.expand(solution[2] - (-8 + 8 * x1)) == 0
    matrice = M([[3, -1, 1], [5, 1, -2], [-2, -2, 3]])
    assert sp.expand(matrice * M(list(solution)) - M([2, 6, -4])) == sp.zeros(3, 1)


def test_page_13_le_systeme_d_origine_etait_impossible() -> None:
    assert syllabus.nature_d_origine() is Nature.IMPOSSIBLE


def test_page_83_la_transformee_du_sinus_par_deux_chemins() -> None:
    par_l_integrale, par_linearite = syllabus.laplace_du_sinus()
    attendue = syllabus.OMEGA / (P**2 + syllabus.OMEGA**2)
    assert sp.simplify(par_l_integrale - attendue) == 0
    assert sp.simplify(par_linearite - attendue) == 0


def test_page_90_les_trois_proprietes_de_la_convolution_sur_des_temoins() -> None:
    temoins = syllabus.convolutions()
    assert set(temoins) == {"Commutativite", "Distributivite", "Associativite"}
    for gauche, droite in temoins.values():
        assert sp.simplify(gauche - droite) == 0
    gauche, _ = temoins["Commutativite"]
    assert sp.simplify(gauche - (sp.exp(T) - T - 1)) == 0


@pytest.mark.parametrize(
    ("rang", "domaine"),
    [
        (0, sp.Interval.Lopen(-1, 1)),
        (1, sp.Interval.Lopen(-2, 2)),
        (2, sp.Interval(-R(1, 3), R(1, 3))),
    ],
)
def test_pages_131_et_84_chaque_serie_converge_ou_on_la_demande(
    rang: int, domaine: sp.Set
) -> None:
    serie = syllabus.SERIES[rang]
    assert serie.domaine_de_convergence() == domaine
    assert serie.rayon_par_le_quotient() == domaine.end


def test_page_181_les_mineurs_que_les_notes_donnent() -> None:
    determinant, mineurs = syllabus.mineurs_du_wronskien()
    assert determinant == 20
    assert tuple(sp.simplify(m) for m in mineurs) == (-5, 4 / T, T**4)


def test_page_181_la_solution_de_l_equation_d_ordre_trois() -> None:
    particuliere, generale = syllabus.equation_d_ordre_trois()
    assert particuliere == T**4 / 12

    def residu(y: sp.Expr) -> sp.Expr:
        return sp.simplify(T**3 * y.diff(T, 3) - 6 * T * y.diff(T) + 12 * y - T**4)

    assert residu(particuliere) == 0
    assert residu(generale) == 0
    assert len(generale.free_symbols - {T}) == 3
    y = sp.Function("y")
    equation = T**3 * y(T).diff(T, 3) - 6 * T * y(T).diff(T) + 12 * y(T) - T**4
    reference = sp.dsolve(equation).rhs
    constantes = sorted(reference.free_symbols - {T}, key=str)
    assert reference.subs(dict.fromkeys(constantes, 0)) == particuliere


def test_page_90_l_integrale_terme_a_terme() -> None:
    """`int_0^(pi/2) cos(nx) / n^3 dx = sin(n pi / 2) / n^4`, terme par terme."""
    for n in range(1, 9):
        assert syllabus.terme_integre(n) == sp.sin(n * sp.pi / 2) / n**4


def test_page_90_la_serie_telescopique_ne_converge_pas_uniformement() -> None:
    partielle, limite, ecart = syllabus.serie_telescopique()
    x, n = syllabus.X, syllabus.N
    assert sp.simplify(partielle - (1 - x**n)) == 0
    assert limite.subs(x, R(1, 2)) == 1
    assert limite.subs(x, 1) == 0
    assert ecart == 1


def test_page_165_le_type_de_l_equation_depend_du_point() -> None:
    discriminant = syllabus.discriminant_de_l_edp()
    x, y = syllabus.X, syllabus.Y
    assert sp.expand(discriminant - (y**2 - x**2)) == 0
    assert discriminant.is_Mul
    assert discriminant.subs({x: 0, y: 1}) > 0
    assert discriminant.subs({x: 1, y: 0}) < 0
    assert discriminant.subs({x: 1, y: 1}) == 0


def test_le_jeu_de_macros() -> None:
    macros = JEUX_DE_MACROS["exercices-des-syllabus"]()
    assert all(nom.startswith("Syl") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["SylParticuliere"] == r"\frac{t^{4}}{12}"
    assert macros["SylSinus"] == r"\frac{\omega}{\omega^{2} + p^{2}}"
    assert "SylSerieTroisDomaine" in macros


def test_les_macros_parlent_la_langue_des_notes() -> None:
    """Le parametre des notes est `x_1` ; un bord exclu s'ecrit a la francaise."""
    macros = JEUX_DE_MACROS["exercices-des-syllabus"]()
    assert macros["SylSystemeSolution"] == (
        r"\left(x_{1},\; 11 x_{1} - 10,\; 8 x_{1} - 8\right)"
    )
    assert macros["SylSerieUnDomaine"] == r"\left]-1,\, 1\right]"
    assert macros["SylSerieTroisDomaine"] == (
        r"\left[- \frac{1}{3},\, \frac{1}{3}\right]"
    )
    assert r"\pi" not in macros["SylAssociativite"]


def test_un_bord_exclu_a_droite_tourne_aussi_son_crochet() -> None:
    """La serie de l'exemple 2 du syllabus converge sur `[-1, 1[`."""
    assert intervalle(sp.Interval.Ropen(-1, 1)) == r"\left[-1,\, 1\right["


def test_pages_54_et_59_les_deux_primitives_des_documents_de_travail() -> None:
    """Page 59, imprime : `1/2 racine(Q) + 3/4 argsh(4/racine(3) (x - 1/4))`."""
    x = syllabus.X
    juin, trinome = travail.primitives()
    assert gestes.ecart(juin.integrande, sp.cos(x) ** 2 / (1 - sp.tan(x) ** 2)) < 1e-12
    assert gestes.ecart(sp.diff(juin.primitive, x), juin.integrande) < 1e-12
    q = 4 * x**2 - 2 * x + 1
    assert gestes.ecart(trinome.integrande, (2 * x + 1) / sp.sqrt(q)) < 1e-12
    imprimee = sp.sqrt(q) / 2 + R(3, 4) * sp.asinh(4 / sp.sqrt(3) * (x - R(1, 4)))
    assert gestes.ecart(sp.diff(imprimee, x), trinome.integrande) < 1e-12
    assert sp.simplify(trinome.primitive - imprimee) == 0


def test_les_macros_des_deux_primitives() -> None:
    macros = JEUX_DE_MACROS["exercices-des-syllabus"]()
    assert {"SylJuinEtape", "SylJuinPrimitive", "SylTrinomeEtape"} <= set(macros)
    assert r"\operatorname{arsinh}" in macros["SylTrinomePrimitive"]
