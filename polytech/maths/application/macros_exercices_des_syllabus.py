r"""Macros des exercices poses au fil des deux syllabus de theorie, prefixe `Syl`."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import calculs_des_notes_manuscrites as manuscrites
from polytech.maths.application import exercices_des_syllabus as syllabus
from polytech.maths.application import primitives_des_documents_de_travail as travail
from polytech.maths.application.formatage import formule
from polytech.maths.application.recueil_algebre import liste, operations
from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.series.entieres import INDICE

ORDINAUX = ("Un", "Deux", "Trois")
PREMIERE = sp.Symbol("x_1")


def intervalle(ensemble: sp.Interval) -> str:
    r"""`\left]-1,\, 1\right]` : crochet tourne vers l'exterieur a un bord exclu."""
    gauche = "]" if ensemble.left_open else "["
    droite = "[" if ensemble.right_open else "]"
    bornes = rf"{formule(ensemble.start)},\, {formule(ensemble.end)}"
    return rf"\left{gauche}{bornes}\right{droite}"


def _en_x_un(solution: tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    """La meme solution, parametree par `x_1` comme dans les notes."""
    (libre,) = sp.Matrix(list(solution)).free_symbols
    (valeur,) = sp.solve(solution[0] - PREMIERE, libre)
    return tuple(sp.expand(terme.subs(libre, valeur)) for terme in solution)


def _systeme() -> dict[str, str]:
    augmentee = syllabus.MODIFIE_13
    echelonnee = echelonner(augmentee, colonnes=augmentee.cols - 1)
    _, solution = syllabus.systeme_modifie()
    return {
        "SylSystemeOperations": operations(echelonnee.operations),
        "SylSystemeEchelonnee": formule(echelonnee.matrice),
        "SylSystemeSolution": rf"\left({liste(_en_x_un(solution))}\right)",
    }


def _laplace() -> dict[str, str]:
    par_l_integrale, _ = syllabus.laplace_du_sinus()
    temoins = syllabus.convolutions()
    associative = sp.simplify(temoins["Associativite"][0])
    return {
        "SylSinus": formule(par_l_integrale),
        "SylCommutativite": formule(temoins["Commutativite"][0]),
        "SylDistributivite": formule(temoins["Distributivite"][0]),
        "SylAssociativite": formule(sp.expand(sp.expand_trig(associative))),
    }


def _series() -> dict[str, str]:
    macros = {}
    for ordinal, serie in zip(ORDINAUX, syllabus.SERIES, strict=True):
        terme = serie.terme * syllabus.X**INDICE
        macros[f"SylSerie{ordinal}"] = formule(terme)
        macros[f"SylSerie{ordinal}Rayon"] = formule(serie.rayon_par_le_quotient())
        macros[f"SylSerie{ordinal}Domaine"] = intervalle(serie.domaine_de_convergence())
    partielle, _, _ = syllabus.serie_telescopique()
    return macros | {"SylPartielle": formule(partielle)}


def _equations() -> dict[str, str]:
    determinant, mineurs = syllabus.mineurs_du_wronskien()
    particuliere, generale = syllabus.equation_d_ordre_trois()
    return {
        "SylWronskien": formule(determinant),
        "SylMineurs": liste(mineurs),
        "SylParticuliere": formule(particuliere),
        "SylGenerale": formule(generale),
        "SylDiscriminant": formule(syllabus.discriminant_de_l_edp()),
    }


def _primitives() -> dict[str, str]:
    juin, trinome = travail.primitives()
    return {
        "SylJuinEtape": formule(juin.etapes[0]),
        "SylJuinPrimitive": formule(juin.primitive),
        "SylTrinomeEtape": formule(trinome.etapes[0]),
        "SylTrinomePrimitive": formule(trinome.primitive),
    }


def _manuscrites() -> dict[str, str]:
    _, borde = manuscrites.determinants_de_la_page_14()
    parties = manuscrites.parties_de_la_page_44()
    arc, fraction = manuscrites.fraction_de_la_page_47()
    return {
        "SylBorde": formule(borde),
        "SylPartiesA": formule(parties.etapes[0]),
        "SylPartiesB": formule(parties.etapes[1]),
        "SylPartiesPrimitive": formule(sp.factor(parties.primitive)),
        "SylArc": formule(arc.primitive),
        "SylFractionDecomposee": formule(sp.apart(manuscrites.FRACTION_47)),
        "SylFraction": formule(fraction),
        "SylReste": formule(manuscrites.reste_du_numerateur()),
        "SylCoefficients": liste(manuscrites.coefficients_de_la_page_46()),
    }


def macros_des_exercices_des_syllabus() -> dict[str, str]:
    """Toutes les formules que le corrige compose."""
    calculs = _systeme() | _laplace() | _series() | _equations()
    return calculs | _primitives() | _manuscrites()
