"""Recueil d'algebre, chapitre VIII, exercices 1 a 9 : lineaire, noyau, inverse.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, section VIII.4
(pages 102 et 103).
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.familles_de_vecteurs import relation_de_dependance
from polytech.maths.domain.algebre.linearite import (
    existe_une_application_lineaire,
    respecte_les_combinaisons,
)

M = sp.Matrix
X: Final = sp.Symbol("x")
CONSTANTE: Final = sp.Symbol("C")
A: Final = sp.Symbol("a")
"""Le reel de l'exercice 8 c), dans `D - a I`."""
REELS: Final = sp.symbols("alpha beta", real=True)
_REGLE = ApplicationLineaire.depuis_la_regle
APPLICATIONS: Final = {
    "5a": _REGLE(lambda _a, b, _c, d: (b, d), 4),
    "9a": _REGLE(lambda x, y: (x + y, x - y), 2),
    "9b": _REGLE(lambda x, y: (2 * x + y, 3 * x - 5 * y), 2),
    "9c": _REGLE(lambda x, y, z: (x - y, x + z, x + y + 2 * z), 3),
    "9d": _REGLE(lambda x, y, z: (2 * x - y + z, x + y, 3 * x + y + z), 3),
}
DONNEES_2: Final = {
    "a": (((3, 1), (-1, 0)), ((1, 2), (1, 1))),
    "b": (((4, 1), (1, 1)), ((1, 1), (3, -2))),
    "c": (((1, 1), (-1, 1)), ((2, 1), (6, 3))),
}
"""Lettre : deux vecteurs, puis leurs images."""
ANTECEDENTS_6: Final = ((1, 1, 3), (1, 1, 2), (1, 1, 1))
IMAGES_6: Final = ((1, 1, 1, 1), (1, 1, 1, 0), (1, 1, 0, 0))
ANTECEDENTS_7: Final = ((1, 1, 1), (1, 1, 0), (1, 0, 0))
IMAGES_7: Final = ((1, 1, 1, 3), (1, 1, 1, 2), (1, 1, 1, 1))


def _generiques(nom: str, taille: int) -> tuple[sp.Matrix, sp.Matrix]:
    return M(sp.symbols(f"{nom}1:{taille + 1}")), M(sp.symbols(f"v1:{taille + 1}"))


def _polynomes() -> tuple[sp.Expr, sp.Expr]:
    p = sum(c * X**k for k, c in enumerate(sp.symbols("p0:4")))
    q = sum(c * X**k for k, c in enumerate(sp.symbols("q0:4")))
    return p, q


def exercice_1() -> dict[str, bool]:
    """Trois regles : lineaire si elle respecte toute combinaison."""
    plan, espace = _generiques("u", 2), _generiques("u", 3)
    return {
        "a": respecte_les_combinaisons(lambda v: M([2 * v[0] + v[1], v[1]]), *plan),
        "b": respecte_les_combinaisons(lambda v: v + M([0, -1, 0]), *espace),
        "c": respecte_les_combinaisons(lambda v: M([v[0] * v[1]]), *plan),
    }


def carre_et_son_image() -> tuple[tuple[tuple[sp.Expr, ...], ...], ...]:
    """Les sommets du carre unite, et leurs images par l'application du 1 a)."""
    f = _REGLE(lambda x, y: (2 * x + y, y), 2)
    carre = ((0, 0), (1, 0), (1, 1), (0, 1))
    return carre, tuple(f.image_de(sommet) for sommet in carre)


def determinants_du_9() -> dict[str, sp.Expr]:
    """Le determinant de chaque matrice : non nul, l'application est inversible."""
    return {cle[1]: APPLICATIONS[cle].matrice.det() for cle in sorted(APPLICATIONS)[1:]}


def exercice_2() -> dict[str, tuple[tuple[sp.Expr, ...], ...]]:
    """`f(1, 0)` et `f(0, 1)`, les images de deux vecteurs etant donnees."""
    resultat: dict[str, tuple[tuple[sp.Expr, ...], ...]] = {}
    for lettre, (antecedents, images) in DONNEES_2.items():
        f = ApplicationLineaire.depuis_les_images(antecedents, images)
        resultat[lettre] = (f.image_de((1, 0)), f.image_de((0, 1)))
    return resultat


def exercice_3() -> dict[str, bool]:
    """Quatre regles sur les polynomes ; `f2` ajoute ici le polynome fixe 1."""
    p, q = _polynomes()
    regles = {
        "f1": lambda u: u * (X**2 + 1),
        "f2": lambda u: u + 1,
        "f3": lambda u: u.subs(X, 1) + (X - 1) * sp.diff(u, X).subs(X, 1),
        "f4": lambda u: u.subs(X, X + 2),
    }
    return {nom: respecte_les_combinaisons(f, p, q) for nom, f in regles.items()}


def exercice_4() -> dict[str, bool]:
    """L'evaluation en un point, et la conjugaison sur `C` vu comme espace reel."""
    f, g = sp.Function("f")(X), sp.Function("g")(X)
    a, b, c, d = sp.symbols("a b c d", real=True)
    point = sp.Symbol("a0", real=True)
    return {
        "a": respecte_les_combinaisons(lambda h: h.subs(X, point), f, g),
        "b": respecte_les_combinaisons(sp.conjugate, a + sp.I * b, c + sp.I * d, REELS),
    }


def partie_impaire(fonction: sp.Expr) -> sp.Expr:
    """5 c) : `g(x) = (h(x) - h(-x)) / 2`."""
    return sp.simplify((fonction - fonction.subs(X, -X)) / 2)


def exercice_6() -> tuple[bool, tuple[sp.Expr, ...]]:
    """Existe-t-elle, et la relation entre les antecedents qui en decide."""
    existe = existe_une_application_lineaire(ANTECEDENTS_6, IMAGES_6)
    return existe, relation_de_dependance(ANTECEDENTS_6)


def exercice_7() -> tuple[sp.Expr, ...]:
    """L'image de `(x, y, z)`."""
    f = ApplicationLineaire.depuis_les_images(ANTECEDENTS_7, IMAGES_7)
    return f.image_de(sp.symbols("x y z"))


def noyau_de_la_derivation_moins(valeur: sp.Expr) -> sp.Expr:
    """Exercice 8 : les fonctions telles que `y' = a y`."""
    y = sp.Function("y")
    solution = sp.dsolve(sp.diff(y(X), X) - valeur * y(X), y(X)).rhs
    return solution.subs(sp.Symbol("C1"), CONSTANTE)


def exercice_9() -> dict[str, tuple[sp.Expr, ...] | None]:
    """L'inverse, applique a `(a, b)` ou `(a, b, c)` ; rien s'il n'existe pas."""
    resultat: dict[str, tuple[sp.Expr, ...] | None] = {}
    for cle in ("9a", "9b", "9c", "9d"):
        f = APPLICATIONS[cle]
        cible = sp.symbols("a b c")[: f.matrice.rows]
        inversible = f.est_injective() and f.est_surjective()
        resultat[cle[1]] = f.inverse().image_de(cible) if inversible else None
    return resultat
