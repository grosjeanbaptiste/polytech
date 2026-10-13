"""Les signaux causaux du cours : echelon, creneau, triangle, et Dirac.

Mathematiques appliquees, 3.1 : l'**echelon** `H(t)` vaut 1 pour `t >= 0`
et 0 avant ; une fonction est **causale** si elle est nulle pour `t < 0`,
et l'on rend `f` causale en la multipliant par `H`.

Les exemples 4 a 6 du 3.4 (page 57) les combinent a la translation
`tau_h f(t) = f(t - h)` :

* l'echelon retarde `tau_h H` ;
* le **creneau** `n H - n tau_(1/n) H`, de hauteur `n` sur `[0, 1/n[` et
  d'aire 1, dont la limite est l'impulsion de Dirac (3.8.1) ;
* le **triangle** `t - 2 tau_1 t + tau_2 t`, qui monte de 0 a 1 puis
  redescend a 0 en `t = 2`.

La **mesure de Dirac** (3.8.2) d'un ensemble `A` vaut 1 si `A` contient
l'origine, 0 sinon.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import TEMPS as T


def echelon(retard: object = 0) -> sp.Expr:
    """`tau_h H`, l'echelon retarde de `h` ; `H` lui-meme pour `h = 0`."""
    return sp.Heaviside(T - sp.sympify(retard), 1)


def rendre_causale(fonction: sp.Expr) -> sp.Expr:
    """`H(t) f(t)` : nulle avant l'origine, egale a `f` apres."""
    return echelon() * fonction


def creneau(n: object) -> sp.Expr:
    """`n H - n tau_(1/n) H` : la hauteur `n` sur une largeur `1/n`."""
    hauteur = sp.sympify(n)
    if not hauteur.is_positive:
        msg = f"le creneau demande un n strictement positif : {hauteur}"
        raise ValueError(msg)
    return hauteur * echelon() - hauteur * echelon(1 / hauteur)


def _rampe(retard: int) -> sp.Expr:
    """`tau_h t`, la rampe causale retardee de `h`."""
    return (T - retard) * echelon(retard)


def triangle() -> sp.Expr:
    """`t - 2 tau_1 t + tau_2 t`."""
    return _rampe(0) - 2 * _rampe(1) + _rampe(2)


def mesure_de_dirac(ensemble: sp.Set) -> int:
    """`delta(A)` : 1 si `A` contient l'origine, 0 sinon."""
    return 1 if ensemble.contains(0) == sp.true else 0
