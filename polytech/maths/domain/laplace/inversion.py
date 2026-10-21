"""Revenir a l'original par le theoreme des residus.

Deuxieme theoreme d'inversion : si `F` est holomorphe sur `C` sauf en un
nombre fini de singularites `z_1, ..., z_n`, et s'il existe `xi`, `M` et
`epsilon > 0` tels que `|z| > xi` entraine `|F(z)| <= M / |z|^epsilon`,
alors `F`, restreinte au demi-plan a droite de ses singularites, est la
transformee de Laplace de la fonction causale
`f(t) = somme_k Res_(z = z_k) (F(z) e^(z t))` pour `t > 0`.

Les deux hypotheses sont verifiees avant le calcul, et chacune leve si
elle manque. La decroissance n'est eprouvee que le long de l'axe reel.
ponytail: une borne uniforme en |z| demanderait une etude par direction ;
elle suffit pour les fractions rationnelles propres du cours.

Source : Mathematiques pour l'ingenieur III, planches de la transformee de
Laplace, planche 12.
"""

from __future__ import annotations

import sympy as sp
from sympy.calculus.singularities import singularities

from polytech.maths.domain.analyse_complexe.residus import residu
from polytech.maths.domain.symboles import IMAGE, TEMPS

_Z = sp.Symbol("z")
"""La variable complexe : `IMAGE` est positif, ses poles ne le sont pas."""


def original_par_les_residus(image: sp.Expr) -> sp.Expr:
    """`f(t)`, `t > 0`, somme des residus de `F(z) e^(z t)`."""
    f = sp.sympify(image).subs(IMAGE, _Z)
    if sp.limit(f, _Z, sp.oo) != 0:
        msg = f"{image} ne tend pas vers 0 a l'infini : le theoreme ne s'applique pas"
        raise ValueError(msg)
    poles = singularities(f, _Z, sp.S.Complexes)
    if not poles.is_finite_set:
        msg = f"singularites de {image} en nombre infini ou inconnu : {poles}"
        raise ValueError(msg)
    integrande = f * sp.exp(_Z * TEMPS)
    somme = sum((residu(integrande, _Z, p) for p in poles), sp.Integer(0))
    return sp.simplify(sp.expand_complex(somme))
