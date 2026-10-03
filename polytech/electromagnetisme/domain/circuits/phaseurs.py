"""Du domaine de Laplace au regime sinusoidal etabli.

Le resolveur operationnel rend `X(p) = H(p) E` pour une source constante
`E`. Le regime sinusoidal en est la restriction a `p = j omega` : la
source s'y lit comme un phaseur **efficace**, et la reponse aussi. Il n'y
a donc pas de second resolveur a tenir d'accord avec le premier.

Ce passage n'a de sens qu'en regime etabli, c'est-a-dire une fois le
transitoire eteint. Les conditions initiales des elements a memoire n'y
entrent pas : elles laissent nulles celles du reseau a resoudre.

La puissance complexe est `S = U I*`, en convention recepteur : `P > 0`
pour ce qui absorbe, `Q > 0` pour ce qui est inductif. Conjuguer la
tension plutot que le courant changerait le signe du reactif, et c'est la
confusion que la definition ferme.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE


def en_regime_sinusoidal(transformee: sp.Expr, pulsation: sp.Expr) -> sp.Expr:
    """`X(j omega)` : le phaseur de la reponse ; leve si `omega <= 0`."""
    if sp.sympify(pulsation).is_positive is not True:
        msg = f"pulsation non strictement positive : {pulsation}"
        raise ValueError(msg)
    return sp.sympify(transformee).subs(VARIABLE, sp.I * pulsation)


def puissance_complexe(tension: sp.Expr, courant: sp.Expr) -> sp.Expr:
    """`S = U I*`, phaseurs efficaces, convention recepteur."""
    return sp.expand(tension * sp.conjugate(courant))


def valeur_instantanee(phaseur: sp.Expr, pulsation: sp.Expr) -> sp.Expr:
    """`x(t) = racine(2) Re(X e^{j omega t})` pour un phaseur efficace `X`."""
    tournant = sp.sqrt(2) * phaseur * sp.exp(sp.I * pulsation * TEMPS)
    return sp.simplify(sp.re(sp.expand_complex(tournant)))
