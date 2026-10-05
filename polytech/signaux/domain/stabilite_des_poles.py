"""La stabilite d'un systeme, lue sur ses poles.

Syllabus de signaux et systemes, section 4.1.8. Une excitation breve
laisse derriere elle la reponse impulsionnelle : c'est elle qui decide.

* Strictement stable : `h(t)` tend vers zero, les poles sont tous dans
  le demi-plan de gauche, axe imaginaire exclu.
* Stable au sens large : `h(t)` reste bornee, des poles sont sur l'axe
  imaginaire mais ils y sont simples.
* Instable sinon : un pole a droite, ou un pole multiple sur l'axe, dont
  l'exponentielle est multipliee par une puissance de `t`.
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P


def nature_de_la_stabilite(transmittance: sp.Expr) -> str:
    """« strictement stable », « stable au sens large » ou « instable »."""
    denominateur = sp.fraction(sp.cancel(transmittance))[1]
    poles = sp.Poly(denominateur, P).all_roots()
    reelles = [sp.re(pole) for pole in poles]
    if any(partie > 0 for partie in reelles):
        return "instable"
    sur_l_axe = [
        pole for pole, partie in zip(poles, reelles, strict=True) if partie == 0
    ]
    if not sur_l_axe:
        return "strictement stable"
    simples = len(set(sur_l_axe)) == len(sur_l_axe)
    return "stable au sens large" if simples else "instable"
