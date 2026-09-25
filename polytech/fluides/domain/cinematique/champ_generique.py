"""Champ de vitesse *generique*, support des definitions du cours.

En appliquant les memes operateurs a un champ abstrait ``(u, v)``, on obtient
les definitions telles que le cours les enonce -- divergence, vorticite,
derivee particulaire. Le document les cite donc sans les recopier, et elles
ne peuvent pas diverger de ce que la bibliotheque calcule reellement.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.symboles import T, X, Y

U_GENERIQUE = sp.Function("u")(X, Y, T)
V_GENERIQUE = sp.Function("v")(X, Y, T)

CHAMP_GENERIQUE = ChampDeVitesse(u=U_GENERIQUE, v=V_GENERIQUE)
"""Champ abstrait : ses operateurs rendent les definitions du cours III."""
