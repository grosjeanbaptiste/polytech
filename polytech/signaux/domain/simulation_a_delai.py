"""La reponse indicielle d'une boucle fermee dont la chaine porte un delai.

Planches « Controleur » du cours de signaux et systemes. Avec un delai,
la boucle fermee n'est plus une fraction rationnelle : aucune table de
transformees ne donne sa reponse. On la simule. La partie rationnelle de
la boucle ouverte, mise sous forme d'etat et discretisee exactement,
recoit l'ecart ; sa sortie, retardee, est celle de la boucle :

`e(t) = 1 - y(t)`, `y(t) = z(t - tau)`.

Le retour est unitaire. Le pas est celui des instants demandes, que l'on
prend petit devant le delai et les constantes de temps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from scipy import signal

from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _coefficients(expression: sp.Expr) -> list[float]:
    return [float(c) for c in sp.Poly(expression, P).all_coeffs()]


def indicielle_en_boucle_fermee(
    rationnelle: sp.Expr, delai: float, instants: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`y(t)` pour une consigne en echelon unite et un retour unitaire."""
    if delai < 0:
        msg = f"delai negatif : {delai} s -- la boucle anticiperait"
        raise ValueError(msg)
    numerateur, denominateur = sp.fraction(sp.cancel(rationnelle))
    pas = float(instants[1] - instants[0])
    etat = signal.tf2ss(_coefficients(numerateur), _coefficients(denominateur))
    a, b, c, _, _ = signal.cont2discrete(etat, pas, method="zoh")
    retard = round(delai / pas)
    x = np.zeros(a.shape[0])
    sorties = np.zeros(len(instants) + retard)
    for rang in range(len(instants)):
        sorties[rang + retard] = float((c @ x)[0])
        x = a @ x + b[:, 0] * (1 - sorties[rang])
    return np.asarray(sorties[: len(instants)], dtype=float)
