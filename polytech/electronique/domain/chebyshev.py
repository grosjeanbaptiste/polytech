"""La table des elements d'un prototype de Chebyshev, calculee et non lue.

Les valeurs `g_k` d'un filtre de Chebyshev se lisent d'ordinaire dans une
table. Les recopier, c'est risquer un chiffre -- et c'est exactement ce qui
est arrive au corrige de janvier 2022-2023, qui ecrit `1,1979` la ou la
table donne `1,1474`.

Les formules fermees existent ; les poser ici rend la table verifiable au
lieu d'etre crue.
"""

from __future__ import annotations

import math

CONSTANTE_DE_RIPPLE = 17.37
"""`40 / ln(10)`, le facteur qui convertit l'ondulation en decibels."""


def elements_de_chebyshev(
    ondulation_en_decibels: float, ordre: int
) -> tuple[float, ...]:
    """Rend les `g_k` du prototype, dans l'ordre, pour terminaisons egales."""
    if ondulation_en_decibels <= 0:
        msg = f"ondulation non positive : {ondulation_en_decibels}"
        raise ValueError(msg)
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)
    beta = math.log(1 / math.tanh(ondulation_en_decibels / CONSTANTE_DE_RIPPLE))
    gamma = math.sinh(beta / (2 * ordre))
    a = [math.sin((2 * k - 1) * math.pi / (2 * ordre)) for k in range(1, ordre + 1)]
    b = [gamma**2 + math.sin(k * math.pi / ordre) ** 2 for k in range(1, ordre + 1)]
    elements = [2 * a[0] / gamma]
    for k in range(1, ordre):
        elements.append(4 * a[k - 1] * a[k] / (b[k - 1] * elements[k - 1]))
    return tuple(elements)


def charge_de_chebyshev(ondulation_en_decibels: float, ordre: int) -> float:
    """`g_(N+1)` : 1 aux ordres impairs, `coth^2(beta / 4)` aux ordres pairs.

    La planche 69 imprime `coth(beta / 4)`, sans le carre ; sa table porte
    1,3554 pour 0,1 dB, qui est bien le carre.
    """
    if ondulation_en_decibels <= 0:
        msg = f"ondulation non positive : {ondulation_en_decibels}"
        raise ValueError(msg)
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)
    if ordre % 2:
        return 1.0
    beta = math.log(1 / math.tanh(ondulation_en_decibels / CONSTANTE_DE_RIPPLE))
    return 1 / math.tanh(beta / 4) ** 2
