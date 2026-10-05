"""Les fonctions recursives du cours, et leurs equivalents iteratifs.

Le cours (partie 1, planches 95, 120 et 121) : « une fonction est dite
recursive lorsqu'elle s'appelle elle-meme ». Elle a toujours un cas
d'arret, sans appel, et un cas general qui se ramene a un probleme plus
petit :

* `n! = n (n - 1)!`, avec `0! = 1` ;
* `pgcd(a, b) = pgcd(b, a % b)`, qui s'arrete quand le reste est nul ;
* la suite de Fibonacci, `u_n = u_(n-1) + u_(n-2)`, `u_0 = u_1 = 1` ;
* la puissance entiere, `a^n = a . a^(n-1)`, `a^0 = 1` ;
* les tours de Hanoi : deplacer `n - 1` disques sur le piquet
  intermediaire, puis le plus grand, puis les `n - 1` par-dessus -- soit
  `2^n - 1` deplacements.
"""

from __future__ import annotations

from polytech.algorithmique.domain.types_cpp import modulo


def factorielle(n: int) -> int:
    """`n!`, et -1 pour code d'erreur si `n` est negatif, comme a la planche 120."""
    if n < 0:
        return -1
    return 1 if n == 0 else n * factorielle(n - 1)


def pgcd(a: int, b: int) -> int:
    """L'algorithme d'Euclide en boucle, tel que la planche 95 l'ecrit."""
    while b > 0:
        a, b = b, modulo(a, b)
    return a


def pgcd_recursif(a: int, b: int) -> int:
    """La version recursive de la planche 120 : `r = a % b`, puis `pgcd(b, r)`."""
    reste = modulo(a, b)
    return b if reste == 0 else pgcd_recursif(b, reste)


def fibonacci(n: int) -> int:
    """`u_n`, avec `u_0 = u_1 = 1`."""
    if n < 0:
        msg = f"rang negatif : {n}"
        raise ValueError(msg)
    return 1 if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)  # noqa: PLR2004 -- les deux premiers termes de la suite


def puissance(a: float, n: int) -> float:
    """`a^n` pour `n` entier positif ou nul."""
    if n < 0:
        msg = f"exposant negatif : {n}"
        raise ValueError(msg)
    return 1 if n == 0 else a * puissance(a, n - 1)


def deplacements_de_hanoi(
    disques: int, depart: str, arrivee: str, intermediaire: str
) -> tuple[tuple[str, str], ...]:
    """Les deplacements, dans l'ordre, qui portent la tour de `depart` a `arrivee`."""
    if disques == 0:
        return ()
    return (
        *deplacements_de_hanoi(disques - 1, depart, intermediaire, arrivee),
        (depart, arrivee),
        *deplacements_de_hanoi(disques - 1, intermediaire, arrivee, depart),
    )
