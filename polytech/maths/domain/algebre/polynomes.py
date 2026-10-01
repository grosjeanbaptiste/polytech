"""Division polynomiale et relations de Viete -- chapitre IV du syllabus.

IV.1 Division polynomiale, IV.3 Utilisation du lien entre racines et
coefficients. Le syllabus donne la division sous deux formes : « quelque
soit la forme du diviseur, on peut effectuer la division sous forme d'un
calcul ecrit », et « si le diviseur est du type `x - a`, on peut utiliser
la methode d'Horner ».

Horner n'est pas qu'un raccourci : son dernier nombre **est** le reste,
donc la valeur du polynome en `a`. Le theoreme du reste n'est donc pas un
resultat separe mais une lecture du meme calcul -- et les tests le
verifient par l'evaluation directe, qui ne divise rien.

Les relations de Viete valent pour elles-memes, et le syllabus choisit
bien son exemple : un degre cinq, dont les racines n'ont pas
d'expression par radicaux en general. Les fonctions symetriques s'en
obtiennent pourtant depuis les seuls coefficients -- c'est tout
l'interet, et c'est ce que « sans calculer les racines de `p(x)` »
souligne.

Convention de ce module : les coefficients sont donnes du degre le plus
eleve au terme constant, comme on les ecrit.
"""

from __future__ import annotations

import math

TOLERANCE = 1e-9
"""En dessous, un coefficient ou un reste est tenu pour nul."""


def degre(coefficients: tuple[float, ...]) -> int:
    """Le degre, les coefficients de tete nuls etant ignores."""
    if not coefficients:
        msg = "polynome sans coefficient : il n'y a rien a diviser"
        raise ValueError(msg)
    for rang, valeur in enumerate(coefficients):
        if abs(valeur) > TOLERANCE:
            return len(coefficients) - 1 - rang
    return 0


def evalue(coefficients: tuple[float, ...], x: float) -> float:
    """`P(x)`, par le schema de Horner -- le meme calcul que la division."""
    degre(coefficients)
    accumule = 0.0
    for coefficient in coefficients:
        accumule = accumule * x + coefficient
    return accumule


def division_de_horner(
    coefficients: tuple[float, ...],
    racine: float,
) -> tuple[tuple[float, ...], float]:
    """Le quotient et le reste de la division par `x - a`.

    Le procede est celui que le syllabus deroule colonne par colonne :
    « on multiplie le premier coefficient par la racine du diviseur, on
    ecrit le resultat sous le deuxieme coefficient et on effectue la
    somme des deux chiffres ».
    """
    degre(coefficients)
    quotient: list[float] = []
    accumule = 0.0
    for coefficient in coefficients:
        quotient.append(accumule)
        accumule = accumule * racine + coefficient
    return tuple(quotient[1:]), accumule


def reste_en(coefficients: tuple[float, ...], racine: float) -> float:
    """Le reste de la division par `x - a`, c'est-a-dire `P(a)`."""
    return division_de_horner(coefficients, racine)[1]


def factorise_par(
    coefficients: tuple[float, ...],
    racine: float,
) -> tuple[float, ...]:
    """Le quotient par `x - a`, refuse si `a` n'est pas racine."""
    quotient, reste = division_de_horner(coefficients, racine)
    if abs(reste) > TOLERANCE:
        msg = f"{racine} n'est pas une racine : le reste vaut {reste:.6g}"
        raise ValueError(msg)
    return quotient


def division_euclidienne(
    dividende: tuple[float, ...],
    diviseur: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`P = Q D + R` avec `deg R < deg D`, pour un diviseur quelconque."""
    if all(abs(c) <= TOLERANCE for c in diviseur):
        msg = "diviseur nul : la division n'existe pas"
        raise ValueError(msg)
    reste = list(dividende)
    tete = len(diviseur) - 1 - (len(diviseur) - 1 - degre(diviseur))
    quotient: list[float] = []
    while len(reste) - 1 >= tete and len(reste) >= len(diviseur):
        facteur = reste[0] / diviseur[0]
        quotient.append(facteur)
        for i, coefficient in enumerate(diviseur):
            reste[i] -= facteur * coefficient
        reste.pop(0)
    return tuple(quotient) or (0.0,), tuple(reste) or (0.0,)


def derivee_formelle(coefficients: tuple[float, ...]) -> tuple[float, ...]:
    """`P'`, terme a terme. Sert a detecter les racines multiples."""
    hauteur = degre(coefficients)
    if hauteur == 0:
        return (0.0,)
    return tuple(
        coefficient * (len(coefficients) - 1 - rang)
        for rang, coefficient in enumerate(coefficients[:-1])
    )


def est_racine_multiple(coefficients: tuple[float, ...], racine: float) -> bool:
    """Vrai si `a` annule `P` **et** `P'` : le critere de multiplicite."""
    return math.isclose(
        evalue(coefficients, racine),
        0.0,
        abs_tol=TOLERANCE,
    ) and math.isclose(
        evalue(derivee_formelle(coefficients), racine),
        0.0,
        abs_tol=TOLERANCE,
    )


def pgcd(
    premier: tuple[float, ...],
    second: tuple[float, ...],
) -> tuple[float, ...]:
    """Le plus grand commun diviseur, par l'algorithme d'Euclide."""
    a, b = tuple(premier), tuple(second)
    while any(abs(c) > TOLERANCE for c in b):
        a, b = b, division_euclidienne(a, b)[1]
    return tuple(c / a[0] for c in a) if abs(a[0]) > TOLERANCE else a


def fonctions_symetriques_des_racines(
    coefficients: tuple[float, ...],
) -> tuple[float, ...]:
    """Les `e_k` des relations de Viete, indexes de 0 a `n`.

    `e_k = (-1)^k a_{n-k} / a_n` : la somme des racines, la somme de
    leurs produits deux a deux, et ainsi de suite -- obtenues « sans
    calculer les racines », ce qui est tout leur interet sur un degre
    cinq.
    """
    hauteur = degre(coefficients)
    dominant = coefficients[0]
    if abs(dominant) <= TOLERANCE:
        msg = "coefficient dominant nul : les relations de Viete n'y valent pas"
        raise ValueError(msg)
    return tuple((-1) ** k * coefficients[k] / dominant for k in range(hauteur + 1))
