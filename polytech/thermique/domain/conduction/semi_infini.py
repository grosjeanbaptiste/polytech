"""Le solide semi-infini : les phenomenes superficiels.

Syllabus de thermique, tome 1, premiere partie, section 4.3. Quand le
gradient ne concerne qu'une couche mince devant les dimensions du solide
-- trempe, ondes thermiques dans le sol --, on le tient pour semi-infini.
Trois sollicitations de sa surface ont une solution fermee, toutes
portees par la fonction d'erreur :

* un echelon de temperature, `t = t_0 + (t_f - t_0) ferc(x / (2 racine(a
  tau)))`, d'ou un flux penetrant en `1 / racine(tau)` ;
* le contact d'un fluide, avec un coefficient de transmission `K` ;
* un echelon de flux `phi_0`.

Deux solides mis en contact prennent a l'interface une temperature
constante, plus proche de celle du corps dont `lambda rho c` est le plus
grand. Et sous une sollicitation periodique, l'oscillation s'amortit en
profondeur comme `exp(-x racine(pi / (a tau_0)))`.
"""

from __future__ import annotations

import math

import sympy as sp

type Grandeur = sp.Expr | float
type Corps = tuple[float, float]
"""La temperature initiale d'un solide et son produit `lambda rho c`."""


def _reduit(abscisse: Grandeur, duree: Grandeur, diffusivite: Grandeur) -> sp.Expr:
    return abscisse / (2 * sp.sqrt(diffusivite * duree))


def sous_un_echelon_de_surface(
    abscisse: Grandeur,
    duree: Grandeur,
    diffusivite: Grandeur,
    *,
    initiale: Grandeur,
    surface: Grandeur,
) -> sp.Expr:
    """`t = t_0 + (t_f - t_0) ferc(x / (2 racine(a tau)))`."""
    return initiale + (surface - initiale) * sp.erfc(
        _reduit(abscisse, duree, diffusivite)
    )


def sous_un_fluide(  # noqa: PLR0913 -- les cinq grandeurs du cas 2, plus `K / lambda`
    abscisse: Grandeur,
    duree: Grandeur,
    diffusivite: Grandeur,
    echange_reduit: Grandeur,
    *,
    initiale: Grandeur,
    fluide: Grandeur,
) -> sp.Expr:
    """Le cas 2 de la page 76, `echange_reduit` etant `K / lambda`."""
    z = _reduit(abscisse, duree, diffusivite)
    avance = echange_reduit * sp.sqrt(diffusivite * duree)
    exposant = echange_reduit * abscisse + avance**2
    reduit = sp.erfc(z) - sp.exp(exposant) * sp.erfc(z + avance)
    return initiale + (fluide - initiale) * reduit


def sous_un_echelon_de_flux(
    abscisse: Grandeur,
    duree: Grandeur,
    diffusivite: Grandeur,
    flux_reduit: Grandeur,
    *,
    initiale: Grandeur,
) -> sp.Expr:
    """Le cas 3 de la page 76, `flux_reduit` etant `phi_0 / lambda`."""
    z = _reduit(abscisse, duree, diffusivite)
    gauss = sp.sqrt(diffusivite * duree / sp.pi) * sp.exp(-(z**2))
    return initiale + 2 * flux_reduit * (gauss - abscisse / 2 * sp.erfc(z))


def temperature_de_contact(premier: Corps, second: Corps) -> float:
    """`t_m`, telle que `(t_2 - t_m) / (t_m - t_1) = racine(e_1) / racine(e_2)`."""
    poids_un, poids_deux = math.sqrt(premier[1]), math.sqrt(second[1])
    return (premier[0] * poids_un + second[0] * poids_deux) / (poids_un + poids_deux)


def _nombre_d_onde(diffusivite: float, periode: float) -> float:
    if periode <= 0:
        msg = f"periode nulle ou negative : {periode} s"
        raise ValueError(msg)
    return math.sqrt(math.pi / (diffusivite * periode))


def amortissement_en_profondeur(
    profondeur: float, diffusivite: float, periode: float
) -> float:
    """`exp(-x racine(pi / (a tau_0)))`."""
    return math.exp(-profondeur * _nombre_d_onde(diffusivite, periode))


def attenuation_a_la_surface(
    longueur: float, diffusivite: float, periode: float
) -> float:
    """Le rapport d'amplitude paroi / fluide, `longueur` etant `lambda / K`.

    C'est le module de `1 / (1 + (lambda / K) racine(j omega / a))`.
    """
    reduit = longueur * math.sqrt(2) * _nombre_d_onde(diffusivite, periode)
    return 1 / math.sqrt(1 + math.sqrt(2) * reduit + reduit**2)
