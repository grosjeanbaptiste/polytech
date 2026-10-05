"""Les noeuds de frontiere d'un reseau de differences finies.

Syllabus de thermique, tome 1, premiere partie, section 5.2.2. Pour
garder a la frontiere l'erreur en `Delta x^2` des noeuds interieurs, on
prolonge le reseau d'un noeud fictif, de l'autre cote de la surface, et
l'on tire sa temperature de la condition aux limites ecrite en
difference centree.

* Flux impose entrant `phi_0` : `t* = t_1 + 2 Delta x phi_0 / lambda`.
* Echange avec un fluide : le flux entrant vaut `K (t_f - t_0)`.

Reportee dans l'equation nodale, la premiere donne
`4 t_0 - 2 t_1 - t_2 - t_4 = q Delta x^2 / lambda + 2 Delta x phi_0 /
lambda` : un flux entrant echauffe le noeud.
"""

from __future__ import annotations


def noeud_fictif_de_neumann(
    interieur: float, pas: float, conductivite: float, *, flux: float
) -> float:
    """`t* = t_1 + 2 Delta x phi_0 / lambda`."""
    if conductivite <= 0:
        msg = f"conductivite nulle ou negative : {conductivite}"
        raise ValueError(msg)
    return interieur + 2 * pas * flux / conductivite


def noeud_fictif_de_fourier(  # noqa: PLR0913 -- le noeud, le reseau, puis le fluide
    interieur: float,
    pas: float,
    conductivite: float,
    *,
    echange: float,
    fluide: float,
    surface: float,
) -> float:
    """`t* = t_1 + (2 Delta x / lambda) K (t_f - t_0)`."""
    entrant = echange * (fluide - surface)
    return noeud_fictif_de_neumann(interieur, pas, conductivite, flux=entrant)


def temperature_de_frontiere(  # noqa: PLR0913 -- trois voisins, le reseau, deux sources
    interieur: float,
    lateraux: tuple[float, float],
    pas: float,
    conductivite: float,
    *,
    flux: float,
    generation: float = 0.0,
) -> float:
    """`t_0`, d'apres l'equation du noeud de frontiere sous un flux impose."""
    fictif = noeud_fictif_de_neumann(interieur, pas, conductivite, flux=flux)
    source = generation * pas**2 / conductivite
    return (interieur + fictif + sum(lateraux) + source) / 4
