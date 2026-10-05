"""L'analyse dimensionnelle de la convection, et ses variables reduites.

Syllabus de thermique, tome 2, premiere partie, chapitre 2. La densite
de flux depend de sept variables, qui s'expriment en quatre unites --
longueur, temps, temperature, energie : le theoreme des pi les ramene a
trois nombres. En convection forcee, `Nu = F(Re, Pr)` ; en convection
naturelle il n'y a pas de vitesse imposee, la force d'Archimede entre
par le produit `beta g`, et `Nu = F(Gr, Pr)`.

Le nombre de Nusselt, `K l / lambda`, rapporte le flux reel a celui de
la conduction pure a travers une couche d'epaisseur `l`.

La dilatation qui met le fluide en mouvement se mesure par
`beta = -(1 / rho) d(rho)/dT`, soit `1 / T` pour un gaz parfait --
le contexte des fluides le porte deja.

Les proprietes se prennent a la temperature de film, moyenne de la
paroi et du fluide. Dans une conduite, la temperature du fluide est
celle de melange, ponderee par le debit. Pour un liquide visqueux, la
viscosite se linearise entre le melange et la paroi.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

_COMMUNES: Final = {
    "phi": {"Q": 1, "L": -2, "T": -1},
    "l": {"L": 1},
    "nu": {"L": 2, "T": -1},
    "theta_o": {"Theta": 1},
    "a": {"L": 2, "T": -1},
    "lambda": {"Q": 1, "L": -1, "T": -1, "Theta": -1},
}
CONVECTION_FORCEE: Final = {**_COMMUNES, "u_o": {"L": 1, "T": -1}}
"""Page 24 : les equations aux dimensions des sept variables."""
CONVECTION_NATURELLE: Final = {**_COMMUNES, "beta_g": {"L": 1, "T": -2, "Theta": -1}}
"""Page 26 : `u_o` disparait, `beta g` le remplace."""


def nusselt(echange: float, longueur: float, conductivite: float) -> float:
    """`Nu = K l / lambda`."""
    if conductivite <= 0:
        msg = f"conductivite nulle ou negative : {conductivite} W/(m K)"
        raise ValueError(msg)
    return echange * longueur / conductivite


def prandtl_des_proprietes(
    viscosite: float, chaleur_massique: float, conductivite: float
) -> float:
    """`Pr = mu c_p / lambda`."""
    return viscosite * chaleur_massique / conductivite


def peclet(reynolds: float, prandtl: float) -> float:
    """`Pe = Re Pr`."""
    return reynolds * prandtl


def stanton(nusselt_local: float, reynolds: float, prandtl: float) -> float:
    """`St = Nu / (Re Pr)`, soit `K / (rho c_p u)`."""
    return nusselt_local / (reynolds * prandtl)


def masse_volumique_dilatee(reference: float, dilatation: float, ecart: float) -> float:
    """`rho = rho_o (1 - beta theta)`."""
    return reference * (1 - dilatation * ecart)


def temperature_de_film(paroi: float, fluide: float) -> float:
    """`(t_p + t_o) / 2`."""
    return (paroi + fluide) / 2


def temperature_de_melange(
    temperature: sp.Expr, vitesse: sp.Expr, rayon: sp.Symbol, tube: sp.Expr
) -> sp.Expr:
    """`t_m` dans un tube de rayon `tube`, a masse volumique uniforme."""
    couronne = 2 * sp.pi * rayon
    debit = sp.integrate(vitesse * couronne, (rayon, 0, tube))
    transporte = sp.integrate(temperature * vitesse * couronne, (rayon, 0, tube))
    return sp.simplify(transporte / debit)


def viscosite_pres_du_melange(
    temperature: float, *, melange: tuple[float, float], paroi: tuple[float, float]
) -> float:
    """`mu(t)`, lineaire entre `(t_m, mu_m)` et `(t_p, mu_p)`."""
    if paroi[0] == melange[0]:
        msg = f"paroi a la temperature de melange : {paroi[0]}"
        raise ValueError(msg)
    avancement = (temperature - melange[0]) / (paroi[0] - melange[0])
    return melange[1] + (paroi[1] - melange[1]) * avancement
