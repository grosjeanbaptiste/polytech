"""L'echauffement d'un composant electronique sur son refroidisseur.

Syllabus de thermique, tome 1, premiere partie, section 3.3. Un
transistor se represente par des constantes concentrees : la jonction,
de capacite negligeable, tient au boitier par `R_JB` ; le boitier, de
capacite `C_B`, au refroidisseur par `R_BR` ; le refroidisseur, de
capacite enorme, a l'ambiance par `R_RA`. Le refroidisseur ne voit donc
que les puissances moyennes, les siennes et celles des autres boitiers.

En valeurs moyennes, les resistances s'ajoutent. Pour une composante
variable de la puissance, la transmittance de la jonction vaut
`R_JB + R_BR / (1 + p R_BR C_B)` : tres vite, il n'en reste que `R_JB`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P

type Grandeur = sp.Expr | float


@dataclass(frozen=True, slots=True)
class ComposantSurRefroidisseur:
    """`R_JB`, `R_BR` et `R_RA`, en degres par watt."""

    jonction_boitier: float
    boitier_refroidisseur: float
    vers_l_ambiance: float

    def jonction_moyenne(
        self, *, ambiance: float, puissance: float, des_autres: float
    ) -> float:
        """`t_J = t_A + Phi (R_JB + R_BR) + (Phi + Phi_2) R_RA`."""
        refroidisseur = ambiance + (puissance + des_autres) * self.vers_l_ambiance
        interne = self.jonction_boitier + self.boitier_refroidisseur
        return refroidisseur + puissance * interne

    def transmittance(self, capacite: Grandeur) -> sp.Expr:
        """`F(p) = [t_J] / [Phi] = R_JB + R_BR / (1 + p R_BR C_B)`."""
        vers_le_boitier = sp.nsimplify(self.jonction_boitier)
        collage = sp.nsimplify(self.boitier_refroidisseur)
        return vers_le_boitier + collage / (1 + P * collage * capacite)

    def amplitude_de_la_jonction(
        self, puissance: float, *, pulsation: float, capacite: float
    ) -> float:
        """L'amplitude de `t_J` sous une puissance sinusoidale d'amplitude donnee."""
        isochrone = self.transmittance(capacite).subs(P, sp.I * pulsation)
        return puissance * float(sp.Abs(isochrone))
