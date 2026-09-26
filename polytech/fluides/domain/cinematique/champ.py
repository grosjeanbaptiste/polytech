"""Cours III -- le champ de vitesse, entite centrale de la cinematique.

Modele *riche* : le champ repond lui-meme aux questions du cours plutot que
d'exposer ses composantes a un service externe qui les manipulerait.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.cinematique import derivees, evolution, lignes, operateurs
from polytech.fluides.domain.cinematique.symboles import T, X, Y


@dataclass(frozen=True, slots=True)
class ChampDeVitesse:
    """Champ eulerien plan ``V(x, y, t) = u e_x + v e_y``."""

    u: sp.Expr
    v: sp.Expr

    def est_stationnaire(self) -> bool:
        """Vrai si ``dV/dt = 0`` partout (cours III.1)."""
        return bool(sp.diff(self.u, T) == 0 and sp.diff(self.v, T) == 0)

    def derivees_partielles(self) -> derivees.DeriveesPartielles:
        """Les six derivees partielles du champ (cours III.2)."""
        return derivees.derivees_partielles(self.u, self.v)

    def gradient_temporel(self) -> tuple[sp.Expr, sp.Expr]:
        """Derivee locale ``dV/dt``, celle qui teste la stationnarite."""
        d = self.derivees_partielles()
        return (d.du_dt, d.dv_dt)

    def loi_de_volume(self) -> sp.Expr:
        """Rapport ``V(t)/V0`` d'une particule, integre depuis la divergence."""
        return evolution.loi_de_volume(self.divergence())

    def ecart_materiel(self, axe: int) -> sp.Expr:
        """Rapport ``delta(t)/delta_0`` le long d'un axe principal."""
        return evolution.ecart_materiel(self.tenseur_taux_deformation(), axe)

    def pente_lignes_de_courant(self) -> sp.Expr:
        """Pente ``dy/dx = v/u`` d'une ligne de courant (cours III.3)."""
        return sp.simplify(self.v / self.u)

    def lignes_de_courant_explicites(self) -> sp.Expr:
        """Rend ``y(x)`` a instant fige, par integration directe."""
        return lignes.lignes_de_courant_explicites(self.u, self.v)

    def equation_lignes_de_courant(self) -> sp.Eq:
        """Equation implicite des lignes de courant (cours III.3)."""
        return lignes.equation_lignes_de_courant(self.u, self.v)

    def divergence(self) -> sp.Expr:
        """Taux de dilatation volumique relatif (cours III.4)."""
        return operateurs.divergence(self.u, self.v)

    def est_incompressible(self) -> bool:
        """Vrai si le volume d'une particule reste constant."""
        return bool(self.divergence() == 0)

    def taux_de_rotation(self) -> sp.Expr:
        """Vitesse angulaire propre ``omega_z`` (cours III.5)."""
        return operateurs.taux_de_rotation(self.u, self.v)

    def vorticite(self) -> sp.Expr:
        """Vorticite ``zeta_z = 2 omega_z`` (cours III.5)."""
        return operateurs.vorticite(self.u, self.v)

    def est_rotationnel(self) -> bool:
        """Vrai si les particules tournent sur elles-memes."""
        return bool(self.taux_de_rotation() != 0)

    def tenseur_taux_deformation(self) -> sp.Matrix:
        """Tenseur des taux de deformation (cours III.4)."""
        return operateurs.tenseur_taux_deformation(self.u, self.v)

    def acceleration(self) -> tuple[sp.Expr, sp.Expr]:
        """Champ d'acceleration, derivee particulaire de la vitesse."""
        return (
            operateurs.derivee_materielle(self.u, self.u, self.v),
            operateurs.derivee_materielle(self.v, self.u, self.v),
        )

    def acceleration_en(self, x: float, y: float, t: float) -> tuple[float, float]:
        """Acceleration evaluee en un point et a un instant donnes."""
        ax, ay = self.acceleration()
        valeurs = {X: x, Y: y, T: t}
        return (float(ax.subs(valeurs)), float(ay.subs(valeurs)))

    def admet_une_fonction_de_courant(self) -> bool:
        """Vrai si ``div V = 0``, condition d'existence de ``psi``."""
        return self.est_incompressible()

    def fonction_de_courant(self) -> sp.Expr:
        """``psi`` telle que ``u = dpsi/dy``, ``v = -dpsi/dx``."""
        return lignes.fonction_de_courant(self.u, self.v)

    def admet_un_potentiel(self) -> bool:
        """Vrai si l'ecoulement est irrotationnel."""
        return not self.est_rotationnel()

    def potentiel_des_vitesses(self) -> sp.Expr:
        """``phi`` telle que ``V = grad phi``."""
        return lignes.potentiel_des_vitesses(self.u, self.v)

    def trajectoire_est_elementaire(self) -> bool:
        """Vrai si la trajectoire admet une forme close.

        Le champ B n'en a pas : ``dy/dt = x - y`` avec ``x = x0 e^{t^2/2}``
        conduit a une primitive non elementaire. C'est une propriete du
        champ, que le corrige doit pouvoir citer sans provoquer d'erreur.
        """
        try:
            self.trajectoire(1.0, 1.0)
        except lignes.TrajectoireNonElementaire:
            return False
        return True

    def trajectoire(self, x0: float, y0: float) -> tuple[sp.Expr, sp.Expr]:
        """Position d'une particule partie de ``(x0, y0)`` a ``t = 0``."""
        return lignes.trajectoire(self.u, self.v, x0, y0)
