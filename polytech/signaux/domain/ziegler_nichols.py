"""La methode de Ziegler-Nichols, sur la reponse indicielle du procede.

Planches « Controleur » du cours de signaux et systemes. Methode
experimentale, rapide, a comparer a la synthese frequentielle. On
enregistre la reponse indicielle du procede autour d'un point de
fonctionnement, et l'on trace sa tangente au point d'inflexion : elle
donne le retard `tau` et la duree de montee `Delta T`. Avec le gain en
continu `K`, la pente `R = K / Delta T` fait du procede un integrateur
retarde, `T(p) = R exp(-p tau) / p`.

Le tableau de reglage s'applique au controleur parallele
`K_p (1 + 1 / (p tau_i) + p tau_d)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P

type Reglage = tuple[float, float | None, float | None]
"""`K_p`, `tau_i` et `tau_d`, les deux derniers quand le controleur les porte."""


@dataclass(frozen=True, slots=True)
class ModeleDeZieglerNichols:
    """Le gain en continu `K`, la duree `Delta T` et le retard `tau`."""

    gain: float
    montee: float
    retard: float

    def pente(self) -> float:
        """`R = K / Delta T`."""
        return self.gain / self.montee

    def transmittance(self) -> sp.Expr:
        """`T(p) = R exp(-p tau) / p`."""
        return self.pente() * sp.exp(-self.retard * P) / P


def reglage_de_ziegler_nichols(modele: ModeleDeZieglerNichols, nature: str) -> Reglage:
    """Le tableau de la planche 34, pour un controleur P, PI ou PID."""
    unite = 1 / (modele.pente() * modele.retard)
    tableau: dict[str, Reglage] = {
        "P": (unite, None, None),
        "PI": (0.9 * unite, 3 * modele.retard, None),
        "PID": (1.2 * unite, 2 * modele.retard, modele.retard / 2),
    }
    if nature not in tableau:
        msg = f"controleur inconnu du tableau de Ziegler-Nichols : {nature}"
        raise ValueError(msg)
    return tableau[nature]
