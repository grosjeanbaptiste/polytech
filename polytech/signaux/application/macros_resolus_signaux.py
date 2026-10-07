"""Macros des exercices resolus, chapitres 1 et 2. Prefixe `Resolus`."""

from __future__ import annotations

import numpy as np
import sympy as sp

from polytech.signaux.application import resolus_convolutions as cv
from polytech.signaux.application import resolus_fourier as fo
from polytech.signaux.application import resolus_inverses as iv
from polytech.signaux.application import resolus_laplace as la
from polytech.signaux.application import resolus_puissances as pu
from polytech.signaux.application import resolus_signaux as sg
from polytech.signaux.application import resolus_trains as tn
from polytech.signaux.application.formatage import coordonnees, nombre
from polytech.signaux.domain.symboles import TEMPS as T


def _courbe(expression: sp.Expr, fin: float, points: int = 161) -> str:
    """Les points d'un original, l'impulsion de Dirac mise a part."""
    instants = np.linspace(0.0, fin, points)
    sans_impulsion = expression.subs(sp.DiracDelta(T), 0)
    valeurs = sp.lambdify(T, sans_impulsion, "numpy")(instants)
    return coordonnees(instants, np.broadcast_to(valeurs, instants.shape), 4)


def _chapitre_un() -> dict[str, str]:
    debut, sommet, fin, hauteur = (float(v) for v in sg.reperes_du_triangle())
    triangle = np.array([[debut, 0.0], [sommet, hauteur], [fin, 0.0]])
    x, y, _ = pu.valeurs_efficaces_des_trains()
    instants = np.linspace(0.0, 5.0, 11)
    echelon = [float(cv.sortie_de_l_echelon(sp.nsimplify(t))) for t in instants]
    return {
        "ResolusDeuxTriangle": coordonnees(triangle[:, 0], triangle[:, 1], 4),
        "ResolusTroisTriangle": sp.latex(sg.energies()[1]),
        "ResolusQuatreX": nombre(float(x), 3),
        "ResolusQuatreY": nombre(float(y), 3),
        "ResolusSixRapport": nombre(pu.rapport_signal_sur_bruit(), 2),
        "ResolusSixAttenue": nombre(pu.rapport_signal_sur_bruit(gain_du_bruit=0.01), 2),
        "ResolusSixAmplifie": nombre(pu.rapport_signal_sur_bruit(gain_du_bruit=100), 2),
        "ResolusSeptFormes": sp.latex(sg.produit_scalaire_des_formes()),
        "ResolusSeptMoyenne": sp.latex(tn.moyenne_du_produit_des_trains()),
        "ResolusSeptCommune": str(tn.periode_commune()),
        "ResolusNeufSortie": str(cv.sortie_de_la_rampe(4)),
        "ResolusDixSortie": coordonnees(instants, np.array(echelon), 2),
    }


def _fourier_et_laplace() -> dict[str, str]:
    rangs = np.arange(-12, 13)
    raies = [float(r) for r in fo.raies_de_la_porte([int(k) for k in rangs])]
    quinze = la.signal_15()
    instants = np.linspace(0.0, 16.0, 81)
    seize = la.signaux_16([round(float(t), 1) for t in instants])
    return {
        "ResolusDouzeRaies": coordonnees(rangs.astype(float), np.array(raies), 4),
        "ResolusDouzeUn": nombre(raies[13], 3),
        "ResolusTreizeRaie": nombre(fo.niveau_de_la_raie_centrale(), 2),
        "ResolusQuinzeSignal": _courbe(quinze, 5.0),
        "ResolusSeizeSignal": coordonnees(instants, np.array(seize), 4),
        "ResolusSeizeEcart": nombre(float(la.ecart_au_raccord()) * 1e3, 3),
    }


def _inverses() -> dict[str, str]:
    exponentielle, amplitude, phase = iv.forme_de_la_solution_20()
    return {
        "ResolusDixSeptA": _courbe(iv.original_de("0.17a"), 6.0),
        "ResolusDixSeptB": _courbe(iv.original_de("0.17b"), 6.0),
        "ResolusDixSeptC": _courbe(iv.original_de("0.17c"), 6.0),
        "ResolusDixHuitSignal": _courbe(iv.original_de("0.18"), 3.0),
        "ResolusDixNeufA": _courbe(iv.original_de("0.19a"), 3.0),
        "ResolusDixNeufB": _courbe(iv.original_de("0.19b"), 3.0),
        "ResolusVingtSignal": _courbe(iv.original_de("0.20"), 12.0),
        "ResolusVingtExponentielle": nombre(float(exponentielle), 1),
        "ResolusVingtAmplitude": nombre(float(amplitude), 3),
        "ResolusVingtPhase": nombre(float(phase), 3),
        "ResolusVingtEtUnSignal": _courbe(iv.original_de("0.21"), 2.0),
    }


def macros_resolus_signaux() -> dict[str, str]:
    """Les macros des exercices 0.2 a 0.21."""
    return {**_chapitre_un(), **_fourier_et_laplace(), **_inverses()}
