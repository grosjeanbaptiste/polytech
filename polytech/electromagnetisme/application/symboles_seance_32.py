"""Les donnees de la seance 3.2, telles que le recueil les imprime.

Section 3.3.3 du recueil (`Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`,
pages imprimees 89 a 92). Les valeurs sont des rationnels exacts : les
arrondis ne se font qu'a la publication, et c'est ce qui permet de dire
d'ou vient l'ecart d'un manuscrit qui, lui, arrondit en chemin.

Les exercices litteraux (2, 5 et 7) partagent leurs symboles quand le
recueil leur donne le meme nom : chaque expression n'en emploie que les
siens, et un `R_1` vaut pour l'exercice ou il apparait.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

# --- Exercice 1 : deux alternateurs en parallele sur une charge ---------
UN_FREQUENCE: Final = sp.Integer(50)
UN_E1: Final = sp.Integer(220)
UN_E2: Final = sp.Rational(439, 2)
UN_RETARD: Final = sp.Rational(5, 2)
"""Le retard de `e_2` sur `e_1`, en degres."""
UN_R: Final = sp.Integer(9)
UN_L1: Final = sp.Rational(10, 1000)
UN_L2: Final = sp.Rational(8, 1000)
OMEGA_DU_UN: Final = 2 * sp.pi * UN_FREQUENCE

# --- Exercice 5 : la branche e3 - R3 // C3 ------------------------------
CINQ_FREQUENCE: Final = sp.Integer(50)
CINQ_E3: Final = sp.Integer(220)
CINQ_R3: Final = sp.Integer(1000)
CINQ_C3: Final = sp.Rational(1, 10**6)
OMEGA_DU_CINQ: Final = 2 * sp.pi * CINQ_FREQUENCE
COURANT_UN_DU_CINQ: Final = sp.Rational(-2, 10) + sp.Rational(2, 10) * sp.I
COURANT_TROIS_DU_CINQ: Final = sp.Rational(2, 10) - sp.Rational(3, 10) * sp.I
"""Les deux courants que l'enonce donne comme resultat de la resolution."""

# --- Exercice 6 : deux circuits couples ---------------------------------
SIX_FREQUENCE: Final = sp.Integer(1000)
SIX_E1: Final = sp.Integer(20)
SIX_R1: Final = sp.Integer(1000)
SIX_R2: Final = sp.Integer(10)
SIX_L1: Final = sp.Rational(10, 1000)
SIX_L2: Final = sp.Rational(20, 1000)
SIX_L12: Final = sp.Rational(5, 1000)
SIX_C2: Final = sp.Rational(1, 10**6)
OMEGA_DU_SIX: Final = 2 * sp.pi * SIX_FREQUENCE

# --- Symboles des exercices litteraux -----------------------------------
E0, R0, C, R, R1, R2, L1, L2 = sp.symbols("E_0 R_0 C R R_1 R_2 L_1 L_2", positive=True)
R3, C3, E, L = sp.symbols("R_3 C_3 E L", positive=True)
THETA0, THETA = sp.symbols("theta_0 theta_u", real=True)
E1, E3 = sp.symbols("E_1 E_3")
"""Les phaseurs des sources de l'exercice 5 : complexes, sans hypothese."""
U0: Final = sp.Symbol("u_{C}(0)", real=True)
"""La tension du condensateur au basculement, exercice 2."""

ILLUSTRATION_OMEGA_TAU: Final = sp.Integer(5)
"""Le produit `omega tau` de la figure d'allure de l'exercice 7.

L'enonce est litteral : il ne fixe aucune valeur, et la figure en
choisit une. Cinq periodes d'oscillation par constante de temps, ou
presque, laissent voir le transitoire sur les premieres periodes et son
extinction ensuite. C'est un choix d'illustration, non une donnee.
"""
