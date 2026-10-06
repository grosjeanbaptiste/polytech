"""Les donnees de l'exercice cote du 19 mai 2016.

Source : `Electromagnetisme/EXS/Exs_Cote_2015_2016_Enonce.pdf`. Un seul
endroit les porte. Les longueurs sont en centimetres dans l'enonce et
converties ici une fois pour toutes.

Deux valeurs ne sont pas des donnees mais des **lectures de courbe** :
elles sont nommees comme telles, avec la planche dont elles viennent.
"""

from __future__ import annotations

import math
from typing import Final

# --- Question 1 --------------------------------------------------------------
E1: Final = 2
E2: Final = 2
E3: Final = 1
R1: Final = 1
R3_Q1: Final = 1
R4: Final = 1
CHARGE_R: Final = 2
"""`R = 2` ohms, le dipole branche ensuite entre A et B."""
CHARGE_C_EN_MICROFARADS: Final = 1
"""`C = 1` microfarad."""

# --- Question 2 --------------------------------------------------------------
EPAISSEUR: Final = 2e-2
LARGEURS: Final = (1e-2, 1.5e-2, 2e-2, 1.5e-2, 1e-2)
"""Colonne gauche, fenetre, colonne centrale, fenetre, colonne droite."""
HAUTEURS: Final = (1e-2, 2.5e-2, 1e-2)
"""Culasse haute, fenetre, culasse basse."""
SPIRES_Q2: Final = 100
INDUCTANCE_PROPRE: Final = 1.0
"""`L11 = 1` H, donnee que la solution n'emploie pas."""
CHAMP_CENTRAL: Final = 250.0
"""`H_c`, en A/m."""
INDUCTION_LUE_A_250: Final = 1.0
"""`B`, en teslas, **lu sur la planche A** (toles ordinaires) pour 250 A/m."""

# --- Question 4 --------------------------------------------------------------
FREQUENCE: Final = 50.0
PULSATION: Final = 2 * math.pi * FREQUENCE
TENSION_EFFICACE: Final = 220.0
AVANCE_DE_E2_EN_DEGRES: Final = 20.0
R1_Q4: Final = 10.0
R2_Q4: Final = 10.0
R3: Final = 10.0
C2: Final = 2e-6
L3: Final = 2e-3
I1_DONNE: Final = complex(11.012, -0.375)
I2_DONNE: Final = complex(-10.968, 0.314)
"""Les deux courants que l'enonce donne, a trois decimales, pour la question 4.2."""

# --- Question 5 --------------------------------------------------------------
E0: Final = 2.0
PHASE: Final = math.pi / 2
R: Final = 10.0
C: Final = 1e-6

# --- Question 6 --------------------------------------------------------------
RAYON_MOYEN: Final = 5e-2
RAYON_DE_SECTION: Final = 1e-2
E0_Q6: Final = 1.0
R0_Q6: Final = 1.0
CHAMP_VOULU: Final = 50.0
INDUCTANCE_VOULUE: Final = 0.9488
INDUCTION_LUE_A_50: Final = 1.51
"""`B`, en teslas, **lu sur la courbe de l'hipersil** pour 50 A/m."""
CHAMP_DE_SATURATION: Final = 250.0
INDUCTION_DE_SATURATION: Final = 1.8
