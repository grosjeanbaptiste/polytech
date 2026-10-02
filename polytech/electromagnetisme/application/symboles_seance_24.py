"""Les symboles et les donnees de la seance 2.4.

Separes des reseaux quand `enonces_seance_24` a franchi la centaine de
lignes. La coupure suit une vraie frontiere : ici, les **noms** des
grandeurs et les **valeurs** que le recueil leur donne ; la-bas, la
facon dont les elements sont relies.
"""

from __future__ import annotations

import sympy as sp

R1, R2, R3, R4, R5 = sp.symbols("R_1 R_2 R_3 R_4 R_5", positive=True)
CHARGE = sp.Symbol("R", positive=True)
CHARGE_PRIME = sp.Symbol("R'", positive=True)
TENSION = sp.Symbol("E", positive=True)
TENSION_1, TENSION_2, TENSION_5 = sp.symbols("E_1 E_2 E_5", positive=True)
COURANT = sp.Symbol("I", positive=True)
COURANT_3 = sp.Symbol("I_3", positive=True)
COURANT_PRIME = sp.Symbol("I'", positive=True)

NUMERO_DE_LA_SEANCE = "2.4"
"""Le numero de la seance, tel que le manuscrit le porte.

Publie comme une macro, et non tape dans le document : il se trouve etre
aussi la valeur de la resistance de Thevenin de l'exercice 6, en ohms. La
garde des valeurs tapees ne distingue pas les deux, et elle a raison de
ne pas essayer.
"""

DONNEES_DU_SIX = {
    R1: 1,
    R3: 5,
    R4: 4,
    R5: 5,
    TENSION_1: 20,
    TENSION_5: 10,
    CHARGE_PRIME: sp.Rational(16, 10),
    COURANT_PRIME: 2,
}
"""Les valeurs de l'exercice 6, lues dans le recueil."""
