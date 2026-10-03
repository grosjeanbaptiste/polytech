"""Les donnees du TP 1 de signaux, telles que l'enonce les donne.

Source : `SignauxSystemes/LABOS/Lab_01_Enonce.pdf`, huit pages, que Moodle
sert sous le nom `sigsyscom-labo1.pdf` et auquel renvoie le devoir « TP1
Devoir » du cours de Charleroi. Les durees des questions 1 et 2 sont lues
sur les graphiques de l'enonce, echantillonnes a 1 Hz.

Une seule valeur n'est pas dans l'enonce : la resistance de la question
4f, que l'enonce demande de **choisir**. Elle est nommee comme un choix.
"""

from __future__ import annotations

from typing import Final

# --- Question 1 : a(t) est une porte de dix secondes --------------------
UN_LARGEUR: Final = 10
UN_SUPPORT: Final = 110
"""Le graphique de `a(t)` s'etend jusqu'a 110 s."""

# --- Question 2 : une porte de cent secondes, periodisee a deux cents ---
DEUX_LARGEUR: Final = 100
DEUX_SUPPORT: Final = 200
DEUX_PERIODE: Final = 200
DEUX_IMPULSIONS: Final = 5
"""Cinq periodes de 200 s couvrent les 1000 s du graphique de `c(t)`."""

# --- Question 3 : rect(t/10), echantillonnee a 10 Hz -------------------
TROIS_DUREE: Final = 10.0
TROIS_FREQUENCE: Final = 10.0
TROIS_ZEROS: Final = 1000

# --- Question 4 : L et C en serie, sortie sur R ------------------------
QUATRE_L: Final = 1.0
QUATRE_C: Final = 1e-7
QUATRE_R: Final = 1e3
QUATRE_R_CHOISIE: Final = 1e4
"""La resistance que la question 4f fait choisir : dix kilo-ohms.

L'enonce demande des poles reels distincts, donc une resistance au-dela
de la valeur critique `2 racine(L/C)`, environ 6,3 kilo-ohms. Dix est le
premier ordre de grandeur rond au-dela, et il laisse les deux poles assez
ecartes pour que leurs deux constantes de temps se lisent.
"""

# --- Question 5 : la capsule spatiale ----------------------------------
CINQ_SORTIE: Final = (1, 1, 10, 15)
"""`theta''' + theta'' + 10 theta' + 15 theta` : les derivees, degre decroissant."""
CINQ_ENTREE: Final = (5,)

# --- Question 6 : le cinquieme ordre -----------------------------------
SIX_NUMERATEUR: Final = (0.00000150802, 0.0, 0.0055586, 0.0, 4.119852)
SIX_DENOMINATEUR: Final = (
    0.001,
    0.01094045,
    0.109461,
    0.59656908,
    2.26158275,
    4.119852,
)
