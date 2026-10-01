"""Contexte borne : electromagnetisme (cours EM_I a EM_IV).

Les treize notions du cours, de l'electrostatique a l'equation d'onde.
Trois ancrages chiffres viennent des transcriptions manuscrites du depot,
qui sont les seuls documents lisibles de la matiere -- les fascicules
d'origine sont des scans dont l'OCR ne tirait que du charabia :

* `Exs_03_Stationnaire_Thevenin_Norton_01` : source de 110 V en serie avec
  1 ohm, en parallele avec 10 A, donne 120 V a vide ;
* `Exs_03_Stationnaire_CircuitsMagnetiques_06` : `B = 1,6 T` lu sur la
  courbe donne `H = 70 A/m`, et `70 x 2 pi x 0,16 / 1000` donne
  `I = 0,07037 A` ;
* `Exs_02_Statique_1.1_Poisson` : la divergence en cylindriques, facteur
  `1/rho` compris.

Et un quatrieme, intrinseque et plus fort que les trois : les constantes du
vide sont declarees **independamment**, donc
`c = 1 / racine(epsilon_0 mu_0)` est une verification. Une faute de
transcription sur l'une des trois se verrait la, et nulle part ailleurs.
"""
