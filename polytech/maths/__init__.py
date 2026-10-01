"""Contexte borne : mathematiques (algebre lineaire, integration, EDO, Laplace).

Une bibliotheque batie sur les **etapes**, et non sur les resultats. La
distinction decide de tout : `sympy.integrate` rend une primitive, il ne
rend pas la methode. Or un corrige pas a pas doit montrer la methode --
c'est meme la seule chose qu'il apporte au-dela du corrige officiel, qui
donne deja la reponse.

Le fascicule d'exercices du cours le dit d'ailleurs lui-meme : ses
solutions ne sont pas des resultats secs mais des indications de methode,
« changement de variable u = sin x », « integrations par parties »,
« multiplier et diviser par cosh x - sinh x ». La bibliotheque expose donc
ces gestes comme des objets du domaine, chacun capable de rendre l'etape
intermediaire qu'un lecteur doit voir.
"""
