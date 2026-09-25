"""Cursus Polytech : un paquet par matiere, un contexte borne par paquet.

Chaque matiere possede son propre langage ubiquitaire, et deux matieres
peuvent nommer differemment la meme chose -- ou pire, nommer pareil deux
choses differentes. Un ``champ`` n'est pas le meme objet en cinematique des
fluides et en electromagnetisme. Les contextes sont donc etanches : aucun
n'importe le domaine d'un autre, et la traduction, s'il en faut une un jour,
se fait explicitement a la frontiere.
"""
