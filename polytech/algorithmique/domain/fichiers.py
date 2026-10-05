"""Les fichiers texte : ecrire, relire, et savoir quand s'arreter.

Le cours (partie 1, planches 136 a 142 ; S3, planches 2 a 10) :

* `ofstream` ouvre un fichier en ecriture et **ecrase** ce qu'il
  contenait ; avec `ios::app`, il ecrit a la suite ;
* `ifstream` l'ouvre en lecture, et se lit comme `cin` : mot a mot par
  `>>`, ligne a ligne par `getline`, avec les memes `ignore` ;
* l'ouverture peut echouer -- le fichier n'existe pas -- et se teste ;
* `eof()` ne devient vrai qu'**apres** une lecture qui a bute sur la fin
  du fichier. La boucle `while (!f.eof()) { f >> ...; }` fait donc un tour
  de trop des que le fichier se termine par un saut de ligne.

Le disque est ici un dictionnaire : le domaine ne touche a aucun fichier.
"""

from __future__ import annotations

from polytech.algorithmique.domain.flux_d_entree import Clavier


class Disque:
    """Des fichiers texte, par leur nom."""

    def __init__(self) -> None:
        """Un disque vide."""
        self._fichiers: dict[str, str] = {}

    def existe(self, nom: str) -> bool:
        """Vrai si l'ouverture en lecture reussirait."""
        return nom in self._fichiers

    def ecrire(self, nom: str, texte: str, *, a_la_suite: bool = False) -> None:
        """Ouvre, ecrit, ferme : le contenu est ecrase, sauf `a_la_suite`."""
        ancien = self._fichiers.get(nom, "") if a_la_suite else ""
        self._fichiers[nom] = ancien + texte

    def contenu(self, nom: str) -> str:
        """Tout le texte du fichier ; leve s'il n'existe pas."""
        if nom not in self._fichiers:
            msg = f"Le fichier {nom} n'existe pas."
            raise FileNotFoundError(msg)
        return self._fichiers[nom]

    def lire(self, nom: str) -> Clavier:
        """Un flux de lecture place au debut du fichier."""
        return Clavier(self.contenu(nom))


def tours_de_la_boucle_eof(contenu: str, *, champs: int) -> int:
    """Le nombre de tours de `while (!f.eof()) { f >> c1 >> ... ; }`, mots ou nombres.

    `eof` se leve quand une lecture bute sur la fin : soit il ne reste plus
    rien a lire, soit le dernier mot touche la fin du fichier.
    """
    mots = contenu.split()
    colle_a_la_fin = contenu != "" and not contenu[-1].isspace()
    tours, lus, fin = 0, 0, False
    while not fin:
        tours += 1
        for _ in range(champs):
            if lus == len(mots):
                fin = True
                break
            lus += 1
            fin = lus == len(mots) and colle_a_la_fin
    return tours
