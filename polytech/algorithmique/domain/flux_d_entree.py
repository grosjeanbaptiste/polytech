"""Ce que `cin` lit : un mot, une ligne, et le saut de ligne qui reste.

Le cours (partie 1, planches 15 et 78 a 80) :

* `cin >> variable` saute les blancs, lit jusqu'au blanc suivant, et
  **laisse ce blanc dans le tampon** : on ne lit pas ainsi une ligne
  entiere contenant des espaces ;
* `getline(cin, chaine)` lit jusqu'au saut de ligne, qu'il consomme ;
* apres un `cin >>`, le saut de ligne tape pour valider est encore la : le
  `getline` suivant lit une ligne vide. Il faut un `cin.ignore()` entre
  les deux -- et un seul, sans quoi il mange la premiere lettre.

`cin.ignore(n, c)` saute au plus `n` caracteres, ou jusqu'a `c` compris.
"""

from __future__ import annotations


class Clavier:
    """Le tampon d'entree : ce qui a ete tape et n'est pas encore lu."""

    def __init__(self, saisie: str) -> None:
        """Tout ce que l'utilisateur tapera, sauts de ligne compris."""
        self._tampon = saisie

    def epuise(self) -> bool:
        """Vrai s'il ne reste rien a lire."""
        return not self._tampon

    def _exiger_une_suite(self, restant: str) -> None:
        if not restant:
            msg = "plus rien a lire : la saisie est epuisee"
            raise EOFError(msg)

    def mot(self) -> str:
        """`cin >> variable` : le prochain mot ; le blanc qui le suit reste."""
        restant = self._tampon.lstrip()
        self._exiger_une_suite(restant)
        lu = restant.split(maxsplit=1)[0]
        self._tampon = restant[len(lu) :]
        return lu

    def caractere(self) -> str:
        """`cin.get()` : le prochain caractere, blanc ou retour a la ligne compris."""
        self._exiger_une_suite(self._tampon)
        lu, self._tampon = self._tampon[0], self._tampon[1:]
        return lu

    def ligne(self, delimiteur: str = "\n") -> str:
        """`getline` : jusqu'au delimiteur, qui est consomme."""
        self._exiger_une_suite(self._tampon)
        lu, _, self._tampon = self._tampon.partition(delimiteur)
        return lu

    def ignorer(self, nombre: int = 1, arret: str | None = None) -> None:
        """`cin.ignore(nombre, arret)` : saute des caracteres, l'arret compris."""
        portee = self._tampon[:nombre]
        if arret is not None and arret in portee:
            nombre = portee.index(arret) + 1
        self._tampon = self._tampon[nombre:]
