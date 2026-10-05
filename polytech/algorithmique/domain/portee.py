"""Les regles de visibilite : ou une variable existe, et combien de temps.

Le cours (partie 1, planches 115 a 119) : « le C++ est un langage structure
en blocs `{ }`, les variables ne peuvent etre utilisees que la ou elles
sont declarees ».

* une variable declaree dans un bloc **masque** celle du meme nom
  declaree au-dehors, et disparait a la fermeture du bloc ;
* une variable **globale**, declaree hors de toute fonction, est visible
  partout et garde ce que chaque fonction y ecrit ;
* une variable locale **statique** n'est initialisee qu'une fois : elle
  garde sa valeur d'un appel au suivant.
"""

from __future__ import annotations

from typing import Any


class Portees:
    """Une pile de blocs ; le premier est celui des variables globales."""

    def __init__(self) -> None:
        """Le seul bloc global, sans variable."""
        self._blocs: list[dict[str, Any]] = [{}]
        self._statiques: dict[tuple[str, str], list[Any]] = {}

    def ouvrir(self) -> None:
        """Une accolade ouvrante."""
        self._blocs.append({})

    def fermer(self) -> None:
        """Une accolade fermante : les variables du bloc disparaissent."""
        if len(self._blocs) == 1:
            msg = "aucun bloc a fermer : il ne reste que les variables globales"
            raise RuntimeError(msg)
        self._blocs.pop()

    def declarer(self, nom: str, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """Declare dans le bloc courant, en masquant un nom exterieur."""
        self._blocs[-1][nom] = [valeur]

    def declarer_statique(self, fonction: str, nom: str, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """`static type nom = valeur;` : initialisee au premier passage seulement."""
        case = self._statiques.setdefault((fonction, nom), [valeur])
        self._blocs[-1][nom] = case

    def _case(self, nom: str) -> list[Any]:
        for bloc in reversed(self._blocs):
            if nom in bloc:
                case: list[Any] = bloc[nom]
                return case
        msg = f"identificateur non declare dans cette portee : {nom}"
        raise NameError(msg)

    def valeur(self, nom: str) -> Any:  # noqa: ANN401 -- ce que la variable contient
        """La variable du bloc le plus interieur qui porte ce nom."""
        return self._case(nom)[0]

    def affecter(self, nom: str, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """Ecrit dans la variable visible sous ce nom."""
        self._case(nom)[0] = valeur
