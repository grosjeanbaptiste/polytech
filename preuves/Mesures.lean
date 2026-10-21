import Preuves.Ancrages
import Preuves.Echelle

/-!
Écrit les mesures que les tests du dépôt relisent. Depuis `preuves/` :

    dotfiles heavy run --quick -- lake build Preuves.Ancrages
    dotfiles heavy run --quick -- lake env lean Mesures.lean
-/

#ancrages "ancrages.json"

#etapes EuclideanGeometry.dist_sq_eq_dist_sq_add_dist_sq_iff_angle_eq_pi_div_two
  "pythagore-etapes.json"

#echelle "echelle.json"
