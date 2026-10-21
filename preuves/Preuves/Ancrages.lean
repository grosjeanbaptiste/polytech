import Mathlib
import Preuves.Etapes

/-!
# Les ancrages : une notion de `polytech/`, ses résultats dans Mathlib

`polytech/` calcule et explique ; Mathlib démontre. Un ancrage relie un
symbole de la bibliothèque aux constantes de Mathlib qui portent la même
notion, et de là, par `etapes`, à tout ce qu'elles emploient jusqu'aux
axiomes.

Les constantes s'écrivent ``` ``nom ``` : Lean refuse à la compilation un nom
qui n'existe pas, si bien qu'un ancrage ne peut pas citer un théorème
imaginaire. Le symbole Python, lui, est vérifié du côté du dépôt
(`tests/test_ancrages.py`). `#ancrages "ancrages.json"` écrit la mesure que ce
test relit.

Un ancrage dit « même notion », pas « même code » : `produit_vectoriel` calcule
sur des listes de nombres, `crossProduct` est défini sur `Fin 3 → R` pour tout
anneau commutatif.
-/

open Lean

namespace Preuves

/-- Symbole de `polytech/`, et les constantes de Mathlib qui portent sa notion. -/
def ancrages : List (String × List Name) := [
  ("polytech.maths.domain.algebre.produit_vectoriel.produit_vectoriel",
    [``crossProduct, ``cross_anticomm, ``dot_self_cross]),
  ("polytech.maths.domain.algebre.produit_vectoriel.produit_mixte",
    [``triple_product_eq_det]),
  ("polytech.maths.domain.algebre.metrique.produit_scalaire",
    [``dotProduct, ``EuclideanSpace.inner_eq_star_dotProduct]),
  ("polytech.maths.domain.algebre.metrique.norme",
    [``EuclideanSpace.norm_eq]),
  ("polytech.maths.domain.algebre.metrique.distance",
    [``EuclideanSpace.dist_eq]),
  ("polytech.maths.domain.algebre.metrique.gram_schmidt",
    [``InnerProductSpace.gramSchmidt, ``InnerProductSpace.gramSchmidt_orthogonal]),
  ("polytech.maths.domain.algebre.complexes.conjugue",
    [``Complex.conj_re, ``Complex.conj_im]),
  ("polytech.maths.domain.algebre.complexes.argument",
    [``Complex.arg]),
  ("polytech.maths.domain.algebre.complexes_exacts.par_moivre",
    [``Complex.cos_add_sin_mul_I_pow])
]

/-- Les axiomes qu'atteint la chaîne d'un ancrage (sans doublon). -/
def axiomesAtteints (env : Environment) (g : Std.HashMap Name (Array Name)) : Array Name :=
  (g.toList.filter (nature env ·.1 == "axiome")).map (·.1) |>.toArray

/-- `#ancrages "ancrages.json"` : pour chaque symbole, ses constantes, le nombre
d'étapes de leur amont réuni, et les axiomes atteints. -/
elab "#ancrages " chemin:str : command => do
  let env ← getEnv
  let lignes := ancrages.map fun (symbole, noms) =>
    let ams := noms.map (amont env)
    let reunion := ams.foldl (fun acc g => g.fold (fun a k v => a.insert k v) acc) {}
    let nEtapes := (noms.map (etapes env) |>.foldl
      (fun acc g => g.fold (fun a k _ => a.insert k) acc) (∅ : Std.HashSet Name)).size
    Json.mkObj [
      ("symbole", Json.str symbole),
      ("mathlib", Json.arr (noms.toArray.map (Json.str ·.toString))),
      ("etapes", Json.num nEtapes),
      ("constantes", Json.num reunion.size),
      ("axiomes", Json.arr ((axiomesAtteints env reunion).qsort (·.toString < ·.toString)
        |>.map (Json.str ·.toString)))]
  IO.FS.writeFile chemin.getString (Json.arr lignes.toArray).pretty
  logInfo m!"{ancrages.length} ancrages écrits dans {chemin.getString}"

end Preuves
