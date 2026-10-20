import Lean

/-!
# L'amont d'un résultat, jusqu'aux axiomes

Consigne du 2026-10-19 : reconstruire chaque formule « en démontrant toutes
les étapes permettant d'accéder aux nouvelles formules, de façon récursive ».

Dans Lean, une preuve est un terme : les résultats qu'elle emploie y sont
écrits, et le noyau a vérifié chacun. L'amont d'un résultat se lit donc dans
l'environnement, sans rien supposer : les constantes de son énoncé et de sa
preuve, puis les leurs, jusqu'à ce qui n'emploie plus rien — les axiomes et
les types de base.

`#amont nom "fichier.json"` écrit ce graphe : pour chaque constante, ce
qu'elle emploie directement, et si c'est un axiome, un théorème ou une
définition.
-/

open Lean Elab Command

namespace Preuves

/-- Les constantes que `c` emploie directement : son énoncé et sa preuve. -/
def directes (env : Environment) (c : Name) : Array Name :=
  match env.find? c with
  | none => #[]
  | some info =>
    let preuve := (info.value? (allowOpaque := true)).map Expr.getUsedConstants
    (info.type.getUsedConstants ++ preuve.getD #[]).toList.eraseDups.toArray

/-- La nature d'une constante, pour la lecture du graphe. -/
def nature (env : Environment) (c : Name) : String :=
  match env.find? c with
  | some (.axiomInfo _) => "axiome"
  | some (.thmInfo _) => "theoreme"
  | some (.defnInfo _) => "definition"
  | some (.inductInfo _) | some (.ctorInfo _) | some (.recInfo _) => "type"
  | some _ => "autre"
  | none => "inconnue"

/-- Le graphe amont de `racine`, en largeur d'abord, sans doublon. -/
partial def amont (env : Environment) (racine : Name) :
    Std.HashMap Name (Array Name) :=
  go #[racine] {}
where
  go (file : Array Name) (vus : Std.HashMap Name (Array Name)) :
      Std.HashMap Name (Array Name) :=
    if file.isEmpty then vus else
      let (suivants, vus) := file.foldl (init := (#[], vus)) fun (s, v) c =>
        if v.contains c then (s, v) else
          let d := directes env c
          (s ++ d.filter (!v.contains ·), v.insert c d)
      go suivants vus

/-- Le graphe en JSON : `{constante: {nature, emploie}}`. -/
def enJson (env : Environment) (g : Std.HashMap Name (Array Name)) : Json :=
  Json.mkObj <| g.toList.map fun (c, d) =>
    (c.toString, Json.mkObj [
      ("nature", Json.str (nature env c)),
      ("emploie", Json.arr (d.map (Json.str ·.toString)))])

/-- `#amont nom "fichier.json"` : écrit l'amont de `nom` et dit sa taille. -/
elab "#amont " id:ident chemin:str : command => do
  let env ← getEnv
  let nom ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
  let g := amont env nom
  IO.FS.writeFile chemin.getString (enJson env g).compress
  let axiomes := g.toList.filter (nature env ·.1 == "axiome") |>.map (·.1)
  logInfo m!"{nom} : {g.size} constantes en amont, axiomes {axiomes}"

end Preuves
