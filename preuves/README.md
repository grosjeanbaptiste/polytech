# Preuves

Le volet vérifié de la bibliothèque : chaque formule se dérive, pas à pas,
de résultats déjà établis, jusqu'aux axiomes (consigne du 2026-10-19,
`AGENTS.md`, section « Sur la bibliothèque »). Le vérificateur est Lean 4
avec Mathlib ; `polytech/` reste la couche de calcul et d'explication.

- `Preuves/Amont.lean` — `#amont nom "fichier.json"` écrit le graphe amont
  d'un résultat : ce que son énoncé et sa preuve emploient, récursivement,
  jusqu'aux axiomes.

Mathlib vient en cache compilé (`lake exe cache get`), dans `.lake/`, hors de
git. Toute compilation passe par la file : `dotfiles heavy run --quick --
lake build`.
