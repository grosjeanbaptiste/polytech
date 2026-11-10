# polypy

Bibliothèque scientifique et de calcul formel/numérique pour les matières de sciences appliquées et de l'ingénieur.

Le package Python s'importe sous le nom `polytech` :
```python
import polytech.fluides
import polytech.maths
```

## Structure
- `polytech/` : Package Python pur (domain-driven design, SymPy, NumPy, SciPy)
- `polytech-core/` : Crate Rust (calculs physiques natifs)
- `preuves/` : Preuves formelles Lean 4 et ancrages de théorèmes
- `tests/` : Suite de tests de vérification et de non-régression
