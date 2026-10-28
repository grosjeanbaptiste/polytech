//! Traduction vers Python : `Err(message)` devient `ValueError(message)`.

use crate::electronique::Defauts;
use crate::{bandes, comparaison, compressibilite, conduction, electronique, gravitation};
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn lever<T>(r: Result<T, String>) -> PyResult<T> {
    r.map_err(PyValueError::new_err)
}

#[pymodule]
mod polytech_core {
    use super::{Complex64, Defauts, PyResult, lever};
    use pyo3::prelude::pyfunction;

    #[pyfunction]
    fn force_gravitationnelle(constante: f64, m1: f64, m2: f64, r: f64) -> PyResult<f64> {
        lever(super::gravitation::force_gravitationnelle(
            constante, m1, m2, r,
        ))
    }

    #[pyfunction]
    fn rayon_critique(conductivite: f64, echange: f64) -> PyResult<f64> {
        lever(super::conduction::rayon_critique(conductivite, echange))
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extrait une sequence Python en Vec, jamais en tranche"
    )]
    #[pyfunction]
    fn valider_bandes(facteurs: Vec<f64>, coupures: Vec<f64>) -> PyResult<()> {
        lever(super::bandes::valider_bandes(&facteurs, &coupures))
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extrait une sequence Python en Vec, jamais en tranche"
    )]
    #[pyfunction]
    fn moyenne_par_bandes(facteurs: Vec<f64>, parts: Vec<f64>) -> PyResult<f64> {
        lever(super::bandes::moyenne_par_bandes(&facteurs, &parts))
    }

    #[pyfunction]
    fn proches(valeur: Complex64, reference: Complex64, tolerance: f64) -> PyResult<bool> {
        lever(super::comparaison::proches(valeur, reference, tolerance))
    }

    #[pyfunction]
    fn eloignees(valeur: Complex64, reference: Complex64, d_au_moins: f64) -> PyResult<bool> {
        lever(super::comparaison::eloignees(valeur, reference, d_au_moins))
    }

    #[pyfunction]
    fn proches_relativement(
        valeur: Complex64,
        reference: Complex64,
        tolerance: f64,
    ) -> PyResult<bool> {
        lever(super::comparaison::proches_relativement(
            valeur, reference, tolerance,
        ))
    }

    #[pyfunction]
    fn en_degres(radians: f64) -> f64 {
        super::comparaison::en_degres(radians)
    }

    #[pyfunction]
    fn variation_relative_a_l_arret(mach: f64, gamma: f64) -> PyResult<f64> {
        lever(super::compressibilite::variation_relative_a_l_arret(
            mach, gamma,
        ))
    }

    #[pyfunction]
    fn capacite_de_liaison(resistance: f64, frequence_de_coupure: f64) -> PyResult<f64> {
        lever(super::electronique::capacite_de_liaison(
            resistance,
            frequence_de_coupure,
        ))
    }

    #[pyfunction]
    fn decalage_du_non_inverseur(
        tension_de_decalage: f64,
        courant_plus: f64,
        courant_moins: f64,
        vers_la_masse: f64,
        contre_reaction: f64,
        source: f64,
    ) -> PyResult<f64> {
        let d = Defauts {
            tension_de_decalage,
            courant_plus,
            courant_moins,
        };
        lever(super::electronique::decalage_du_non_inverseur(
            d,
            vers_la_masse,
            contre_reaction,
            source,
        ))
    }

    #[pyfunction]
    fn decalage_de_l_inverseur(
        entree: f64,
        tension_de_decalage: f64,
        courant_plus: f64,
        courant_moins: f64,
        resistance_entree: f64,
        contre_reaction: f64,
    ) -> PyResult<f64> {
        let d = Defauts {
            tension_de_decalage,
            courant_plus,
            courant_moins,
        };
        lever(super::electronique::decalage_de_l_inverseur(
            entree,
            d,
            resistance_entree,
            contre_reaction,
        ))
    }
}
