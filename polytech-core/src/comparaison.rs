//! Comparaisons scalaires (reels ou complexes) a une tolerance.

use crate::texte::repr;
use num_complex::Complex64;

fn tolerance_valide(tolerance: f64) -> Result<f64, String> {
    if tolerance.is_nan() || tolerance < 0.0 {
        return Err(format!(
            "tolerance invalide : {} (un ecart se mesure par un reel >= 0)",
            repr(tolerance)
        ));
    }
    Ok(tolerance)
}

/// `|valeur - reference| <= tolerance`.
///
/// # Errors
/// Tolerance NaN ou negative.
pub fn proches(valeur: Complex64, reference: Complex64, tolerance: f64) -> Result<bool, String> {
    Ok((valeur - reference).norm() <= tolerance_valide(tolerance)?)
}

/// `|valeur - reference| > d_au_moins`.
///
/// # Errors
/// Ecart NaN ou negatif.
pub fn eloignees(valeur: Complex64, reference: Complex64, d_au_moins: f64) -> Result<bool, String> {
    Ok((valeur - reference).norm() > tolerance_valide(d_au_moins)?)
}

/// `|valeur - reference| <= tolerance |reference|`.
///
/// # Errors
/// Tolerance NaN ou negative.
pub fn proches_relativement(
    valeur: Complex64,
    reference: Complex64,
    tolerance: f64,
) -> Result<bool, String> {
    Ok((valeur - reference).norm() <= tolerance_valide(tolerance)? * reference.norm())
}

#[must_use]
pub fn en_degres(radians: f64) -> f64 {
    radians.to_degrees()
}

#[cfg(test)]
mod tests {
    use super::{eloignees, en_degres, proches, proches_relativement};
    use num_complex::Complex64 as C;

    fn r(x: f64) -> C {
        C::new(x, 0.0)
    }

    #[test]
    fn proches_inclut_la_borne() {
        assert!(proches(r(1.0), r(1.5), 0.5).unwrap());
        assert!(!proches(r(1.0), r(1.6), 0.5).unwrap());
        assert!(proches(r(2.0), r(2.0), 0.0).unwrap());
    }

    #[test]
    fn complexes_par_le_module() {
        assert!(proches(C::new(3.0, 4.0), r(0.0), 5.0).unwrap());
        assert!(!proches(C::new(3.0, 4.0), r(0.0), 4.9).unwrap());
    }

    #[test]
    fn eloignees_exclut_la_borne() {
        assert!(!eloignees(r(1.0), r(1.5), 0.5).unwrap());
        assert!(eloignees(r(1.0), r(1.6), 0.5).unwrap());
    }

    #[test]
    fn relativement_et_reference_nulle() {
        assert!(proches_relativement(r(101.0), r(100.0), 0.01).unwrap());
        assert!(!proches_relativement(r(102.0), r(100.0), 0.01).unwrap());
        assert!(proches_relativement(r(0.0), r(0.0), 0.5).unwrap());
        assert!(!proches_relativement(r(1e-9), r(0.0), 0.5).unwrap());
    }

    #[test]
    fn tolerance_invalide() {
        let e = proches(r(1.0), r(1.0), -1.0).unwrap_err();
        assert_eq!(
            e,
            "tolerance invalide : -1.0 (un ecart se mesure par un reel >= 0)"
        );
        assert!(proches(r(1.0), r(1.0), f64::NAN).is_err());
        assert!(eloignees(r(1.0), r(1.0), -0.1).is_err());
        assert!(proches_relativement(r(1.0), r(1.0), -0.1).is_err());
    }

    #[test]
    fn degres() {
        assert!((en_degres(std::f64::consts::PI) - 180.0).abs() < 1e-12);
        assert!(en_degres(0.0).abs() < f64::MIN_POSITIVE);
    }
}
