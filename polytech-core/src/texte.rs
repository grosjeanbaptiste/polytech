//! `repr` d'un flottant tel que Python l'ecrit, pour que les messages d'erreur
//! des noyaux Rust soient ceux de la facade (`f"{x}"`).

#[must_use]
pub fn repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.into();
    }
    let sci = format!("{x:e}");
    let (mantisse, exposant) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exposant: i32 = exposant.parse().unwrap_or(0);
    if (-4..16).contains(&exposant) {
        let plat = format!("{x}");
        return if plat.contains('.') {
            plat
        } else {
            format!("{plat}.0")
        };
    }
    let signe = if exposant < 0 { '-' } else { '+' };
    format!("{mantisse}e{signe}{:02}", exposant.abs())
}

#[must_use]
pub fn liste(xs: &[f64]) -> String {
    let corps: Vec<String> = xs.iter().map(|x| repr(*x)).collect();
    format!("[{}]", corps.join(", "))
}

#[cfg(test)]
mod tests {
    use super::{liste, repr};

    #[test]
    fn repr_suit_python() {
        assert_eq!(repr(1.0), "1.0");
        assert_eq!(repr(-2.5), "-2.5");
        assert_eq!(repr(0.0), "0.0");
        assert_eq!(repr(0.0001), "0.0001");
        assert_eq!(repr(1e-5), "1e-05");
        assert_eq!(repr(1.5e-7), "1.5e-07");
        assert_eq!(repr(1e15), "1000000000000000.0");
        assert_eq!(repr(1e16), "1e+16");
        assert_eq!(repr(1.2345e100), "1.2345e+100");
        assert_eq!(repr(f64::NAN), "nan");
        assert_eq!(repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(repr(6.6743e-11), "6.6743e-11");
    }

    #[test]
    fn liste_suit_python() {
        assert_eq!(liste(&[0.5, 2.0]), "[0.5, 2.0]");
        assert_eq!(liste(&[]), "[]");
    }
}
