extern crate proc_macro;

//mod kvec_template; // for testing
//mod template; // for testing

use proc_macro::{TokenStream, TokenTree};
use std::collections::HashMap;
use std::num::NonZeroUsize;

/*
Encoding for N Dimensional Basis Multivectors (Products of Basis Vectors in ascending Order; The Vector Basis is assumed to be Orthonormal)
1   =^= 0
e_n =^= 1 << n
a = product_i e_i; b = product_j e_j; a * b =^= a xor b // ascending order is implied; factors of (-1) computed separately

KEEP IN MIND: basis vectors are numbered from 0 (except in explicitly user-facing code, i.e. the macro and the Debug impl)
*/
mod repr;

mod repr_ext {
    use super::repr::*;

    fn fac_neg1_row(e_x: usize, n: usize) -> Vec<bool> {
        (0..n).map(|e_y| fac_neg1(e_x, e_y)).collect()
    }

    pub fn fac_neg1_mat_flat(n: usize) -> Vec<bool> {
        (0..n).map(|e_x| fac_neg1_row(e_x, n)).flatten().collect()
    }
}

const TEMPLATE: &str = include_str!("template.rs");
const K_VEC_TEMPLATE: &str = include_str!("kvec_template.rs");
const REPR_SRC_CODE: &str = include_str!("repr.rs");

#[derive(Debug, Clone, Eq, PartialEq)]
struct Config {
    dim: usize,
    multivector_public: bool,
    // k => public
    pub k_vecs: HashMap<NonZeroUsize, bool>,
}

impl Config {
    pub fn new(input: TokenStream) -> Result<Self, String> {
        let mut mv_public = false;

        let mut iter = input.into_iter();

        let tok = iter.next().ok_or("Input too short")?;

        let tok = match tok {
            TokenTree::Ident(id) if id.to_string() == "pub" => {
                mv_public = true;
                iter.next()
                    .ok_or("Input too short (Expected Integer Literal)")?
            }
            t => t,
        };

        let dim = match tok {
            TokenTree::Literal(lit) => {
                if let Ok(n) = lit.to_string().parse::<usize>() {
                    n
                } else {
                    return Err(format!("Invalid Integer Literal: `{}`", lit));
                }
            }
            t => return Err(format!("Expected Literal; found `{}`", t)),
        };

        if dim > std::mem::size_of::<usize>() * 8 {
            return Err(format!("Dimension Too High: {}", dim));
        }

        match iter.next() {
            None => {
                return Ok(Self {
                    dim,
                    multivector_public: mv_public,
                    k_vecs: HashMap::new(),
                })
            }
            Some(TokenTree::Punct(p)) if p.as_char() == ';' => (),
            Some(t) => return Err(format!("Expected `;`, found `{}`", t)),
        }

        let mut k_vecs = HashMap::new();
        let mut public = false;

        loop {
            let tok = match iter.next() {
                Some(TokenTree::Ident(id)) if id.to_string() == "pub" => {
                    public = true;
                    iter.next()
                        .ok_or("Input too short (Expected Integer Literal)")?
                }
                Some(t) => t,
                None => break,
            };

            let k = match tok {
                TokenTree::Literal(lit) => {
                    if let Ok(n) = lit.to_string().parse::<usize>() {
                        n
                    } else {
                        return Err(format!("Invalid Integer Literal: `{}`", lit));
                    }
                }
                t => return Err(format!("Expected Literal; found `{}`", t)),
            };

            if k > dim {
                return Err(format!(
                    "k Too High (larger than the dimension ({})): {}",
                    dim, k
                ));
            } else if k == 0 {
                return Err(
                    "k is 0 (Scalars of T are simply T and there will be no extra type)"
                        .to_string(),
                );
            }

            k_vecs.insert(NonZeroUsize::new(k).unwrap(), public);
            public = false;
            match iter.next() {
                Some(TokenTree::Punct(p)) if p.as_char() == ',' => (),
                Some(t) => return Err(format!("Expected `,`, found `{}`", t)),
                None => break,
            }
        }

        Ok(Self {
            dim,
            multivector_public: mv_public,
            k_vecs,
        })
    }

    fn preprocessor(&self, max_facneg1_mat_size: u128) -> Preprocessor {
        let arr_len = 1 << self.dim;

        Preprocessor::new("Z$sz", "/* @$s */")
            .with_replace_def("d", self.dim.to_string())
            .with_replace_def("d_usize", self.dim.to_string())
            .with_replace_def("len", format!("{}usize", arr_len))
            .with_replace_def("s(d)", if self.dim == 1 { "" } else { "s" })
            .with_insert_def(
                "Fac-1Mat",
                // matrix size: arrlen^2 = (2^dim)^2 = 2^(2*dim)
                // the size can get out of hand quickly: for dim=7, it is 2kiB, for dim=12: 2MiB, for dim=17: 2GiB and for dim=22: 2TiB !!!
                if 1 << (2 * self.dim as u128) <= max_facneg1_mat_size {
                    format!(
                        "{:?}[{}*i + j]",
                        repr_ext::fac_neg1_mat_flat(arr_len),
                        arr_len
                    )
                } else {
                    format!("{}\nfac_neg1(i, j)", REPR_SRC_CODE)
                },
            )
            .with_insert_def("MVpub", if self.multivector_public { "pub" } else { "" })
            .with_bool_def("dgt0", self.dim > 0)
            .with_bool_def("dgt1", self.dim > 1)
    }
}

#[derive(Default)]
struct Preprocessor {
    replace_fmt: String,
    insert_fmt: String,
    pub replace_defs: HashMap<String, String>,
    pub insert_defs: HashMap<String, String>,
    pub bool_defs: HashMap<String, bool>,
}

impl Preprocessor {
    #[inline]
    pub fn new(replace_fmt: impl Into<String>, insert_fmt: impl Into<String>) -> Self {
        Self {
            replace_fmt: replace_fmt.into(),
            insert_fmt: insert_fmt.into(),
            replace_defs: HashMap::new(),
            insert_defs: HashMap::new(),
            bool_defs: HashMap::new(),
        }
    }

    #[inline]
    pub fn replace_def(&mut self, ident: impl Into<String>, replacement: impl Into<String>) {
        self.replace_defs.insert(ident.into(), replacement.into());
    }

    #[inline]
    pub fn with_replace_def(
        mut self,
        ident: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        self.replace_def(ident, replacement);
        self
    }

    #[inline]
    pub fn comment_def(&mut self, s: impl Into<String>, replacement: impl Into<String>) {
        self.insert_defs.insert(s.into(), replacement.into());
    }

    #[inline]
    pub fn with_insert_def(mut self, s: impl Into<String>, replacement: impl Into<String>) -> Self {
        self.comment_def(s, replacement);
        self
    }

    #[inline]
    pub fn bool_def(&mut self, s: impl Into<String>, value: bool) {
        self.bool_defs.insert(s.into(), value);
    }

    #[inline]
    pub fn with_bool_def(mut self, s: impl Into<String>, value: bool) -> Self {
        self.bool_def(s, value);
        self
    }

    pub fn preprocess(&self, s: &str) -> String {
        let mut body = s.replace("//insert", "");

        body = body
            .split("\n")
            .filter(|line| !line.trim_end().ends_with("//rm"))
            .collect::<Vec<_>>()
            .join("\n");

        body = body
            .split("\n")
            .filter_map(|line| {
                if line.contains("//only_if") {
                    let mut spl = line.rsplitn(2, "//only_if");
                    let cond = spl.next().unwrap();
                    if *self.bool_defs.get(cond.trim()).unwrap_or(&false) {
                        Some(spl.next().unwrap())
                    } else {
                        None
                    }
                } else {
                    Some(line)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        body = {
            let mut spl = body.splitn(2, "//rm_all_before");

            let first = spl.next().unwrap();

            spl.next().unwrap_or(first).to_string()
        };

        body = {
            let spl = body.split("\n").collect::<Vec<_>>();

            let mut to_rm = Vec::new();

            for i in 0..spl.len() {
                if spl[i].trim() == "//rm_nextline" {
                    to_rm.push(i);
                    to_rm.push(i + 1);
                }
            }

            spl.into_iter()
                .enumerate()
                .filter_map(|(i, t)| if to_rm.contains(&i) { None } else { Some(t) })
                .collect::<Vec<_>>()
                .join("\n")
        };

        for (ident, replacement) in self.replace_defs.iter() {
            let formatted = self.replace_fmt.replace("$s", ident);
            body = body.replace(&formatted, replacement);
        }

        for (s, replacement) in self.insert_defs.iter() {
            let formatted = self.insert_fmt.replace("$s", s);
            body = body.replace(&formatted, replacement);
        }

        body
    }
}

include!("bin_coeff.rs");

fn vec_name(dim: usize, k: NonZeroUsize) -> String {
    match k.get() {
        1 => "Vector".to_string(),
        2 => "Bivector".to_string(),
        3 => "Trivector".to_string(),
        k if k == dim => "Pseudoscalar".to_string(),
        k => format!("K{}Vector", k),
    }
}

/// The macro for creating Multivector types.
///
/// Syntax: `register_multivector!($(pub)? $n $(; $($(pub)? $k),+)?);`
///
/// - `$n` generates the Multivector struct (with n dimensions), It is absolutely crucial and if you omit it, just don't use this crate
/// - `$k` generate several k-vector structs
/// - the `pub` markers mark if the corresponding struct will be `pub`
///
/// # Conditions
/// Make sure, `clif::{Field, Trig, Zero}` are present in the scope this macro is called in, otherwise it won't work
#[proc_macro]
pub fn register_multivector(input: TokenStream) -> TokenStream {
    let conf = Config::new(input).unwrap_or_else(|e| panic!("Macro Error: {}", e));

    let mut k_vecs = Vec::new();

    let mut iter = conf.k_vecs.iter().collect::<Vec<_>>();
    iter.sort_by(|a, b| a.0.cmp(b.0));

    let mut first = true;

    for (&k, &public) in iter {
        let prep = conf
            .preprocessor(512 * 8)
            .with_replace_def("k", k.to_string())
            .with_replace_def("d_choose_k", bin_coeff(conf.dim, k.get()).to_string())
            .with_replace_def("k_vector", vec_name(conf.dim, k))
            .with_insert_def("KVPub", if public { "pub" } else { "" })
            .with_insert_def("BinCoeffFn", include_str!("bin_coeff.rs"))
            .with_insert_def("PermFn", include_str!("perm.rs"))
            .with_bool_def("first", first);
        k_vecs.push(prep.preprocess(K_VEC_TEMPLATE));
        first = false;
    }

    let prep = conf
        .preprocessor(512 * 8)
        .with_insert_def("Kvecs", k_vecs.join("\n"));

    let res = prep.preprocess(TEMPLATE);

    //    let _ = std::fs::write("expansion.temp.rs", &res); // for testing

    res.parse::<TokenStream>().unwrap()
}
