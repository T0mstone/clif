fn permutation(n: usize, k: usize, mut idx: usize) -> usize {
    assert!(idx < bin_coeff(n, k), "Index too high");

    if k == 0 {
        return 0; // the assert ensures that idx is also 0
    } else if k == 1 {
        // special case for optimisation
        return idx;
    }

    let mut add = 0;

    for m in (k - 1..n).rev() {
        let c = bin_coeff(m, k - 1);
        if idx < c {
            return permutation(m, k - 1, idx) + add;
        }
        idx -= c;
        add += c;
    }

    unreachable!()
}
