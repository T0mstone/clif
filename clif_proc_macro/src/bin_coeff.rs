fn bin_coeff(n: usize, k: usize) -> usize {
    if k == 0 || k == n {
        1
    } else {
        bin_coeff(n - 1, k - 1) + bin_coeff(n - 1, k)
    }
}
