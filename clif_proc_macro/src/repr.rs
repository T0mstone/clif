use core::num::NonZeroUsize;

fn lowest_bit_pos(i: NonZeroUsize) -> usize {
    let mut r = 0;
    let mut i = i.get();

    loop {
        if i & 1 == 1 {
            return r;
        }

        i >>= 1;
        r += 1;
    }
}

#[inline]
pub fn fac_neg1(e_x: usize, e_y: usize) -> bool {
    if e_x == 0 || e_y == 0 {
        return false;
    }
    let b = lowest_bit_pos(unsafe { NonZeroUsize::new_unchecked(e_y) }); // get the `n` from `e_n` (0-based)
    match e_y.count_ones() {
        0 => unreachable!(),
        // now e_y is guaranteed > 0
        1 => {
            // an ugly version of `e_x & !(1 << (b + 1) - 1)` to avoid overflows
            // it extracts all bits higher than the `b`th of e_x
            let num_swaps = (e_x & !(((1 << b) - 1) + (1 << b))).count_ones();
            num_swaps % 2 != 0
        }
        _ => {
            let e_z = 1 << b;
            fac_neg1(e_x, e_z) ^ fac_neg1(e_x, e_y ^ e_z)
        }
    }
}
