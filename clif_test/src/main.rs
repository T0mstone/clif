use clif_proc_macro::register_multivector;

register_multivector!(pub 3; pub 1, pub 2, pub 3);

fn main() {
    let a = multivector3!([1] => 1.0);

    let rot = multivector3!([1 2] => -1.0);

    let b = a.rotate(rot, std::f64::consts::PI / 2.0);

    println!("{:?}", b);
}
