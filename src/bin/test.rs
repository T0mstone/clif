use clif::{multivector, BasisMultivector, Multivector};

fn main() {
    let mv = Multivector::from_data(vec![1, 2, 3, 4, 5, 0, 0, 0]).unwrap();

    println!("(1) mv is {:?} ({:?})", mv, mv.data());

    let mv = Multivector::build()
        .with_component(BasisMultivector::one(), 1)
        .with_component(BasisMultivector::from_vec(vec![0]), 2)
        .with_component(BasisMultivector::from_vec(vec![1]), 3)
        .with_component(BasisMultivector::from_vec(vec![2]), 4)
        .with_component(BasisMultivector::from_vec(vec![0, 1]), 5)
        .finish();

    println!("(2) mv is {:?} ({:?})", mv, mv.data());

    let mv = multivector![[] => 1, [1] => 2, [2] => 3, [3] => 4, [1 2] => 5];

    println!("(3) mv is {:?} ({:?})", mv, mv.data());
}
