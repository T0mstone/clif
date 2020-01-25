use clif::{multivector, BasisMultivector, Multivector};

#[allow(dead_code)]
fn creation_methods() {
    let mv = Multivector::build()
        .with_component(BasisMultivector::one(), 1)
        .with_component(BasisMultivector::from_vec(vec![0]), 2)
        .with_component(BasisMultivector::from_vec(vec![1]), 3)
        .with_component(BasisMultivector::from_vec(vec![2]), 4)
        .with_component(BasisMultivector::from_vec(vec![0, 1]), 5)
        .finish();

    println!("(CR1) mv is {:?}", mv);

    let mv = multivector![[] => 1, [1] => 2, [2] => 3, [3] => 4, [1 2] => 5];

    println!("(CR2) mv is {:?}", mv);
}

#[allow(dead_code)]
fn geometric_product() {
    let a = multivector!([1 2] => 2);
    let b = multivector!([1] => -1);
    println!("(*) {:?}", a * b);
}

#[allow(dead_code)]
fn addition() {
    let a = multivector!([1] => 3, [1 2] => 4, [1 2 3] => 1);
    let b = multivector!([] => 9, [1 2] => 1, [1 2 3] => -1);
    println!("(+) {:?}", a + b);
}

#[allow(dead_code)]
fn grade_part() {
    let b = multivector!([] => 9, [1 2] => 1, [1 2 3] => -1);
    println!("(GP) {:?}", b.grade_part(2));
    println!("(GP2) {:?}", b.grade_part(3));
}

#[allow(dead_code)]
fn other_stuff() {
    let a = multivector!([3] => 7, [2] => 3, [2 1] => 1, [2 3] => 5);
    let b = multivector!([1 2] => 1);
    println!("(OS1) {:?}", a.clone().project(b.clone()));
    println!("(OS2) {:?}", a.clone().reject(b.clone()));
    println!("(OS3) {:?}", a.reflect(b));
}

fn main() {
    creation_methods();
    geometric_product();
    addition();
    grade_part();
    other_stuff();
}
