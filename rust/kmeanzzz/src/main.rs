use cogset::{Euclid, Kmeans};

fn main() {
    let data = [Euclid([0.0, 0.0]),
        Euclid([1.0, 0.5]),
        Euclid([0.2, 0.2]),
        Euclid([0.3, 0.8]),
        Euclid([0.0, 1.0])];
    let k = 3;

    let kmeans = Kmeans::new(&data, k);

    println!("{:?}", kmeans.clusters());
}
