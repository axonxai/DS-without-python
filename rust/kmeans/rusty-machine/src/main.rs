use rusty_machine::linalg::Matrix;
use rusty_machine::learning::k_means::KMeansClassifier;
use rusty_machine::learning::UnSupModel;
use std::fs::File;
use std::io::BufReader;
use std::error::Error;

fn read_from_file(path: &str) -> Result<Vec<f64>, Box<dyn Error>>{
    let mut reader = csv::Reader::from_path(path)?;
    let mut points: Vec<f64> = Vec::new();
    for result in reader.records() {
        let record = result?;
        points.push(record[0].parse::<f64>()?);
        points.push(record[1].parse::<f64>()?);
    }
    Ok(points)
}

fn main()   {
    // Load data
    let points = read_from_file("../../../data/clustering/clustering.csv");
    let inputs = Matrix::new(299, 2, points.unwrap());
    let test_inputs = Matrix::new(1, 2, vec![1.0, 3.5]);

    // Create model with k classes.
    let mut model = KMeansClassifier::new(4);

    // Where inputs is a Matrix with features in columns.
    model.train(&inputs).unwrap();
    println!("test {:?}", model);

    // Where test_inputs is a Matrix with features in columns.
    //let a = model.predict(&test_inputs).unwrap();
    //println!("test {}", a);


}
