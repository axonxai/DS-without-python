use rustlearn::prelude::*;
use rustlearn::ensemble::random_forest::Hyperparameters;
use rustlearn::datasets::iris;
use rustlearn::trees::decision_tree;


fn main() {
    // Load IRIS data
    let (data, target) = iris::load_data();

    let mut tree_params = decision_tree::Hyperparameters::new(data.cols());
    tree_params.min_samples_split(10)
    .max_features(4);

    let mut model = Hyperparameters::new(tree_params, 10)
    .one_vs_rest();

    model.fit(&data, &target).unwrap();

    // Optionally serialize and deserialize the model

    // let encoded = bincode::rustc_serialize::encode(&model,
    //                                               bincode::SizeLimit::Infinite).unwrap();
    // let decoded: OneVsRestWrapper<RandomForest> = bincode::rustc_serialize::decode(&encoded).unwrap();

    let prediction = model.predict(&data).unwrap();
    println!("prediction {:?}", prediction)
}
