using DecisionTree
using CSV
using DataFrames
using MLDataUtils


# Load Data
dataset = CSV.read("../data/iris/iris.csv", DataFrame; header=0)  # load dataset
rename!(dataset, ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"])  # rename columns
feature_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]
features = select(dataset, feature_names)
labels = string.(dataset.species)

# Split into train and test set
(feat_train, labels_train), (feat_test, labels_test) = stratifiedobs((features, labels), p=0.7)  # 70% training and 30% test
feat_train, feat_test = Matrix(feat_train), Matrix(feat_test)

# Run model
model = build_forest(labels_train, feat_train)
labels_pred = apply_forest(model, feat_test)
println(labels_pred)
println()

# Evaluate
accuracy = sum(labels_pred .== labels_test) / size(labels_pred, 1)
println("Accuracy:", accuracy)
