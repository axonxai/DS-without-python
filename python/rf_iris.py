#!/usr/bin/python3
"""
based on: https://www.kaggle.com/tcvieira/simple-random-forest-iris-dataset
"""
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def load_dataset():
    """
    Load local Iris dataset.
    :return:    df, species_list:   pandas_data_frame, list(str())
    """
    df = pd.read_csv("../data/iris/iris.csv",
                     names=['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])
    species_list = df.species.unique()
    df.species.replace(species_list, list(range(len(species_list))), inplace=True)
    return df, species_list


def get_feature_importance(model, feature_names):
    """
    Get the most important features from the Random Forrest Model.
    :param      model:              trained model
    :param      feature_names:      list with features used
    :return:    feature_imp:        feature importance score per feature
    """
    feature_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return feature_imp


def main(argv):
    data, species_list = load_dataset()
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    features = data[feature_names]  # Features
    labels = data['species']  # Labels
    # Split dataset into training set and test set:
    feat_train, feat_test, labels_train, labels_test = train_test_split(features, labels,
                                                                        test_size=0.3)  # 70% training and 30% test

    model = RandomForestClassifier(n_estimators=100)  # Define model
    model.fit(feat_train, labels_train)  # Train model

    print("Feature Importance:\n{}\n".format(get_feature_importance(model, feature_names)))

    labels_pred = model.predict(feat_test)  # Run model
    print("Accuracy:", metrics.accuracy_score(labels_test, labels_pred))
    print("F1-Score:", metrics.f1_score(labels_test, labels_pred,
                                        average='macro'))  # Macro as there is no class imbalance
    print("ROC-AUC: ", metrics.roc_auc_score(label_binarize(labels_test, classes=list(range(len(species_list)))),
                                             label_binarize(labels_pred, classes=list(range(len(species_list)))),
                                             average='macro',  # Macro as there is no class imbalance
                                             multi_class='ovr'))  # OVR as there is no class imbalance


if __name__ == "__main__":
    main(sys.argv)
