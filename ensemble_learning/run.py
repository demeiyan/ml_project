
from BoostMain import AdaBoost
from RandomForestMain import Random_Forest
from sklearn.metrics import roc_curve, auc
from BoostMain import load_features_labels

if __name__ == "__main__":
    train_features, train_labels = load_features_labels('./adult_dataset/adult_train_feature.txt',
                                                    './adult_dataset/adult_train_label.txt')

    test_features, test_labels = load_features_labels('./adult_dataset/adult_test_feature.txt',
                                                      './adult_dataset/adult_test_label.txt')
    y = AdaBoost(35, train_features, train_labels, test_features, test_labels)
    fpr, tpr, _ = roc_curve(test_labels, y)
    auc_score = auc(fpr, tpr)
    print("The AUC of {} base leaner by using AdaBoost method is {}".format(35, auc_score))
    y = Random_Forest(10, train_features, train_labels, test_features, test_labels)
    fpr, tpr, _ = roc_curve(test_labels, y)
    auc_score = auc(fpr, tpr)
    print("The AUC of {} base leaner by using Random Forest method is {}".format(10, auc_score))