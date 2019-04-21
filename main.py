import pandas as pd
import sklearn
from sklearn import svm, preprocessing
import numpy as np


def InitializData(training_data=None, testing_data=None):
    # print(training_data["target_product_category"].unique())
    # print(training_data["shopper_segment"].unique())
    # print(training_data["delivery_time"].unique())

    # shopper_segment_dic = {'new': 0, 'casual': 0, 'heavy shopper': 0}
    # target_product_category_dic = {'gardening': 0, 'video games': 0, 'clothing': 0, 'smartphones': 0,'opera tickets': 0}
    # for index,row in training_data.iterrows():
    #     target_product_category_dic[str(row[4])] += 1
    #     shopper_segment_dic[str(row[6])] += 1

    target_product_category_dic = {'gardening': 1, 'video games': 5, 'clothing': 4, 'smartphones': 2,'opera tickets': 3}
    shopper_segment_dic = {'new': 1, 'casual': 2, 'heavy shopper': 3}
    delivery_time_dic = {'1-3 days': 4, '4-8 days': 3, '9-14 days': 2, '15+ days': 1}

    training_data['target_product_category'] = training_data['target_product_category'].map(target_product_category_dic)
    training_data['shopper_segment'] = training_data['shopper_segment'].map(shopper_segment_dic)
    training_data['delivery_time'] = training_data['delivery_time'].map(delivery_time_dic)
    training_data[training_data == np.inf] = np.nan
    training_data.fillna(training_data.mean(),inplace=True)

    testing_data['target_product_category'] = testing_data['target_product_category'].map(target_product_category_dic)
    testing_data['shopper_segment'] = testing_data['shopper_segment'].map(shopper_segment_dic)
    testing_data['delivery_time'] = testing_data['delivery_time'].map(delivery_time_dic)
    testing_data[testing_data == np.inf] = np.nan
    testing_data.fillna(testing_data.mean(),inplace=True)

    training_data = sklearn.utils.shuffle(training_data)
    Predict(training_data,testing_data)

def Predict(training_data=None, testing_data=None):
    X_train = training_data.drop("tag", axis=1).values
    X_train = preprocessing.scale(X_train)
    y_train = training_data["tag"].values

    X_test = testing_data.drop("tag", axis=1).values
    X_test = preprocessing.scale(X_test)

    clf = svm.SVR()
    clf.fit(X_train, y_train)

    f = open('result.txt', 'w')
    for X in list(X_test):
        if (clf.predict([X])[0]) >= 0.5 :
            f.write('1\n')
            p = 1
        else:
            f.write('0\n')
            p = 0
        print(f"model predicts {p}")
    f.close()

def main():
    header = ["viewed_ads", "times_visited_website", "products_in_cart", "target_product_price", "hour",
              "target_product_category", "age", "shopper_segment", "delivery_time", "tag"]

    testing_data = pd.read_csv('/Users/eranedri/Documents/ראיונות עבודה/paypal/Decision classifier model/interview_dataset_test_no_tags', sep='\t',
                                      skiprows=[0], header=None, names=header, index_col=0)
    training_data = pd.read_csv('/Users/eranedri/Documents/ראיונות עבודה/paypal/Decision classifier model/interview_dataset_train',
                                       sep='\t', skiprows=[0], header=None, names=header, index_col=0)
    InitializData(training_data,testing_data)


if __name__ == "__main__":
    main()
##eran test




