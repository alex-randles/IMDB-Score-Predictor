import pandas
import math
import numpy
import scipy
from collections import Counter


def split_data(df):

    # Split data randomly 80:20
    training = df.sample(frac=0.8, random_state=200)
    testing = df.drop(training.index)

    return training, testing


def mean_normalisation(df):
    df = (df-df.mean())/df.std()

    return df


def min_max_normalisation(df):
    df = (df-df.min())/(df.max()-df.min())

    return df


def reorder_to_put_imdb_score_at_front(df):
    cols = list(df)
    print(cols)
    cols.insert(0, cols.pop(cols.index('imdb_cat')))
    df = df.ix[:, cols]

    return df


def encode_categorical_attributes(train, test):
    #test = pandas.get_dummies(test, columns=["country", "content_rating"])
    #train = pandas.get_dummies(train, columns=["country", "content_rating"])

    test["country"] = test["country"].astype('category')
    test["country"] = test["country"].cat.codes
    test["content_rating"] = test["content_rating"].astype('category')
    test["content_rating"] = test["content_rating"].cat.codes

    train["country"] = train["country"].astype('category')
    train["country"] = train["country"].cat.codes
    train["content_rating"] = train["content_rating"].astype('category')
    train["content_rating"] = train["content_rating"].cat.codes

    print(test["content_rating"])
    return train, test


def euclidean_distance(x1, x2):
    d = 0.0
    for i in range(1, len(x1)):
        d += pow((float(x1[i] - float(x2[i]))), 2)
        d = math.sqrt(d)

    return d


def knn(train_data, test_data, k_value):
    new_test_data = []
    for i in test_data:
        euclid_dist = []
        knn = []
        terrible = 0
        bad = 0
        average = 0
        good = 0
        very_good = 0
        for j in train_data:
            dist = euclidean_distance(i, j)
            euclid_dist.append((j[0], dist))
            euclid_dist.sort(key=lambda x: x[1])
            knn = euclid_dist[:k_value]
            for k in knn:
                if k[0] == "terrible":
                    terrible += 1
                elif k[0] == "bad":
                    bad += 1
                elif k[0] == "average":
                    average += 1
                elif k[0] == "good":
                    good += 1
                else:
                    very_good += 1

        if max(terrible, bad, average, good, very_good) == terrible:
            new_test_data.append(numpy.append(i, "terrible"))
        elif max(terrible, bad, average, good, very_good) == bad:
            new_test_data.append(numpy.append(i, "bad"))
        elif max(terrible, bad, average, good, very_good) == average:
            new_test_data.append(numpy.append(i, "average"))
        elif max(terrible, bad, average, good, very_good) == good:
            new_test_data.append(numpy.append(i, "good"))
        else:
            new_test_data.append(numpy.append(i, "very_good"))

    return new_test_data


def accuracy(test_data):
    correct = 0
    for i in test_data:
        if i[0] == i[-1]:
            correct += 1

    acc = float(correct)/len(test_data) * 100
    return acc


def categorise_score(df):
    print(df)
    df["imdb_cat"] = ""
    for i in df.index:
        if df.at[i, "imdb_score"] <= 0.2:
            df.at[i, "imdb_cat"] = "terrible"
        elif df.at[i, "imdb_score"] <= 0.4:
            df.at[i, "imdb_cat"] = "bad"
        elif df.at[i, "imdb_score"] <= 0.6:
            df.at[i, "imdb_cat"] = "average"
        elif df.at[i, "imdb_score"] <= 0.8:
            df.at[i, "imdb_cat"] = "good"
        else:
            df.at[i, "imdb_cat"] = "very good"

    df = df.drop("imdb_score", 1)
    return df


def main():
    # Read in dataset
    df = pandas.read_csv('cleaned_dataset.csv', index_col=None)

    training, testing = split_data(df)

    training2, testing2 = encode_categorical_attributes(training, testing)

    training2, testing2 = min_max_normalisation(training2), min_max_normalisation(testing2)

    training2, testing2 = training2.drop("Unnamed: 0", 1), testing2.drop("Unnamed: 0", 1)

    training2, testing2 = categorise_score(training2), categorise_score(testing2)

    training2, testing2 = reorder_to_put_imdb_score_at_front(training2), reorder_to_put_imdb_score_at_front(testing2)

    training2matrix, testing2matrix = training2.as_matrix(), testing2.as_matrix()

    print(training2matrix)
    print(testing2matrix)
    for k in range(1, 21):
        new_test_data = knn(training2matrix, testing2matrix, k)
        print("k = " + str(k) + " Accuracy = " + str(accuracy(new_test_data)) + "%")


if __name__ == '__main__':
    main()
