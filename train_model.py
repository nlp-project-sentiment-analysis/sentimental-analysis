print("Import packages")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegressionCV
import warnings

warnings.filterwarnings("ignore")

print("Done import packages\n")

print("Fetch dataset")
df = pd.read_csv("./datasets/tokens.csv")
print("Done fetch dataset\n")

print("Splitting dataset")
X_train, X_test, Y_train, Y_test = train_test_split(
    df["lemmats"].apply(lambda x: "".join(x)),
    df["sentiment"],
    test_size=0.3,
    random_state=60,
    shuffle=True,
    stratify=df["sentiment"],
)
print("Done splitting dataset\n")


def model_result(clf, X_train, Y_train, X_test, Y_test):
    print(f"Start {clf}")
    start = time.time()
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf)])  # min_df=0.001
    model.fit(X_train, Y_train)
    test_predict = model.predict(X_test)

    train_accuracy = round(model.score(X_train, Y_train) * 100)
    test_accuracy = round(accuracy_score(test_predict, Y_test) * 100)
    report = classification_report(test_predict, Y_test, target_names=["pos", "neg"])
    cm = confusion_matrix(Y_test, test_predict)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["pos", "neg"])
    disp.plot()
    plt.title(f"Confusion matrix for {clf}")
    plt.savefig(f"./screenshot/{clf}.png")

    end = time.time()

    with open("results.txt", "a") as f:
        f.write(f"\n\n=====================================================\n\n")
        f.write(f"Clf: {clf}\t\t\tTime taken: {end-start}\n")
        f.write(f"Train Accuracy Score: {train_accuracy}%\n")
        f.write(f"Test Accuracy Score: {test_accuracy}%\n")
        f.writelines(f"Report: \n{report}")
        f.writelines(f"Confusion matrix: \n{cm}")
    # plt.show()
    print(f"Done {clf}\n")
    return model


clfs = [
    MultinomialNB(),
    BernoulliNB(),
    LogisticRegressionCV(cv=5,max_iter=10000, random_state=24, n_jobs=-1),
    # LogisticRegressionCV(cv=3, max_iter=10000, n_jobs=-1),
]

for clf in clfs:
    model_result(clf, X_train, Y_train, X_test, Y_test)
