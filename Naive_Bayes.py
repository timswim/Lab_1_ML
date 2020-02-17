import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import timeit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

data_source = "AB_NYC_2019.csv"
data = pd.read_csv(data_source)


X = data[["host_id", "longitude"]]
#X = data[["host_id"]]
y = data["neighbourhood_group"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
print(N_train, N_test)

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
a = timeit.default_timer()
model.fit(X_train, y_train)
print("Время обучения: {}".format(timeit.default_timer()-a))

a = timeit.default_timer()
time = 0
for num in range(100):
    NB_prediction = model.predict(X_test)
    time += timeit.default_timer() - a
    print(num, "--", timeit.default_timer() - a)
    a = timeit.default_timer()

time = time / 100
print("Время работы: {}".format(time))


print("Точнсть: {}".format(accuracy_score(NB_prediction, y_test)))
print(classification_report(NB_prediction, y_test))

