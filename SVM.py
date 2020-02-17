import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import timeit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

data_source = "AB_NYC_2019.csv"
data = pd.read_csv(data_source)


X = data[["longitude"]]
y = data["neighbourhood_group"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
print(N_train, N_test)

svc = SVC(C=8)


a = timeit.default_timer()
svc.fit(X_train, y_train)
print("Время обучения: {}".format(timeit.default_timer()-a))

a = timeit.default_timer()
time = 0
for num in range(10):
    NB_prediction = svc.predict(X_test)
    time += timeit.default_timer() - a
    print(num, "--", timeit.default_timer() - a)
    a = timeit.default_timer()

time = time / 10
print("Время работы: {}".format(time))

print("Точнсть: {}".format(accuracy_score(NB_prediction, y_test)))
print(classification_report(NB_prediction, y_test))
'''

print("Кросс-валидация ----------------")

svc_params = {'C': range(1, 10)}
svc_grid = GridSearchCV(svc, svc_params, cv=5, n_jobs=-1, verbose=True)
svc_grid.fit(X_train, y_train)

print("Лучшее заначение параметра: {}".format(svc_grid.best_params_))
print("Лучшая точность: {}".format(svc_grid.best_score_))
tree_prediction = svc_grid.predict(X_test)
print("Точнсть: {}".format(accuracy_score(tree_prediction, y_test)))
print(classification_report(tree_prediction, y_test))
'''