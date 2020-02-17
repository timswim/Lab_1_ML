import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz, plot_tree
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import timeit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

data_source = "AB_NYC_2019.csv"
data = pd.read_csv(data_source)


X = data[["host_id"]]

y = data["neighbourhood_group"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=2)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
print(N_train, N_test)

# -------------------------------------------------------------
# построим дерево
# -------------------------------------------------------------

# Укажем критерий качества разбиения
crit = 'gini' # "gini"
# Создаем объект-классификатор с заданными параметрами
clf_tree = DecisionTreeClassifier(criterion=crit, max_depth=10, random_state=5, min_samples_leaf=1)

# обучим дерево и выведем его глубину
a = timeit.default_timer()
clf_tree.fit(X=X_train, y=y_train)
print("Время обучения: {}".format(timeit.default_timer()-a))
print("Глубина дерева: {}".format(clf_tree.get_depth()))

a = timeit.default_timer()
time = 0
for num in range(100):
    tree_prediction = clf_tree.predict(X_test)
    time += timeit.default_timer() - a
    print(num, "--", timeit.default_timer() - a)
    a = timeit.default_timer()

time = time / 100
print("Время работы: {}".format(time))


print("Точнсть: {}".format(accuracy_score(tree_prediction, y_test)))
print(classification_report(tree_prediction, y_test))

'''
print("Кросс-валидация ----------------")

tree_params = {'max_depth': range(1,15), 'min_samples_leaf': range(1, 10)}
tree_grid = GridSearchCV(clf_tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)

print("Лучшее заначение параметра: {}".format(tree_grid.best_params_))
print("Лучшая точность: {}".format(tree_grid.best_score_))
tree_prediction = tree_grid.predict(X_test)
print("Точнсть: {}".format(accuracy_score(tree_prediction, y_test)))
print(classification_report(tree_prediction, y_test))
'''