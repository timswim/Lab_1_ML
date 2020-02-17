import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_source = "AB_NYC_2019.csv"
data = pd.read_csv(data_source)
print(data.info(), end="\n-------------\n")
# data.fillna("nan") # Не работает


def task_1(): # Сколько записей в базе

    print(data.info(), end="\n-------------\n")
    print(len(data))


def task_2(): # Построение гистограмм
    data.drop(['name', 'host_name', 'last_review','neighbourhood_group', 'neighbourhood', 'room_type'], axis='columns', inplace=True)  # Удаление проблемных столбцов
    groups = list(data.columns)
    for group in groups:
        plt.figure(num=group)
        plt.hist(x=data[group], bins=None)
        plt.xlabel('value')
        plt.ylabel('quantity')
    plt.show()

# Удаление проблемных данных

data = data.drop(data[data["price"] > 1200].index)
data = data.drop(data[data["minimum_nights"] > 100].index)
data = data.drop(data[data["number_of_reviews"] > 300].index)
data = data.drop(data[data["reviews_per_month"] > 15].index)


def task_4(): # Построение карты по районам
    fig, ax = plt.subplots()
    colors = {'Brooklyn': 'red', 'Manhattan': 'blue', 'Queens': 'green', 'Staten Island': 'yellow', 'Bronx': 'black'}
    grouped = data.groupby('neighbourhood_group')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='latitude', y='longitude', label=key, color=colors[key])
    plt.show()


def corr(): # task_5 Матрица корреляции
    data['neighbourhood_group_id'] = pd.factorize(data.neighbourhood_group)[0]
    data['room_type_id'] = pd.factorize(data.room_type)[0]
    data['neighbourhood_id'] = pd.factorize(data.neighbourhood)[0]
    data['host_name_id'] = pd.factorize(data.host_name)[0]
    corr = data.corr()
    sns.heatmap(corr,annot=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()


def task_6():
    data.plot.scatter(x='latitude',
                    y = 'longitude',
                    c = 'price',
                    colormap = 'viridis')
    plt.show()


def task_7():
    a = ''
    for num in range(len(data)):
        try:
            a += data.loc[num,'name'] + " "
        except KeyError:
            continue
        except TypeError:
            continue
    d = pd.Series(a.lower().split()).value_counts()[:25]
    print(d)


task_1()

#plt.figure(num='longitude')
#plt.hist(x=data['longitude'], bins=None)
#plt.show()
