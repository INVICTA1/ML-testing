import numpy as np
import pandas as pd
import scipy
import matplotlib


# a = np.array([[1, 4, 5, 8],
#               [15,16,8,89],
#               [87,65,98,56]], float)
# print('Mass: ',a)
# print('row and column:',a.shape)# возвращает количество строк и столбцов в матрице:
# print('the type of the variables stored in the array: ',a.dtype)
# print('',len(a))
# b=np.array([8,89,19,49,7,94,1189,98,4,9])
# print('changing the mass yo a given ',b.reshape((5,2)))
def series():
    ds = pd.Series([2, 4, 6, 8, 10])
    print("Pandas Series and type")
    print(ds)
    print(type(ds))
    print("Convert Pandas Series to Python list")
    print(ds.tolist())
    print(type(ds.tolist()))


def calculateSeries():
    ds1 = pd.Series([2, 4, 6, 8, 10])
    ds2 = pd.Series([1, 3, 5, 7, 9])
    ds = ds1 + ds2
    print("Add two Series:")
    print(ds)
    print("Subtract two Series:")
    ds = ds1 - ds2
    print(ds)
    print("Multiply two Series:")
    ds = ds1 * ds2
    print(ds)
    print("Divide Series1 by Series2:")
    ds = ds1 / ds2
    print(ds)


def display_data_from_dictionary():
    exam_data = {
        'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    df = pd.DataFrame(exam_data, index=labels)
    print(df)


def choose_row_more2():
    exam_data = {
        'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    df = pd.DataFrame(exam_data, index=labels)
    print("Number of attempts in the examination is greater than 2:")
    print(df[df['attempts'] > 2])


def add_row():
    d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
    df = pd.DataFrame(data=d)
    print("Original DataFrame")
    print(df)
    print('After add one row:')
    df2 = {'col1': 10, 'col2': 11, 'col3': 12}
    df = df.append(df2, ignore_index=True)
    print(df)


add_row()


def number_of_people_in_city():
    df1 = pd.DataFrame(
        {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
         'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles',
                  'Georgia', 'Georgia', 'Los Angeles']})
    g1 = df1.groupby(["city"]).size().reset_index(name='Number of people')
    print(g1)


def size_diamand():
    diamonds = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
    print("Original Dataframe:")
    print(diamonds.head())
    print("\nMultiply of length, width and depth for each cut:")
    print((diamonds.x * diamonds.y * diamonds.z).head())

number_of_people_in_city()
size_diamand()


def row_from_column():
    diamonds = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
    print("Original Dataframe:")
    print(diamonds.head())
    print("\nRows 2 through 5 and all columns :")
    print(diamonds.loc[0:2, :])