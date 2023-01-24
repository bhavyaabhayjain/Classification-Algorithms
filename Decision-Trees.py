import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

my_data = pd.read_csv('dataset.csv', delimiter=',')
print(my_data[0:5])

Column1 = list(my_data.columns.values)[1:9]
print(Column1)

A = my_data.drop(my_data.columns[[0, 1, 2]], axis=1).values
print(A[0:5])


Column2 = my_data['EPoch'].unique().tolist()
print(Column2)

B = my_data['EPoch']
print(B[0:5])


A_trainset, A_testset, B_trainset, B_testset = train_test_split(A, B, test_size=0.3, random_state=9)


print(A_trainset.shape)
print(B_trainset.shape)
print(A_testset.shape)
print(B_testset.shape)


skullsTree = DecisionTreeClassifier(criterion='entropy')

skullsTree.fit(A_trainset, B_trainset)


Tree = skullsTree.predict(A_testset)

print(Tree[0:5])
print(B_testset[0:5])

print("The Accuracy DecisionTree is: ", metrics.accuracy_score(B_testset, Tree))