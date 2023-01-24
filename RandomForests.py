from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


skullsForest = RandomForestClassifier(n_estimators=10, criterion='entropy')


my_data = pd.read_csv('dataset.csv', delimiter=',')


Column1 = list(my_data.columns.values)[1:9]


A = my_data.drop(my_data.columns[[0, 1, 2]], axis=1).values


Column2 = my_data['EPoch'].unique().tolist()
B = my_data['EPoch']


A_trainset, A_testset, B_trainset, B_testset = train_test_split(A, B, test_size=0.3, random_state=9)

skullsForest.fit(A_trainset, B_trainset)
Forest = skullsForest.predict(A_testset)


print(Forest)
print(B_testset)
print("The Accuracy for RandomForests is : ", metrics.accuracy_score(B_testset, Forest))

