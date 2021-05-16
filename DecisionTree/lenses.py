from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pydotplus

if __name__ == '__main__':
    with open('lenses.txt') as fr:
        lenses = [example.strip().split('\t') for example in fr.readlines()]
    lenses_target = []
    #print(lenses)
    for lenses_example in lenses:
        lenses_target.append(lenses_example[-1])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for label in lensesLabels:
        for lenses_example in lenses:
            lenses_list.append(lenses_example[lensesLabels.index(label)])
        lenses_dict[label] = lenses_list
        lenses_list = []
    
    #print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(lenses_pd.values.tolist(),lenses_target) 
    print(clf.predict([[1,1,1,0]]))
""" if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [example.strip().split('\t') for example in fr.readlines()]
    print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(lenses,lensesLabels) """