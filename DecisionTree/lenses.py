from sklearn import tree
import pandas as pd

if __name__ == '__main__':
    with open('lenses.txt') as fr:
        lenses = [example.strip().split('\t') for example in fr.readlines()]
    lenses_target = []
    for lenses_example in lenses:
        lenses_target.append(each[-1])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for label in lensesLabels:
        for lenses_example in lenses:
            

""" if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [example.strip().split('\t') for example in fr.readlines()]
    print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(lenses,lensesLabels) """