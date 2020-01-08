from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt
# import graphviz
import pydotplus
from six import StringIO
iris = load_iris()

#Data split
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state = 33,test_size = 0.2)
print("x_train.shape = {}".format(x_train.shape))
print("y_train.shape = {}".format(y_train.shape))
print("x_test.shape = {}".format(x_test.shape))
print("y_test.shape = {}".format(y_test.shape))
print("__________")

#Train DT with gini criterion
clf_gn = tree.DecisionTreeClassifier(criterion="gini")
clf_gn = clf_gn.fit(x_train,y_train)
clf_gn_predict = clf_gn.predict(x_test)
tree.plot_tree(clf_gn)
plt.show()
print("y_test = ",y_test)
print("DT_predict = ",y_test)
print("__________")
dot_data_gn=StringIO()
tree.export_graphviz(clf_gn, out_file=dot_data_gn,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph_gn = pydotplus.graph_from_dot_data(dot_data_gn.getvalue())
graph_gn.write_jpg("result01.jpg")

#Train DT with entropy criterion
clf_en = tree.DecisionTreeClassifier(criterion="entropy")
clf_en = clf_en.fit(x_train,y_train)
clf_en_predict = clf_en.predict(x_test)
tree.plot_tree(clf_en)
plt.show()
dot_data_en=StringIO()
tree.export_graphviz(clf_en, out_file=dot_data_en,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph_en = pydotplus.graph_from_dot_data(dot_data_en.getvalue())
graph_en.write_jpg("result02.jpg")
#Change the number of leaf nodes
for i in range(9, 2, -1):
    clf_gn = tree.DecisionTreeClassifier(max_leaf_nodes= i)
    clf_gn = clf_gn.fit(x_train, y_train)
    clf_gn_predict = clf_gn.predict(x_test)
    tree.plot_tree(clf_gn)
    plt.show()

#Change the depth of decision tree
for i in range(9, 2, -1):
    clf_gn = tree.DecisionTreeClassifier(max_depth= i)
    clf_gn = clf_gn.fit(x_train, y_train)
    clf_gn_predict = clf_gn.predict(x_test)
    tree.plot_tree(clf_gn)
    plt.show()