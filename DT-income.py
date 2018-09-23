import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import pydot
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

dirt = "../ML/D3/"

adult = pd.read_csv(dirt + 'adult.csv')
#adult.head()
#adult.describe()
#adult.info()
#adult.dtypes
adult.dropna(inplace=True)
adult['workclass']=adult['workclass'].astype('category')
adult['education']=adult['education'].astype('category')
adult['marital-status']=adult['marital-status'].astype('category')
adult['occupation']=adult['occupation'].astype('category')
adult['relationship']=adult['relationship'].astype('category')
adult['race']=adult['race'].astype('category')
adult['sex']=adult['sex'].astype('category')
adult['native-country']=adult['native-country'].astype('category')
adult['income']=adult['income'].astype('category')
#adult.dtypes
#adult.describe()
#adult.shape
Xcols=['workclass','education','marital-status','occupation','relationship','race','sex','native-country','age',
      'fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
Ycol = ['income']
Xcols=adult[Xcols]
Xcols=pd.get_dummies(Xcols)
Xcols.dtypes
Y=adult[Ycol]
X = pd.concat([Xcols], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state=42)
feature_names=X.columns
#print (feature_names)
###############################3
#################################

from sklearn import metrics
adult_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=18)
adult_tree.fit(X_train, Y_train)
predictions = adult_tree.predict(X_test)
print(metrics.confusion_matrix(Y_test, predictions))
print("Accuracy Score =", metrics.accuracy_score(Y_test, predictions))

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    train_errors, val_errors = [], []
    for m in range (1, int(len(X_train)),100):
        model.fit(X_train[:m], Y_train[:m])
        Y_train_predict = model.predict(X_train[:m])
        Y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(Y_train_predict, Y_train[:m]))
        val_errors.append(mean_squared_error(Y_test_predict, Y_test))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.grid()
    plt.ylim(0,1)
    plt.xlabel('Train data size')
    plt.ylabel('MSE')
    plt.legend(loc="best",prop={'size': 7},shadow=False, fancybox=False)
    return plt
plt=plot_learning_curves(adult_tree, X, Y)
plt.title("Learning Curves")
plt.savefig("Learning Curve")
###### 10-fold cross-validation ###################
from sklearn.model_selection import cross_val_score
cross_val_score(adult_tree, X_train, Y_train, cv=5, scoring = "accuracy")

from sklearn.tree import export_graphviz

## Decision Tree Visualization
export_graphviz(
   adult_tree,
   out_file="tree_income.dot",
   feature_names=feature_names,
   class_names=('0','1','2','3','4','5'),
   rounded=True,
   filled=True)

import matplotlib.pyplot as plt
importances = adult_tree.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

#for f in range(X.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
print(X.shape[1])
# Plot the feature importances 
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
#plt.xlim([-1, 10])
plt.savefig("dt-income.png")
(graph,) = pydot.graph_from_dot_file('tree_income.dot')
graph.write_png('2tree_income.png')

#def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                        n_jobs=5, train_sizes=np.linspace(.01, 1.0, 20)):
#    """
#    Generate a simple plot of the test and training learning curve.
#
#    Parameters
#    ----------
#    estimator : object type that implements the "fit" and "predict" methods
#        An object of that type which is cloned for each validation.
#
#    title : string
#        Title for the chart.
#
#    X : array-like, shape (n_samples, n_features)
#        Training vector, where n_samples is the number of samples and
#        n_features is the number of features.
#
#    y : array-like, shape (n_samples) or (n_samples, n_features), optional
#        Target relative to X for classification or regression;
#        None for unsupervised learning.
#
#    ylim : tuple, shape (ymin, ymax), optional
#        Defines minimum and maximum yvalues plotted.
#
#    cv : int, cross-validation generator or an iterable, optional
#        Determines the cross-validation splitting strategy.
#        Possible inputs for cv are:
#          - None, to use the default 3-fold cross-validation,
#          - integer, to specify the number of folds.
#          - An object to be used as a cross-validation generator.
#          - An iterable yielding train/test splits.
#
#        For integer/None inputs, if ``y`` is binary or multiclass,
#        :class:`StratifiedKFold` used. If the estimator is not a classifier
#        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
#
#        Refer :ref:`User Guide <cross_validation>` for the various
#        cross-validators that can be used here.
#
#    n_jobs : integer, optional
#        Number of jobs to run in parallel (default 1).
#    """
#    plt.figure()
#    plt.title(title)
#    if ylim is not None:
#        plt.ylim(*ylim)
#    plt.xlabel("Training examples")
#    plt.ylabel("Score")
#    train_sizes, train_scores, test_scores = learning_curve(
#        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    plt.grid()
#
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#             label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation score")
#
#    plt.legend(loc="best")
#    return plt
#title = "Learning Curves (DT)"
#
#cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
#
#estimator = DecisionTreeClassifier()
##plot_learning_curve(estimator, title, X, Y, ylim=(0.5, 1.01), cv=cv, n_jobs=5)
#
## SVC is more expensive so we do a lower number of CV iterations:
#
##plt.show()
#
