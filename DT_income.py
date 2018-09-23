import pandas as pd
import numpy as np
import traceback
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pydot
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

def getData1():
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
    #print (feature_names)
    feature_names=X.columns
    print("get dataset adult!")
    return X,Y,feature_names
###############################3
def plot_learning_curves(model, X, Y,stepsize):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    train_scores, val_scores = [], []
    for m in range (1, int(len(X_train)),stepsize):
        model.fit(X_train[:m], Y_train[:m])
        Y_train_predict = model.predict(X_train[:m])
        Y_test_predict = model.predict(X_test)
        train_scores.append(metrics.accuracy_score(Y_train[:m], Y_train_predict))
        val_scores.append(metrics.accuracy_score(Y_test, Y_test_predict))
    plt.plot(np.sqrt(train_scores), "r-+", linewidth=2, label="Training")
    plt.plot(np.sqrt(val_scores), "b-", linewidth=3, label="Validation")
    plt.grid()
    plt.ylim(0.5,1.01)
    plt.xlabel('Train data size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc="best",prop={'size': 10},shadow=False, fancybox=False)
    plt.title("Learning Curves")
    return plt
#################################
if __name__ == "__main__":
  try: 
    ######################################### #########################################
    #        Data Preparation 
    ##################################################################################
    X,Y,feature_names=getData1()
    
    ## ################################################################################
    # Intial Train/test split        
    ##################################################################################
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=42)
    ##################################################################################
    ##   Learning Curve 
    ##################################################################################
    adult_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=len(X.columns))
    adult_tree.fit(X_train, Y_train)
    predictions = adult_tree.predict(X_test)
    plt=plot_learning_curves(adult_tree, X, Y,100)
    plt.savefig("DT_adult_LearningCurve")
    ##################################################################################
    ## Tuning and Pruning
    #######################################################################################
    fmax_depth = 4
    fmax_leaf_nodes = 5
    ftest_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ftest_size,random_state=42)
    adult_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=fmax_depth, max_leaf_nodes=fmax_leaf_nodes)
    adult_tree.fit(X_train, Y_train)
    
    #######################################################################################
    # 10-fold cross-validation
    #######################################################################################
    predictions = adult_tree.predict(X_test)
    print(metrics.confusion_matrix(Y_test, predictions))
    print("Accuracy Score =", metrics.accuracy_score(Y_test, predictions))
    print(cross_val_score(adult_tree, X_train, Y_train, cv=5, scoring = "accuracy"))
    
    #######################################################################################
    ## Decision Tree Visualization
    #######################################################################################
    export_graphviz(adult_tree, out_file="tree_income.dot", feature_names=feature_names, 
    class_names=('0','1','2','3','4','5'), rounded=True,filled=True)
    (graph,) = pydot.graph_from_dot_file('tree_income.dot')
    graph.write_png('dt_income.png')
    #######################################################################################
    ## Feature ranking Visualization
    #######################################################################################
    importances = adult_tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    #print("Feature ranking:")
    #for f in range(X.shape[1]):
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances 
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],color="r", align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, 10])
    plt.savefig("FI_income.png")




  except:
    traceback.print_exc()
