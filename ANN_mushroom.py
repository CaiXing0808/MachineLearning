import pandas as pd
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
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
#########################################
#        Data Preparation 
#########################################
dirt = "./"
def getData2():
    mush = pd.read_csv(dirt + 'mush.csv')
    mush.dropna(inplace=True)
    mush['edible']=mush['edible'].astype('category')
    mush['cap-shape']=mush['cap-shape'].astype('category')
    mush['cap-surface']=mush['cap-surface'].astype('category')
    mush['cap-color']=mush['cap-color'].astype('category')
    mush['bruises']=mush['bruises'].astype('category')
    mush['odor']=mush['odor'].astype('category')
    mush['gill-attachment']=mush['gill-attachment'].astype('category')
    mush['gill-spacing']=mush['gill-spacing'].astype('category')
    mush['gill-size']=mush['gill-size'].astype('category')
    mush['gill-color']=mush['gill-color'].astype('category')
    mush['stalk-shape']=mush['stalk-shape'].astype('category')
    mush['stalk-root']=mush['stalk-root'].astype('category')
    mush['stalk-surface-above-ring']=mush['stalk-surface-above-ring'].astype('category')
    mush['stalk-surface-below-ring']=mush['stalk-surface-below-ring'].astype('category')
    mush['stalk-color-above-ring']=mush['stalk-color-above-ring'].astype('category')
    mush['stalk-color-below-ring']=mush['stalk-color-below-ring'].astype('category')
    mush['veil-type']=mush['veil-type'].astype('category')
    mush['veil-color']=mush['veil-color'].astype('category')
    mush['ring-number']=mush['ring-number'].astype('category')
    mush['ring-type']=mush['ring-type'].astype('category')
    mush['spore-print-color']=mush['spore-print-color'].astype('category')
    mush['population']=mush['population'].astype('category')
    mush['habitat']=mush['habitat'].astype('category')
    Xcols=['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
           'stalk-shape','stalk-root','stalk-surface-above-ring',
           'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number',
           'ring-type','spore-print-color','population','habitat']
    Ycol = ['edible']
    #######################3
    Xcols=mush[Xcols]
    Xcols=pd.get_dummies(Xcols)
    Xcols.dtypes
    X = pd.concat([Xcols], axis=1)
    feature_names=X.columns
    Y = mush[Ycol]
    print("get data mushroom!")
    return X,Y,feature_names
#############################################3
def plot_learning_curves(model, X, Y,stepsize):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    train_scores, val_scores = [], []
    for m in range (int(len(X_train)/100), int(len(X_train)),stepsize):
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
##########################################################
if __name__ == "__main__":
  try:  #########################################
    #        Data Preparation 
    #########################################
    X,Y,feature_names=getData2()

    ## #######################################
    # Intial Train/test split        
    #########################################
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=42)
    ##################################################################################
    ##   Learning Curve 
    ##################################################################################
    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier

    ANN_mush = MLPClassifier(hidden_layer_sizes = (4, 4),
                               max_iter=10000,
                               activation = 'relu')
    ANN_mush.fit(X_train, Y_train.values.ravel())

    pred = ANN_mush.predict(X_test)
    print(metrics.confusion_matrix(Y_test.values.ravel(), pred))
    print("Accuracy Score =", metrics.accuracy_score(Y_test.values.ravel(), pred))
    print(cross_val_score(ANN_mush, X_train, Y_train.values.ravel(), cv=5, scoring = "accuracy"))

    plt=plot_learning_curves(ANN_mush, X, Y.values.ravel(),100)
    plt.savefig("ANN_mush_LearningCurve")

  except:
    traceback.print_exc()
