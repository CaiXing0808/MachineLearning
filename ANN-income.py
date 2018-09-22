
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier


# In[14]:


dirt = "C:/Users/Xing/Desktop/ML/D3/"
adult = pd.read_csv(dirt + 'adult.csv')


# In[42]:


adult.head()


# In[43]:


adult.describe()


# In[44]:


adult.info()


# In[45]:


adult.dtypes


# In[15]:


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


# In[5]:


adult.dtypes


# In[48]:


adult.describe()


# In[16]:


Xcols=['workclass','education','marital-status','occupation','relationship','race','sex','native-country','age',
      'fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
Ycol = ['income']
Xcols=adult[Xcols]
Xcols=pd.get_dummies(Xcols)
Xcols.dtypes


# In[17]:


Y=adult[Ycol]


# In[18]:


X = pd.concat([Xcols], axis=1)


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.3, random_state=42)


# In[22]:


## ANN with TensorFlow
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[30,10], n_classes=10, feature_columns=feature_columns)
dnn_clf.fit(x=X_train, y=Y_train, batch_size=50, steps=40000)


# In[12]:


# ANN with TensorFlow
#
from sklearn import metrics
pred4 = dnn_clf.predict(X_test)
print (metrics.confusion_matrix(Y_test, pred4))
print ("Accuracy Score = ", metrics.accuracy_score(Y_test, pred4))


# In[23]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[38]:


from sklearn.neural_network import MLPClassifier
from sklearn import metrics

ANN_income = MLPClassifier(hidden_layer_sizes = (4, 4, 4),
                           max_iter=1000,
                           activation = 'relu')
ANN_income.fit(X_train, Y_train.values.ravel())
#score = ANN_income.score(X_test, Y_test)
#print(score)

pred3 = ANN_income.predict(X_test)
print(metrics.confusion_matrix(Y_test, pred3))
print("Accuracy Score =", metrics.accuracy_score(Y_test, pred3))


# In[34]:


from sklearn.metrics import classification_report
print (classification_report(Y_test, pred3))


# In[12]:


feature_names=X.columns
print (feature_names)


# In[18]:


ada_income = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=10),
                               n_estimators=300, algorithm="SAMME.R", learning_rate=0.1)
ada_income.fit(X_train, Y_train)
pred2 = ada_income.predict(X_test)
print(metrics.confusion_matrix(Y_test, pred2))
print("Accuracy Score =", metrics.accuracy_score(Y_test, pred2))


# In[ ]:


##
AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1.0, n_estimators=300, random_state=30)


# In[49]:


from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn import metrics

ada_income = (ada_Boosts.predict(X_test))

print(metrics.confusion_matrix(Y_test, ada_income))
print("Accuracy Score =", metrics.accuracy_score(Y_test, ada_income))

#print ("Accuracy score on testing data:{:.4f}".format(
#accuracy_score(Y_test, ada_income)))
print ("F-score on testing data:{:.4f}".format(fbeta_score(
Y_test, ada_income, beta=0.5)))



# In[97]:


from sklearn import metrics
adult_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, max_leaf_nodes=5)
adult_tree.fit(X_train, Y_train)
predictions = adult_tree.predict(X_test)
print(metrics.confusion_matrix(Y_test, predictions))
print("Accuracy Score =", metrics.accuracy_score(Y_test, predictions))


#adult_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, max_leaf_nodes=5)


# In[54]:


###### n-fold cross-validation ###################
from sklearn.model_selection import cross_val_score
cross_val_score(ada_income, X_train, Y_train, cv=5, scoring = "accuracy")

