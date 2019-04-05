
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib


# In[2]:


df=pd.read_csv("D:/DOWNLOADs/Final datasaet previous year.csv")


# In[3]:


df=df.dropna()


# In[4]:


df.head(5)


# In[5]:


y=df.Placeability


# In[6]:


independent=df.columns
independent=independent.delete(5)
independent=independent.delete(0)
independent=independent.delete(4)
independent


# In[7]:


df1=df[independent]
x=df[independent]


# In[8]:


#import sklearn.preprocessing as pp
#lb=pp.LabelBinarizer()
#y=lb.fit_transform(y)


# In[9]:


y[:5]              #x=x.dropna


# In[10]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x=scale.fit_transform(x)
# x=x.dropna()
# y=y.dropna()


# In[14]:


validation_size = 0.0
seed = 7
X_train,X_validation,Y_train,Y_validation= model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)


# In[15]:


seed=7
scoring = 'accuracy'


# In[16]:


# spot check algorithms

models = []
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))

# evaluate each model

results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)
    


# In[17]:


# validation_size = 0.20
# seed = 7
# X_train,X_validation,Y_train,Y_validation= model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)
# lm=sm.OLS(Y_train,Y_validation).fit()
# print(lm.summary)
# import statsmodels.api as sm
# model=sm.OLS(y,x)
# model=model.fit()
# model.summary
model = KNeighborsClassifier()
model.fit(X_train,Y_train)
out=model.predict(X_validation)
print(out)


# In[18]:


df3=pd.DataFrame()
df3["Actual"]=y
df3["Predicted"]=model.predict(x)
df3


# In[20]:


from sklearn.metrics import accuracy_score
print("Train - Accuracy :",accuracy_score(Y_train, model.predict(X_train)))
print("Test - Accuracy :",accuracy_score(Y_validation, model.predict(X_validation)))


# In[21]:


y_pred=model.predict(x)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df.CPI,df3["Actual"],color="blue")
plt.scatter(df.CPI,df3["Predicted"],color="red")
plt.xlabel("CPI")
plt.ylabel("Placeability")
plt.title("Placed or Unplaced")


# In[22]:


joblib.dump(model,"predictor.pkl",protocol=2)


# In[23]:


new_model = joblib.load("predictor.pkl")
new_model.predict(x)


# In[29]:


#take user input for house price predicion
dict1={}
for column in df1.columns:
    temp=input("enter "+column+ ":")
    dict1[column] = temp
dict1
#create a dataframe using dictionart
user_input=pd.DataFrame(dict1,index=[0],columns=df1.columns)
pred=model.predict(user_input)
if pred==1:
    print('Placed')
else:
    print('Unplaced')

