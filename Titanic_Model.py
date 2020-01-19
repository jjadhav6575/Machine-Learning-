#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from sklearn import preprocessing


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[18]:


dataset = pd.read_csv('E:\\ML Datasets\\Titanic\\train.csv')


# In[19]:


dataset.head()


# In[7]:


dataset.count()


# In[8]:


dataset.isnull().sum()


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=dataset)


# In[10]:


sns.distplot(dataset['Age'].dropna(),bins=30)


# In[11]:


dataset['Age'].hist(bins=40,color='Green')


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=dataset)


# In[21]:


val_to_replace = {'male': 0, 'female':1}
dataset['Sex'] = dataset['Sex'].map(val_to_replace)


# In[22]:


dataset.head()


# In[23]:


dataset.info()


# In[26]:


#Extracting titles from name
for row in dataset:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)          


# In[27]:


dataset.head()


# In[71]:


dataset['Title'].value_counts()


# In[28]:


title_mapping = {"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Col":3,"Mlle":3,"Major":3,"Sir":3,"Lady":3,"Ms":3,"Capt":3,"Jonkheer":3,"Don":3,"Countless":3,"Mme":3}
dataset['Title']=dataset['Title'].map(title_mapping)


# In[29]:


dataset['Title'].isnull().sum()


# In[30]:


dataset[dataset['Title'].isnull()].index.tolist()


# In[31]:


dataset.iloc[759]


# In[32]:


dataset.iloc[759,-1] = 3.0


# In[33]:


dataset.iloc[759]


# In[34]:


dataset['Title'].isnull().sum()


# In[35]:


dataset.info()


# In[36]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"),inplace=True)


# In[37]:


dataset['Age'].isnull().sum()


# In[38]:


dataset['Embarked'].isnull().sum()


# In[39]:


sns.set_style('whitegrid')
sns.countplot(x='Embarked',data=dataset)


# In[40]:


dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[41]:


dataset['Embarked'].isnull().sum()


# In[42]:


embark = pd.get_dummies(dataset['Embarked'],drop_first=True)


# In[43]:


dataset = pd.concat([dataset,embark],axis=1)


# In[44]:


dataset.head()


# In[45]:


train_df = dataset


# In[46]:


train_df.head()


# In[47]:


feature_drop = ['Name','Embarked','Ticket','Cabin']
train_df.drop(feature_drop,axis=1,inplace=True)


# In[48]:


train_df.head()


# In[49]:


train_df.isnull().sum()


# In[50]:


x = train_df.drop('Survived',axis=1)


# In[51]:


y = train_df['Survived']


# In[52]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=10)


# In[53]:


from sklearn.linear_model import LogisticRegression


# In[54]:


lg_model = LogisticRegression()
lg_model.fit(x_train,y_train)


# In[55]:


y_pred = lg_model.predict(x_test)


# In[56]:


from sklearn.metrics import confusion_matrix
acc = confusion_matrix(y_test,y_pred)


# In[57]:


acc


# In[58]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[59]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier


# In[60]:


rfc = RandomForestClassifier(n_estimators = 150,random_state=0)


# In[61]:


rfc.fit(x_train,y_train)


# In[62]:


rtc_pred = rfc.predict(x_test)


# In[63]:


accuracy=accuracy_score(y_test,rtc_pred)
accuracy


# In[64]:


from sklearn.model_selection import GridSearchCV


# In[65]:


grid_search = RandomForestClassifier()


# In[66]:


parameters = { "n_estimators" : [90,100,150,200],
              "max_depth" : [5,7,9,12,15],
              "criterion" : ['entropy','gini']    
              }


# In[67]:


grid = GridSearchCV(grid_search,parameters)


# In[68]:


grid.fit(x_train,y_train)


# In[69]:


print(grid.best_params_)


# In[70]:


new_rfc = RandomForestClassifier(criterion='gini', max_depth= 5, n_estimators=150)


# In[71]:


new_rfc.fit(x_train,y_train)
new_rfc_pred = new_rfc.predict(x_test)


# In[72]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,new_rfc_pred)
accuracy


# In[73]:


# Support Vector Machine
clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
clf_pred = clf.predict(x_test)


# In[74]:


from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,clf_pred)
accuracy


# In[75]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier()


# In[76]:


model_dt.fit(x_train,y_train)


# In[78]:


y_pred_dt = model_dt.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred_dt)
accuracy


# In[79]:


from sklearn.ensemble import GradientBoostingClassifier
model_gb = GradientBoostingClassifier()
model_gb.fit(x_train,y_train)


# In[80]:


y_pred_gb = model_gb.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_dt)
accuracy


# In[82]:


test_data = pd.read_csv('E:\\ML Datasets\\Titanic\\test.csv')


# In[83]:


test_data.head()


# In[84]:


test_data.isnull().sum()


# In[85]:


val_to_replace = {'male': 0, 'female':1}
test_data['Sex'] = test_data['Sex'].map(val_to_replace)


# In[87]:


test_data.head()


# In[88]:


for row in test_data:
    test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)  


# In[89]:


test_data.head()


# In[91]:


test_data['Title'].value_counts()


# In[92]:


title_mapping = {"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Col":3,"Mlle":3,"Major":3,"Sir":3,"Lady":3,"Ms":3,"Capt":3,"Jonkheer":3,"Don":3,"Countless":3,"Mme":3}
test_data['Title']=test_data['Title'].map(title_mapping)


# In[93]:


test_data['Title'].value_counts()


# In[94]:


test_data['Title'].isnull().sum()


# In[95]:


test_data[test_data['Title'].isnull()].index.tolist()


# In[96]:


test_data.iloc[414]


# In[97]:


test_data.iloc[414,-1] = 3.0


# In[98]:


test_data.iloc[414]


# In[100]:


test_data['Title'].isnull().sum()


# In[101]:


test_data['Age'].fillna(test_data.groupby("Title")["Age"].transform("median"),inplace=True)


# In[102]:


test_data['Age'].isnull().sum()


# In[103]:


test_data['Embarked'] = test_data['Embarked'].fillna('S')


# In[104]:


test_data['Embarked'].isnull().sum()


# In[106]:


embark = pd.get_dummies(test_data['Embarked'],drop_first=True)


# In[107]:


test_data = pd.concat([test_data,embark],axis=1)


# In[108]:


test_data.head()


# In[109]:


feature_drop = ['Name','Embarked','Ticket','Cabin']
test_data.drop(feature_drop,axis=1,inplace=True)


# In[110]:


test_data.head()


# In[111]:


test_data.isnull().sum()


# In[112]:


test_data[test_data['Fare'].isnull()].index.tolist()


# In[113]:


test_data.iloc[152]


# In[116]:


test_data['Fare'].fillna((test_data["Fare"].mean()),inplace=True)


# In[117]:


test_data.isnull().sum()


# In[118]:


#Prediction for test data set
test_pred = lg_model.predict(test_data)


# In[119]:


test_pred


# In[ ]:





# In[ ]:





# In[ ]:




