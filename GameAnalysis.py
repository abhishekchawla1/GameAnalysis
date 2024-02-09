#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json


# In[2]:


api_url = "https://jsonblob.com/api/jsonBlob/1203954373381447680"


# In[3]:


def make_get_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")


# In[4]:


result = make_get_request(api_url)


# In[5]:


type(result)


# In[6]:


result


# In[7]:


df=pd.DataFrame(result)


# In[8]:


df


# In[9]:


df.shape


# In[10]:


df=df.T


# In[11]:


df


# In[12]:


print(df['status'].unique())
df['status'].value_counts()


# In[13]:


df[df['status']!='COMPLETED']


# In[14]:


df.isnull().sum()


# In[15]:


df['lives'].value_counts()


# In[16]:


df[df['lives'].isna()]['lives']


# In[17]:


from sklearn.impute import SimpleImputer


# In[18]:


df[df['lives']==2]


# In[19]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


# In[20]:


ct = ColumnTransformer([
    ('si1', SimpleImputer(strategy='constant', fill_value=3), ['lives']),
    ('si2', SimpleImputer(strategy='constant', fill_value='NOT COMPLETED'), ['status'])
], remainder='passthrough')


# In[21]:


df2=ct.fit_transform(df)


# In[22]:


new_df=pd.DataFrame(df)
new_df.index=df.index


# In[23]:


new_df.isnull().sum()


# In[24]:


new_df=new_df.rename(columns={0:'lives',1:'status',2:'attempts',3:'level',4:'score'})


# In[25]:


new_df


# In[26]:


new_df['lives'].value_counts()


# In[27]:


new_df['status'].value_counts()


# All the players who have exhausted their lives are marked as those who have completed the game. Players who didnt exhaust their 3 lives and didnt qualift level 10 are marked as those who have not completed this game.

# In[28]:


l=LabelEncoder()
new_df['status']=l.fit_transform(new_df['status'])


# In[29]:


new_df


# In[30]:


import matplotlib.pyplot as plt


# In[31]:


ax=new_df['status'].value_counts().plot(kind='bar')
plt.title('STATUS OF PLAYERS')
plt.xlabel('0:COMPLETED  1:NOT_COMPLETED')
plt.ylabel('NUMBER OF PLAYERS')
for bars in ax.containers:
    ax.bar_label(bars)


# In[32]:


new_df


# In[33]:


ax=new_df['level'].value_counts().plot(kind='bar')
plt.title('PLOT REPRESENTING NUMBER OF PLAYERS CORROSPONDING TO THEIR LEVEL REACHED IN THE GAME')
plt.xlabel('Level')
plt.ylabel('Players')
for bars in ax.containers:
    ax.bar_label(bars)


# In[34]:


new_df


# In[35]:


df.score.values


# In[36]:


plt.figure(figsize=(20,8))
ax=sns.barplot(x=new_df.index,y=new_df.score,data=new_df)
for bars in ax.containers:
    ax.bar_label(bars,rotation='vertical')
plt.xticks(rotation='vertical')
plt.show()


# Top 10 Scorers

# In[37]:


scorers=new_df['score'].sort_values(ascending=False).head(10)


# In[38]:


x=scorers.index
y=scorers.values


# In[39]:


plt.plot(x,y)
plt.xticks(rotation=90)
plt.title('TOP 10 SCORERS')
plt.xlabel('Players')
plt.ylabel('Score')
plt.show()


# In[40]:


new_df


# In[41]:


dd=new_df.attempts.sample(1).values[0]


# In[42]:


xx=pd.DataFrame(dd).T


# In[43]:


xx


# In[71]:


def ret_mat(data):
    return [[1 if cell['isCorrect'] else 0 for cell in row] for row in data]

def create_matrix(input_data):    
    def process_element(element):
        if len(element)>0:
            return element
        else:
            return 1
            
    processed_data = [[process_element(element) for element in row] for row in input_data[0]]    
    result_matrix = np.array(processed_data)    
    return result_matrix

def calculate_neighbor_sum(matrix1, matrix2):
    rows = len(matrix1)
    cols = rows
    total_sum = 0
    for i in range(rows):
        for j in range(cols):
            if matrix1[i][j] != 0 and matrix2[i][j] != 0:
                if i > 0:
                    total_sum += matrix1[i - 1][j]
                if i < rows - 1:
                    total_sum += matrix1[i + 1][j]
                if j > 0:
                    total_sum += matrix1[i][j - 1]
                if j < cols - 1:
                    total_sum += matrix1[i][j + 1]
    if total_sum>3:
        total_sum=3
    return total_sum

def scoring(ele):
    x=pd.DataFrame(ele).T
    return calculate_neighbor_sum(x['matrix'].apply(ret_mat),x['matrix'].apply(create_matrix))


# In[ ]:


new_df['density']=new_df['attempts'].apply(scoring)


# In[83]:


new_df


# In[84]:


new_df.density.value_counts()


# In[87]:


new_df


# In[108]:


plt.figure(figsize=(30,10))
plt.bar(new_df.index,new_df['density'])
plt.xticks(rotation=90)
plt.axhline(y=1,c='r')
plt.axhline(y=2,c='r')
plt.axhline(y=3,c='r')
plt.grid()
plt.title('PLAYER STYLE in the Beginning')
plt.ylabel('1: Low Density first move 2: Medium Density first move, 3: High Density first move')
plt.xlabel('user_id')
plt.show()


# In[112]:


ax=sns.barplot(x='level',y='density',data=new_df)
for bars in ax.containers:
    ax.bar_label(bars)


# The above plot tells us on a given level the mean of density which shows how density impacts decision making in the first move

# In[115]:


new_df.drop(columns='player_style',inplace=True)


# In[116]:


new_df


# In[121]:


final_df=new_df[['level','lives','score','status','density']]


# In[122]:


final_df


# In[128]:


from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[130]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[131]:


x=['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','XGBClassifier']


# In[139]:


def gscv(model,param_grid):
    g=GridSearchCV(model,param_grid=param_grid,scoring='accuracy',n_jobs=-1,cv=3)
    g.fit(final_df.drop(columns='density'),final_df['density'])
    return g.best_params_


# In[ ]:


for m in x:
    model=eval(m)()
    if isinstance(model,LogisticRegression):
        param_grid={'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1],'solver':['liblinear', 'newton-cg', 'newton-cholesky', 'sag']}
    if isinstance(model,DecisionTreeClassifier):
        param_grid={'critereon':['gini','entropy'],'max_depth':[1,5,8,10,12,None]}
    if isinstance(model,RandomForestClassifier):
        param_grid={'n_estimators':[1,10,50,100,120],'max_features':[0.25,0.5,0.75,1],'max_samples':[0.25,0.5,0.75,1],'bootstrap':[True,False]}
    if isinstance(model,GradientBoostingClassifier):
        param_grid={'learning_rate':[0.001,0.01,0.1,0.5,0.75,1],'n_estimators':[1,10,50.100,120]}
    if isinstance(model,XGBClassifier):
        param_grid={'n_estimators':[10,100,120,150]}
    best_params=gscv(model,param_grid)
    mod=eval(m)(**best_params)
    mod.fit(final_df.drop(columns='density'),final_df['density'])


# In[ ]:




