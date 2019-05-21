"""
Steps:
Exploratory data analysis
    Check datashape, missing values
    Correlation survival with all different variables
Feature engineering
    Missing values Age: Make a correlation heatmap/factorplot of the variable Age with other variables. 
        Extract Title from Name. Impute missing values from median age grouped by Sex, Pclass and Title, FamilySize. 
    Missing values Embarked: Fill with S
    Missing values Cabin: Fill missing with U from Unknown
    Make FamilySize. Single, small, med, large
Build model
Evaluate model
Feature importance check
Train model again only with important features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

""" Exploratory Data Analysis"""

#Show the train dataset
pd.set_option('display.max_columns', None)
train_df.head()

# Check missing values
train_df.isnull().sum()

#Check correlation between survival and Sex
g = sns.barplot(x='Sex', y='Survived', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. sex.png')

train_df[['Sex','Survived']].groupby('Sex').mean()

#Check correlation between survival and Pclass
g = sns.barplot(x='Pclass', y='Survived', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. pclass.png')
plt.show()

#Check correlation between survival and SibSp
g = sns.barplot(x='SibSp', y='Survived', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. sibsp.png')
plt.show()

#Check correlation between survival and Parch
g = sns.barplot(x='Parch', y='Survived', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. parch.png')
plt.show()

#Check correlation between survival and Fare
g = sns.violinplot(x = 'Survived', y='Fare', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. fare.png')
plt.show()

""" Feature Engineering """

#Check correlation between survival and Ticket by extracting ticket prefixes (returns X if prefix is not available)
train_df['Ticket'].head() 

Ticket = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        Ticket.append("X")

train_df["Ticket"] = Ticket
train_df["Ticket"].head() 
train_df[["Ticket", "Survived"]].groupby("Ticket").mean()

train_df.groupby('Ticket').mean()[['Survived']].plot(kind = 'bar')
plt.ylabel('Survival probability')
plt.show()

# g = sns.barplot(x='Ticket', y='Survived', data=train_df)
# g = g.set_ylabel('Survival probability')
# plt.figure(figsize=(5,5))
# plt.savefig('survival vs. ticket prefix.png')
# plt.figure(figsize=(5,5)).show()


# Check correlation between survival and Embarked by imputing missing values with most frequent value S
train_df['Embarked'] = train_df['Embarked'].fillna('S')  
train_df['Embarked'].isnull().sum()

g = sns.barplot(x='Embarked', y='Survived', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. embarked.png')
plt.show()

# Check correlation between survival and Cabin by imputing missing values with U from Unknown
train_df['Cabin'] = train_df['Cabin'].fillna('U') 

# Assign each cabin value with its first cabin letter
train_df['Cabin'] = train_df['Cabin'].map(lambda c: c[0])  

g = sns.countplot(train_df["Cabin"],order=['A','B','C','D','E','F','G','T','U'])

g = sns.catplot(y="Survived",x="Cabin",data=train_df,kind="bar", order=['A','B','C','D','E','F','G','T','U'])
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. cabin.png')
plt.show()

# Fill missing values in column Age by checking first the correlation between the the variable with the other ones
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)
plt.show()

# Fill the missing values Age with median age based on SibSp, Parch and Pclass
index_missing_age = list(train_df['Age'][train_df['Age'].isnull()].index)

for i in index_missing_age:
    median_age = train_df['Age'].median()
    pred_age = train_df['Age'][((train_df['SibSp'] == train_df.iloc[i]['SibSp']) & (train_df['Parch'] == train_df.iloc[i]['Parch']) & (train_df['Pclass'] == train_df.iloc[i]['Pclass']))].median()
    if not np.isnan(pred_age):
        train_df['Age'].iloc[i] = pred_age
    else:
        train_df['Age'].iloc[i] = median_age     

g = sns.violinplot(x= 'Pclass', y='Age', hue= 'Survived', data= train_df, split= True)
plt.show()

# Make categories from Name 
train_df['Name'].head() 

titles = [] 
for i in list(train_df['Name']):
     titles.append(i.split(',')[1].split('.')[0].strip())

train_df['Title'] = titles      
g = sns.countplot(x='Title', data=train_df)
plt.tick_params(axis= 'x', labelsize = 8)
plt.savefig('titles.png')
plt.show()

train_df['Title'].head() 
train_df['Title'].value_counts() 
train_df['Title'].nunique() 

# Group Mlle and Ms together with Miss, Mme with Mrs and the upper class people together
train_df['Title'] = train_df['Title'].replace(['Mlle','Ms'],'Miss')
train_df['Title'] = train_df['Title'].replace(['Mme'],'Mrs')

# Cluster other categories with low frequency
train_df['Title'] = train_df['Title'].replace([''])  



