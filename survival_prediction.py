"""
Steps:
Exploratory data analysis
    Check datashape, missing values
    Correlation survival with all different variables
Feature engineering
    Missing values Age: Make a correlation heatmap/factorplot of the variable Age with other variables. 
        Extract Title from Name. Impute missing values from median age grouped by Sex, Pclass and Title, FamilySize, 
    Missing values Embarked: Fill with S
    Missing values Cabin: Fill missing with U from Unknown
    Make FamilySize. Single, small, med, large
Build model
Evaluate model
Feature importance check
Train model again only with important features
"""
import pandas as pd
import numpy as numpy
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

#Check correlation between survival and Ticket
train_df['Ticket'].head() 

#Extract ticket prefixes (returns X if prefix is not available)
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


#Check correlation between survival and Fare
g = sns.violinplot(x = 'Survived', y='Fare', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. fare.png')
plt.show()

# Check correlation between survival and Embarked

# Impute missing values with most frequent value S
train_df['Embarked'] = train_df['Embarked'].fillna('S')  
train_df['Embarked'].isnull().sum()

g = sns.barplot(x='Embarked', y='Survived', data=train_df)
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. embarked.png')
plt.show()

# Check correlation between survival and Cabin

# Impute missing values with U from Unknown
train_df['Cabin'] = train_df['Cabin'].fillna('U') 

# Assign each cabin value with its first cabin letter
train_df['Cabin'] = train_df['Cabin'].map(lambda c: c[0])  

g = sns.countplot(train_df["Cabin"],order=['A','B','C','D','E','F','G','T','U'])

g = sns.catplot(y="Survived",x="Cabin",data=train_df,kind="bar", order=['A','B','C','D','E','F','G','T','U'])
g = g.set_ylabel('Survival probability')
plt.savefig('survival vs. cabin.png')
plt.show()

