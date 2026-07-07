import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix, roc_curve, precision_score, recall_score, precision_score
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)

df = pd.read_csv('churn_prediction.csv')
#print(pd.isnull(df).sum())
#print(df['gender'].value_counts())

dict_gender = {'Male':1, "Female":0}
df.replace({'gender':dict_gender}, inplace=True)

df['gender'] = df['gender'].fillna(-1)
#print(pd.isnull(df).sum())

#print(df['dependents'].value_counts())
#print(df['occupation'].value_counts())
df['dependents'] = df['dependents'].fillna(0)
df['occupation'] = df['occupation'].fillna('self_employed')
#print(pd.isnull(df).sum())

#print(df['city'].value_counts())
df['city'] = df['city'].fillna(1020)
#print(pd.isnull(df).sum())

#print(df['days_since_last_transaction'].value_counts())
df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(999)
#print(df.isnull().sum())

#print(df.head())
df = pd.concat([df,pd.get_dummies(df['occupation'],prefix = str('occupation'),prefix_sep='_')],axis = 1)
#print(df.head())

num_cols = ['customer_nw_category', 'current_balance',
            'previous_month_end_balance', 'average_monthly_balance_prevQ2', 'average_monthly_balance_prevQ',
            'current_month_credit','previous_month_credit', 'current_month_debit',
            'previous_month_debit','current_month_balance', 'previous_month_balance']
for i in num_cols:
    df[i]=np.log1p(df[i]+17000)


std = StandardScaler()
scaled = std.fit_transform(df[num_cols])
scaled = pd.DataFrame(scaled, columns=num_cols)

df_df_org = df.copy()

df = df.drop(columns = num_cols)
df=df.merge(scaled,left_index=True,right_index=True, how='left')


y_all = df.churn
df = df.drop(['churn','customer_id','occupation'],axis=1)


baseline_cols = ['current_month_debit', 'previous_month_debit','current_balance','previous_month_end_balance','vintage'
                 ,'occupation_retired', 'occupation_salaried','occupation_self_employed', 'occupation_student']
df_baseline = df[baseline_cols]

xtrain, xtest, ytrain, ytest = train_test_split(df_baseline, y_all, test_size=1/3, random_state=11,stratify=y_all)
model = LogisticRegression()
model.fit(xtrain, ytrain)
predictions = model.predict_proba(xtest)[:,1]
