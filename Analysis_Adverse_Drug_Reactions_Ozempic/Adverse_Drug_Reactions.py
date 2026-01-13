import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Download
## Source: https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files
## File: ASCII Data Files July–September 2025
### Contents to use:
###   - DEMOyyQq.TXT → Demographic and administrative information per report
###   - DRUGyyQq.TXT → Information about drugs per report
###   - REACyyQq.TXT → Adverse reactions coded with MedDRA
###   - OUTCyyQq.TXT → Patient outcomes (hospitalization, death, etc.)

Demo_data = pd.read_csv('ASCII/DEMO25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'age', 'age_cod', 'sex'],)
Drug_data = pd.read_csv('ASCII/DRUG25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'drugname', 'role_cod'],)
Reac_data = pd.read_csv('ASCII/REAC25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'pt'])
Outc_data = pd.read_csv('ASCII/OUTC25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'outc_cod'],)

# Since the datasets are very large, we will work with a reduced sample and focus on drugs with a sufficient number of cases (~100) and relevance.
Drug_data['drugname'] = (Drug_data['drugname'].str.upper().str.strip())

drug_counts = Drug_data.groupby('drugname')['caseid'].nunique().sort_values(ascending=False)

# Visualizing the sample counts, a drug worth analyzing due to its current industry relevance, safety, and importance is OZEMPIC
## Additionally, only those cases where Ozempic is considered the primary drug (PS) will be taken into account

Ozempic_data = Drug_data[(Drug_data['drugname'].str.contains('OZEMPIC', na=False)) & (Drug_data['role_cod']=='PS')]['caseid'].unique()

# We filter the other dataframes
Demo_ozempic = Demo_data[Demo_data['caseid'].isin(Ozempic_data)]
Reac_ozempic = Reac_data[Reac_data['caseid'].isin(Ozempic_data)]
Outc_ozempic = Outc_data[Outc_data['caseid'].isin(Ozempic_data)]
Outc_ozempic['serious']= Outc_ozempic['outc_cod'].isin(['DE','LT','HO'])
Outc_ozempic['serious']=Outc_ozempic['serious'].replace({True:1, False:0})

base = Outc_ozempic.copy()

# We merge the dataframes using the cases present in `base` (Outc_ozempic → fewer cases). For now, we will not consider the names of the adverse reactions.
base = base.merge(Demo_ozempic[['caseid', 'age', 'age_cod', 'sex']], on='caseid', how='left')

# We add variables of interest
reac_counts = Reac_ozempic.groupby('caseid').size().reset_index(name='n_adv_events') # Number of adverse reactions
base = base.merge(reac_counts, on='caseid', how='left')

# We check if other drugs have been reported for the selected patients 
n_drugs = Drug_data.groupby('caseid')['drugname'].nunique().reset_index(name='n_drugs')
base = base.merge(n_drugs, on='caseid', how='left')
base = base.set_index('caseid')

# Visualize the contents of the variables
print(base['age_cod'].value_counts(dropna=False))
print(base['sex'].value_counts(dropna=False))
print(base['age'].value_counts(dropna=False))

# Convert patients with age in DEC (decades) or MON (months) to YR (years)
def age_year (i):
    if pd.isna(i['age']):
        return None
    if i['age_cod'] =='YR':
        return i['age']
    elif i['age_cod'] == 'MON':
        return round(i['age']/12)
    elif i['age_cod']=='DEC':
        return i['age']*10
    else:
        return None
    
base['age_year']= base.apply(age_year,axis=1)
base = base.drop(columns=['age', 'age_cod'])
base = base.dropna(subset=['age_year']) # Remove patients whose age is unknown
base['sex'] = base['sex'].fillna('D') # Unknown

# Finally, we check for NaN values
base.isna().sum()

# Perform descriptive analysis
## Age distribution
sns.histplot(base['age_year'], bins=30, kde=True)
plt.title('Distribución por edades')
plt.xlabel('Edad')
plt.show()

## Sex distribution
base['sex'].value_counts().plot(kind='bar')
plt.title('Distrubución por sexo')
plt.show()

## Distribution of the number of adverse reactions per patient
sns.histplot(base['n_adv_events'],bins=20)
plt.title('Reacciones adversas por pacientes')
plt.show()

## Distribution of the number of drugs used by patients (excluding Ozempic)
sns.histplot(base['n_drugs'], bins=20)
plt.title('Fármacos por pacientes')
plt.show()

## Proportion of patients with serious outcomes from Ozempic use
base['serious'].value_counts(normalize=True).plot(kind='bar')
plt.title('Proporción de efectos graves')
plt.show()

(base['serious'].value_counts(normalize=True)*100).plot(kind='pie', autopct='%1.1f%%')
plt.title('Porcentaje de efectos adversos graves en pacientes')
plt.show()

## Comparison by age and sex
sns.boxplot(x='serious', y='age_year', data=base)
plt.title('Edad vs Efectos graves')
plt.show()

sns.countplot(data = base, x='sex',hue='serious')
plt.title('Sexo vs efectos graves')
plt.show()

# Is there a higher risk of serious effects if Ozempic is combined with other drugs?
sns.boxplot(x='serious', y='n_drugs', data=base)
plt.title('Fármacos vs efectos graves')
plt.show()

sns.boxplot(x='serious', y='n_adv_events', data=base)
plt.title('Número de reacciones vs. efectos graves')
plt.show()

# From this analysis, it could be stated that the number of adverse reactions does not seem to be a determining factor for whether the outcome after taking Ozempic is serious or not.
# However, combining Ozempic with other drugs tends to be slightly more significant in experiencing serious effects. This suggests that as the number of drugs increases, the probability of experiencing a serious adverse effect also increases.

# We create a predictive model to determine which variables most influence whether a patient experiences a serious adverse effect
y = base['serious']
x = base[['age_year', 'n_adv_events', 'n_drugs']]
x = pd.concat([pd.get_dummies(base['sex'], drop_first=True), x], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2002)

model = RandomForestClassifier(n_estimators=100, random_state=2002)
model.fit(x_train, y_train)

y_pred= model.predict(x_test)
print(classification_report(y_test, y_pred))


## Variable importance
importancia = pd.Series(model.feature_importances_, index=x.columns)
importancia.sort_values().plot(kind='barh')
plt.title('Importancia de variables para predecir outcomes graves')
plt.show()


# We try a linear model to better capture class 1
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(max_iter=1000, class_weight='balanced')
LR.fit(x_train, y_train)

y_pred_LR = LR.predict(x_test)
print(classification_report(y_test, y_pred_LR))

# By applying class weighting to address data imbalance, the model significantly improves its ability to identify serious adverse events, achieving a recall of 46% for the class of interest.

## Metrics to evaluate the machine learning models
confusion_matrix(y_test, y_pred_LR)

y_proba= LR.predict_proba(x_test)[:,1]
roc_auc_score(y_test, y_proba)

# By applying class weighting to address data imbalance, the model notably improves its ability to identify serious adverse events, achieving a 46% recall for the class of interest

## Metrics to evaluate the machine learning models
coef = pd.Series(LR.coef_[0], index=x_train.columns)
coef.sort_values(ascending=False)
np.exp(coef).sort_values(ascending=False)


# Analysis of specific adverse reactions in individuals who took Ozempic and experienced adverse effects
top_reacs = Reac_ozempic[Reac_ozempic['caseid'].isin(base[base['serious']==1].index)]['pt'].value_counts().head(10)

sns.barplot(top_reacs, orient='h')
plt.title('Reacciones adversas más producidas por el consumo de Ozempic')
plt.xlabel('Número de casos con efectos graves')
plt.ylabel('Reacción adversa (MedDRA PT)')
plt.show()

## We can compare this with the adverse reactions of patients who did not experience a serious outcome
reac_serious = Reac_ozempic.merge(base[['serious']], left_on='caseid', right_index=True)
reac_counts2 = reac_serious.groupby(['pt','serious']).size().unstack(fill_value=0)
reac_counts_norm = reac_counts2.div(reac_counts2.sum(axis=0), axis=1)

reac_counts_norm.sort_values(1, ascending=False).head(10).plot(kind='barh')
plt.xlabel('Proporción de pacientes')
plt.title('Top reacciones adversas según gravedad')
plt.gca().invert_yaxis()
plt.show()

#---------------------CONCLUSIONS-------------------

# 1. The number of drugs combined with Ozempic is the most relevant factor in the dataframe for predicting serious outcomes.  
# 2. Adverse reactions (Emotional distress, Nausea, Dehydration, and Vomiting) could serve as risk indicators.  
# 3. Predictive models are subject to limitations due to the inherently noisy nature of FAERS data.

