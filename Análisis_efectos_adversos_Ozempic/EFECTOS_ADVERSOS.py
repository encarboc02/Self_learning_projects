import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


#Descarga de archivos
##Fuente: https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files
##Archivo: ASCII Data Files Julio–Septiembre 2025
###Contenido a usar
###   - DEMOyyQq.TXT → información demográfica y administrativa por reporte
###   - DRUGyyQq.TXT → información de medicamentos por reporte
###   - REACyyQq.TXT → efectos adversos codificados con MedDRA
###   - OUTCyyQq.TXT → resultados del paciente (hospitalización, muerte, etc.)

Demo_data = pd.read_csv('ASCII/DEMO25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'age', 'age_cod', 'sex'],)
Drug_data = pd.read_csv('ASCII/DRUG25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'drugname', 'role_cod'],)
Reac_data = pd.read_csv('ASCII/REAC25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'pt'])
Outc_data = pd.read_csv('ASCII/OUTC25Q3.txt', sep= '$', encoding='latin1', usecols=['caseid', 'outc_cod'],)

#Al ser datasets muy grandes, trabajaremos con una muestra reducida y para fármacos con número suficiente de muestras (~100) y relevantes
Drug_data['drugname'] = (Drug_data['drugname'].str.upper().str.strip())

drug_counts = Drug_data.groupby('drugname')['caseid'].nunique().sort_values(ascending=False)

#Visualizando el conteo de muestras, un fármaco en los que un análisis merece la pena por surelevancia en la industria actual, seguridad y actualidad es el OZEMPIC
##Además solo se tendrán en ceunta aquellos casos donde el Ozempic es considerado como medicamento primario (PS)

Ozempic_data = Drug_data[(Drug_data['drugname'].str.contains('OZEMPIC', na=False)) & (Drug_data['role_cod']=='PS')]['caseid'].unique()

#Filtramos en los otros dataframe
Demo_ozempic = Demo_data[Demo_data['caseid'].isin(Ozempic_data)]
Reac_ozempic = Reac_data[Reac_data['caseid'].isin(Ozempic_data)]
Outc_ozempic = Outc_data[Outc_data['caseid'].isin(Ozempic_data)]
Outc_ozempic['serious']= Outc_ozempic['outc_cod'].isin(['DE','LT','HO'])
Outc_ozempic['serious']=Outc_ozempic['serious'].replace({True:1, False:0})

base = Outc_ozempic.copy()

#Unimos los dataframe con los casos coincidentes con base (Outc_ozempic --> menos número de casos), de momento no tendremos en cuenta el nombre de las reacciones adversas
base = base.merge(Demo_ozempic[['caseid', 'age', 'age_cod', 'sex']], on='caseid', how='left')

#Añadimos variables de interés
reac_counts = Reac_ozempic.groupby('caseid').size().reset_index(name='n_adv_events') #número de reacciones adversas
base = base.merge(reac_counts, on='caseid', how='left')

#Vemos si se han reportado otros fármacos a los pacientes seleccionados 
n_drugs = Drug_data.groupby('caseid')['drugname'].nunique().reset_index(name='n_drugs')
base = base.merge(n_drugs, on='caseid', how='left')
base = base.set_index('caseid')

#Visualizamos el contenido de variables
print(base['age_cod'].value_counts(dropna=False))
print(base['sex'].value_counts(dropna=False))
print(base['age'].value_counts(dropna=False))

#Transformamos los apceintes con edad en DEC o MON a YR
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
base = base.dropna(subset=['age_year']) #Eliminamos los pacientes cuya edad es desconocida
base['sex'] = base['sex'].fillna('D') #Desconocido

#Comprobamos finalmente los valores Nan 
base.isna().sum()

#Realizamos análisis descriptivos
##Distribución por edades
sns.histplot(base['age_year'], bins=30, kde=True)
plt.title('Distribución por edades')
plt.xlabel('Edad')
plt.show()

##Distribución por sexo
base['sex'].value_counts().plot(kind='bar')
plt.title('Distrubución por sexo')
plt.show()

##Distrubución por el número de reacciones adversas en los pacientes
sns.histplot(base['n_adv_events'],bins=20)
plt.title('Reacciones adversas por pacientes')
plt.show()

##Distribución por número de fármacos empleados por los pacientes (aparte de Ozempic)
sns.histplot(base['n_drugs'], bins=20)
plt.title('Fármacos por pacientes')
plt.show()

##Proporción de pacietnes con efectos graves por el uso de Ozempic
base['serious'].value_counts(normalize=True).plot(kind='bar')
plt.title('Proporción de efectos graves')
plt.show()

(base['serious'].value_counts(normalize=True)*100).plot(kind='pie', autopct='%1.1f%%')
plt.title('Porcentaje de efectos adversos graves en pacientes')
plt.show()

##Comparación por edad y sexo
sns.boxplot(x='serious', y='age_year', data=base)
plt.title('Edad vs Efectos graves')
plt.show()

sns.countplot(data = base, x='sex',hue='serious')
plt.title('Sexo vs efectos graves')
plt.show()

#¿Hay más riesgo de efecto grave si se combina Ozempic con otros fármacos?
sns.boxplot(x='serious', y='n_drugs', data=base)
plt.title('Fármacos vs efectos graves')
plt.show()

sns.boxplot(x='serious', y='n_adv_events', data=base)
plt.title('Número de reacciones vs. efectos graves')
plt.show()

#De este análisis se podría afirmar que el número de reacciones adversas no parace ser un factor determinante si el resultado tras tomar Ozempic en grave o no. 
#Sin embargo, la combinación de Ozempic con otros fármacos si tiende a ser ligeramente más significativo a la hora de sufrir un efectos graves. Por lo que sugiere que a medida que se aumenta el número de fármacos, la probabilidad de que padecer un efecto grave aumenta

#Hacemos un modelo predictivo para saber que variables influyen más a la hora de que un paceinte padezca un efecto grave
y = base['serious']
x = base[['age_year', 'n_adv_events', 'n_drugs']]
x = pd.concat([pd.get_dummies(base['sex'], drop_first=True), x], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2002)

model = RandomForestClassifier(n_estimators=100, random_state=2002)
model.fit(x_train, y_train)

y_pred= model.predict(x_test)
print(classification_report(y_test, y_pred))


##Importancia de las variables
importancia = pd.Series(model.feature_importances_, index=x.columns)
importancia.sort_values().plot(kind='barh')
plt.title('Importancia de variables para predecir outcomes graves')
plt.show()


#Probamos con un modelo lineal para capturar mejor la clase 1
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(max_iter=1000, class_weight='balanced')
LR.fit(x_train, y_train)

y_pred_LR = LR.predict(x_test)
print(classification_report(y_test, y_pred_LR))

#Al aplicar ponderación de clases para abordar el desbalance de los datos, el modelo mejora notablemente su capacidad para identificar eventos adversos graves, alcanzando un recall del 46% en la clase de interés

##Métricas para comprobar el modelos de amchine learning
confusion_matrix(y_test, y_pred_LR)

y_proba= LR.predict_proba(x_test)[:,1]
roc_auc_score(y_test, y_proba)

#El modelo muestra una capacidad limitada de discriminación global (ROC-AUC ~ 0.51), consistente con la naturaleza ruidosa de los datos FAERS. Sin embargo, logra identificar una proporción relevante de eventos graves, lo que lo hace útil como herramienta de cribado

##Coeficiente e importancia de las variables para el modelo
coef = pd.Series(LR.coef_[0], index=x_train.columns)
coef.sort_values(ascending=False)
np.exp(coef).sort_values(ascending=False)


#Análsis de reacciones adversos específicos de personas que consumieron Ozempic y tuvieron efectos adversos
top_reacs = Reac_ozempic[Reac_ozempic['caseid'].isin(base[base['serious']==1].index)]['pt'].value_counts().head(10)

sns.barplot(top_reacs, orient='h')
plt.title('Reacciones adversas más producidas por el consumo de Ozempic')
plt.xlabel('Número de casos con efectos graves')
plt.ylabel('Reacción adversa (MedDRA PT)')
plt.show()

##Podemos compararlo con las reacciones adversas de los pacientes que no han tenido un efecto grave
reac_serious = Reac_ozempic.merge(base[['serious']], left_on='caseid', right_index=True)
reac_counts2 = reac_serious.groupby(['pt','serious']).size().unstack(fill_value=0)
reac_counts_norm = reac_counts2.div(reac_counts2.sum(axis=0), axis=1)

reac_counts_norm.sort_values(1, ascending=False).head(10).plot(kind='barh')
plt.xlabel('Proporción de pacientes')
plt.title('Top reacciones adversas según gravedad')
plt.gca().invert_yaxis()
plt.show()

#---------------------CONCLUSIONES-------------------

# 1. El número de fármacos combinados con Ozempic es el factor más relevante del Dataframe para predecir efectos graves
# 2. las reacciones adversas (Emotional distress, Nausea, Dehydration y Vomiting) podrían servir como indicadores de riesgo
# 3. Los modelos predictivos están sujetos a limitación debido a la inherente naturaleza ruidosa de os datos FAERS
