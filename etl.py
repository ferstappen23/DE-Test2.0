from helpers.db import DB
import pandas as pd
import json
import numpy as np
import pdb
import streamlit as st
from subprocess import call
import requests  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
import category_encoders as ce



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.metrics







db=DB()
def extract():
    data=db.engine.execute("SELECT * FROM db_name ").fetchall()
   # pd.from_sql()
    data =pd.DataFrame(data)
    data=data.rename(columns={0:"datasetid",1:'recordid',2:'fields',3:'geometry',4:'record_timestamp'})
    
    print(data.head())
    return data

print(extract())


def transform(df):
    #create dic relate index and recordid
    df.fields=df.fields.apply(json.loads)
    df.geometry=df.geometry.apply(json.loads)
    #preporccesion for fields
    for column in df.fields[0].keys():
        values=[]
        for i in range(df.shape[0]):
            if (column not in df.fields[i].keys()):
                df.fields[i][column]=None
                values.append(df.fields[i][column])
            else:
                values.append(df.fields[i][column])
        df[column]=values
    #preprocessing for geometry 
    for column in df.geometry[0].keys():
        values=[]
        for i in range(df.shape[0]):
            if type(df.geometry[i])!=dict:
                values.append(None)
            else:
                values.append(df.geometry[i][column])
        df[column]=values
    #preprocessing for geo_shape
    for i in range(df.shape[0]):
        if (type(df['geo_shape'][i])!=dict):
            df['geo_shape'][i]=[]
        else:
            df['geo_shape'][i]=df['geo_shape'][i]['coordinates']
    #creation duration:
    df.drop(columns=["recordid","datasetid","geometry","fields","record_timestamp","geo_point_2d","type"],inplace=True)
    df.date_fin=pd.to_datetime(df.date_fin)
    df.date_debut=pd.to_datetime(df.date_debut)
    df["duree(day)"]=(df["date_fin"]-df["date_debut"])/np.timedelta64(1, 'D')

    return(df)
rt=transform(extract())
#pdb.set_trace()

def valeurs_manquantes(Data):
    print("\n\n les variables qui contiennent les valeurs manquantes \n",Data.loc[:, Data.isnull().any()].columns)
    percent_missing=(Data.isnull().sum()*100/Data.shape[0]).sort_values(ascending=True)
    fig = plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.figure(figsize = (20,10))
    plt.xticks(rotation=90)
    plt.title("Missing Value Analysis")
    plt.xlabel("Features")
    plt.ylabel("% of missing values")
    plt.bar(percent_missing.sort_values(ascending=False).index,percent_missing.sort_values(ascending=False),color=(0.1, 0.1, 0.1, 0.1),edgecolor='blue')
    plt.show()

def discription(Data):
    categorique = Data.select_dtypes(include='object')
    numerique= Data.select_dtypes(exclude='object')
    print("\n")
    print("discription des variables numerique\n",numerique.describe())
    print("\n")
    print("discription des variables categorique\n",categorique.describe())
    print("\n")
    print("les valeurs des variables categoriques\n")
    for col in categorique:
        if (col!='geo_shape') and (col!="coordinates"):
            print(f'{col:-<25}{df[col].unique()[:3]}\n')


def corr(Data,columns):
    fig, ax=plt.subplots(figsize=(15,15))
    correlation=Data[columns].corr(method="pearson")
    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlation,square=True, vmin=-1, mask=mask,vmax=1,cmap=sns.diverging_palette(220,10,as_cmap=True),annot=True)
    
    plt.title('PEARSON CORRELATION')
    plt.show()
    print("on remarque qu'il y a une correlation entre typologie et niveau de perturbation")

def plot_typologie_niveau_perturbation(Data,dic_NIVEAU_PERTURBATION,dic_TYPOLOGIE):
    bar_plot(Data.assign(typologie=Data['typologie'].map(dic_TYPOLOGIE),niveau_perturbation=Data['niveau_perturbation'].map(dic_NIVEAU_PERTURBATION)),"typologie",'niveau_perturbation')
    plt.show()
    print('On remarque que lorsque les travaux est de typologie privé et concessionnaire on a un niveau de perturbation trés elevé ')

def plot_count_impact_de_circulation(Data,dic):
    plt.figure(figsize=[10,10])
    i = Data['impact_circulation'].map(dic).value_counts()

    sns.barplot(i.index, i.values, alpha=0.9,palette='pastel')
    plt.title("impact de circulation")
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.show()

#on remarque que les travaux a paris ont un impact restrint sur la circulation 


def plot_count_par_objet(Data,dic_TYPOLOGIE):
    Data[['duree(day)','objet']].assign(typologie=Data['objet'].map(dic_TYPOLOGIE)).groupby('objet').mean().sort_values("duree(day)",ascending=False)
    plt.figure(figsize=[10,10])
    sns.lineplot(data=Data[['duree(day)','objet']].assign(typologie=Data['objet'].map(dic_TYPOLOGIE)).groupby('objet').mean().sort_values("duree(day)",ascending=False), x='objet', y="duree(day)",palette='pastel')
    plt.xticks(rotation=90)
    plt.show()
    print("on remarque que les projets d'objet reamengement secteur durent les plus")

def duree_par_typologie(Data,dic_TYPOLOGIE):
    print(Data[['duree(day)','typologie']].assign(typologie=Data['typologie'].map(dic_TYPOLOGIE)).groupby('typologie').mean())
    print('on remarque que les travaux de typologie privé dure les plus') 

def nombre_par_statut(Data,dic_STATUT):
    plt.figure(figsize=[10,10])
    i =Data.assign(statut=Data['statut'].map(dic_STATUT)).groupby('statut').count()
    sns.barplot(i.index, i.typologie, alpha=0.9,palette='pastel')
    plt.title("nombre de travaux ")
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.show()
    print("on remarque que la majorité des travaux sont en cours de construction")

def pie(df,col,dic):
    colors = sns.color_palette('pastel')[0:10]
    if col in ["niveau_perturbation","typologie","statut","impact_circulation","numero_stv"]:
        labels=list(df[col].map(dic).value_counts().index)
    else:
        labels=list(df[col].value_counts().index)
    x=list(df[col].value_counts().values)
    labels=list(df[col].map(dic).value_counts().index)
    plt.pie(x,labels = labels, colors = colors, autopct='%.0f%%')

def visualisation(df):
    
    dic_impact_circulation={1:'RESTREINTE',2:'SENS_UNIQUE',3:'BARRAGE_TOTAL',4:'IMPASSE'}
    dic_NIVEAU_PERTURBATION	={2:'Perturbant',1:"Tres perturbant"}
    dic_NUMERO_STV={9:'Nord-Ouest',10:'Nord-Est',11:'Centre',12:'Sud',13:'Sud-Ouest',14:'Sud-Est'}
    dic_TYPOLOGIE={1:'Ville',2:'Concessionnaire',3:'Prive'} 
    dic_STATUT={1:'	A venir',2:'En cours',3:'Suspendu',4:'Prolongé',5:'Terminé'}
    print("data shape:",df.shape)
    print("df_info:",df.info())
    print(df.dtypes.value_counts())

    Data=df.copy()
    
    #valeurs manquantes:
    print("valeur manquantes" )
    valeurs_manquantes(Data)



    # Exploratory Data Analaysis
    print("Exploratory Data Analaysis" )
    discription(Data)
    print("values counts de quelques varibales:\n")
    columns=["niveau_perturbation","typologie","statut","impact_circulation","cp_arrondissement","numero_stv","duree(day)"]
    for col in columns:
        print(df[col].value_counts(),'\n\n')
    
    pie_col={"niveau_perturbation":dic_NIVEAU_PERTURBATION,"typologie":dic_TYPOLOGIE,"statut":dic_STATUT}
    plt.figure(figsize=[20,20])
    n=1
    for cle, valeur in pie_col.items():
        plt.subplots_adjust(hspace=1)
        plt.subplot(1,3,n)
        pie(Data,cle,valeur)
        plt.title(" {} ".format(cle),fontsize=24)
        n=n+1
    plt.tight_layout()
    plt.show()


    #relation entre les variables :
    print("relation entre les variables" )

    print("correlation")
    Data['impact_circulation']=Data['impact_circulation'].replace({'RESTREINTE':1,'SENS_UNIQUE':2,'BARRAGE_TOTAL':3,'IMPASSE':4},regex=True)
    corr(Data,columns)
    print("\n\n")
    
    plot_typologie_niveau_perturbation(Data,dic_NIVEAU_PERTURBATION,dic_TYPOLOGIE)
    print("\n\n")
    
    plot_count_impact_de_circulation(Data,dic_impact_circulation)
    print("\n\n")

    plot_count_par_objet(Data,dic_TYPOLOGIE)
    print("\n\n")

    duree_par_typologie(Data,dic_TYPOLOGIE)
    print("\n\n")

    nombre_par_statut(Data,dic_STATUT)


#preprocessing before training on different model to attempt and predcit the duration of each construction work 

def perprocessing(df):
    #transform date to day/month/year 
    train=df.copy()
    train.drop(columns=["date_debut"],inplace=True)
    #transform objet to dummy variables
    one_hot_encoder=ce.OneHotEncoder(cols='objet',return_df=True,use_cat_names=True)
    train =one_hot_encoder.fit_transform(train)
    #transform impact_circulation
    train['impact_circulation']=train['impact_circulation'].replace({'RESTREINTE':1,'SENS_UNIQUE':2,'BARRAGE_TOTAL':3,'IMPASSE':4},regex=True)
    #transform castisen into polair coordinates 
    L=[]
    N=[]
    for i in df['coordinates']:
        if i==None:
            L.append(i)
            N.append(i)
        else: 
            L.append(np.sqrt(i[0]**2+i[1]**2))
            N.append((math.atan(i[1]/i[0]))/math.pi)
    train['rayon']=L
    train['theta']=N
    #drop unecessary variables
    train.drop(columns=["maitre_ouvrage",'coordinates',"identifiant","geo_shape","precision_localisation","description",'date_fin',"voie"],axis=1,inplace=True)
    train.dropna(inplace=True)
    train.drop_duplicates(inplace=True)
    print(train.head())

    return (train)

def regression_duree(df):
    #train=perprocessing(df)
    print('\n\n')
    print("train info",train.info())
    x= train.drop('duree(day)', axis=1)
    y = train['duree(day)']
    print('\n\n')
    print('colonnes:',x.columns)

    seed      = 9
    test_size = 0.20
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = seed)
    print('\n\n')
    print("X_train.shape:",X_train.shape)
    print("X_test.shape",X_test.shape)
    print("Y_train.shape",Y_train.shape)
    print("Y_test.shape",Y_train.shape)

    scaler = MinMaxScaler().fit(X_train)  
    X_train= scaler.transform(X_train)
    scaler = MinMaxScaler().fit(X_test)  
    X_test= scaler.transform(X_test)

    #variables to tune
    folds   = 10
    metric  = "neg_mean_absolute_error"

    # hold different regression models in a single dictionary
    models = {}
    models["Linear"]        = LinearRegression()
    models["Lasso"]         = Lasso()
    models["Ridge"]         = Ridge()
    models["RandomForest"]  = RandomForestRegressor()
    print('\n\n')
    print("resultat des differents modeles\n")
    # 10-fold cross validation for each model
    model_results = []
    model_names   = []
    for model_name in models:
        model   = models[model_name]
        #10 folds cad k=10
        k_fold  = KFold(n_splits=folds, random_state=seed,shuffle=True)
        results = cross_val_score(model, X_train, Y_train, cv=k_fold,scoring=metric)
        model_results.append(results)
        model_names.append(model_name)
        print("{}: {}, {} \n Parameters currently in use: \n {}".format(model_name, round(results.mean(), 3), round(results.std(), 3),model.get_params()),'\n\n\n')
    figure = plt.figure(figsize=[10,8])
    figure.suptitle('Regression models comparison')
    axis = figure.add_subplot(111)
    plt.boxplot(model_results)
    axis.set_xticklabels(model_names, rotation = 45, ha="right")
    axis.set_ylabel("Mean Absolute Error  (MAE)")
    plt.show()
    print('\n\n')
    print("on trouve que la regression Lasso est le meilleur modele a utilisé \n")

    #create and fit the best regression model
    best_model =Lasso(random_state=seed)
    best_model.fit(X_train, Y_train)
    # make predictions using the model
    predictions = best_model.predict(X_test)
    def scores_(y,x):
        print('MAE:', metrics.mean_absolute_error(y, x))
        print('MSE:', metrics.mean_squared_error(y, x))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y, x)))
        print('R2 Score:' ,metrics.r2_score(y,x))
    scores_(Y_test, predictions)

    print('\n\n')
    # plot model's feature importance
    print("feature importance")
    coefficients=best_model.coef_
    feature_importance = np.abs(coefficients)
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)
    pos        = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(20,10))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, x.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

    #evaluation de best_model
    print('\n\n')
    print('evaluaton du meilleur model : Lasso')
    def evaluate(model, X_test, y_test):
        predictions = model.predict(X_test)
        results=metrics.mean_squared_error(y_test, predictions)
        print('MAE:', metrics.mean_absolute_error(predictions , y_test))
        print('MSE:', metrics.mean_squared_error(predictions , y_test))
        print('RMSE:', np.sqrt(metrics.mean_squared_error( predictions , y_test)))
        print('R2 Score:' ,metrics.r2_score(y_test , predictions))
        return results
    evaluate(best_model,X_test,Y_test)
    print("le modele de regression lineaire n'explique pas assez les données ")



#convert th preprocessed data to a dataframe then to a csv file 
def load ():
    df=transform(extract())
    visualisation(df)
    df.to_csv('chantiersPerturbants.csv',index=False)
    df1=perprocessing(df)
    regression_duree(df1)
    return df,df1
load()



