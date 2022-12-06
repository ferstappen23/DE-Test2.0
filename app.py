import streamlit as st
import pandas as pd 
from helpers.db import DB
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
from etl import transform ,load, extract , description , visualisation ,plot_typologie_niveau_perturbation , corr , plot_count_par_objet,perprocessing,regression_duree
import matplotlib.pyplot as plt

#st.set_page_config (
  #  page_title="Multipage App",
  #  page_icon= "📈", )
# Create sidebar
st.sidebar.markdown("<div><img src='https://www.vkcreativelearning.com/uploads/services/3dsafety/3d-safety-header.png' width=300 /><h1 style='display:inline-block'>Analyse du trafic </h1></div>", unsafe_allow_html=True)
st.sidebar.markdown("La construction des infrastructures de transport urbain doit être améliorée à mesure que le processus d'urbanisation s'accélère. Les zones du projet de construction deviennent des goulots d'étranglement pour le trafic, qui sont des lieux de congestion et d'accidents. Nous pourrions évaluer les caractéristiques du trafic et proposer des mesures d'amélioration en analysant les problèmes actuels, l'état des routes et la durée de chaque chantier de construction") 
#st.sidebar.markdown("Pour commencer  <ol><li>Enter the <i>hashtag</i> you wish to analyse</li> <li>Hit <i>Get Data</i>.</li> <li>Get analyzing</li></ol>",unsafe_allow_html=True)
#st.sidebar.success("ygyg")



st.write("""
# Trafic et fermetures
## Retrouvez ici toutes les infos qui vous permettront de préparer votre itinéraire en voiture!
***
""")

#image = Image.open('traffic.jpg')

#st.image(image,)"""
db=DB()
data=db.engine.execute("SELECT * FROM db_name ").fetchall()
   #pd.from_sql()
#data =pd.DataFrame(data)

#df = transform(extract())
#df
#pd.DataFrame(
    #np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    #olumns=['lat', 'lon'])

#st.map(df)
df1=pd.DataFrame(transform(extract(data))["coordinates"].dropna().tolist(),columns=["lon","lat"])
st.map(df1)

# Customize the sidebar



st.title('Analyse exploratoire des données ')
st.markdown("### Aperçu des Données")
st.dataframe(pd.DataFrame(data).head())
st.write('---')
st.markdown(" ##### Voici quelques statistiques très basiques pour savoir à quoi nous avons affaire ")
#st.write("Data shape:",extract(data).shape)
st.write("**Datasettype:** {}".format(type(extract(data))))
st.write("**Dataset shape:** {}".format(extract(data).shape))
st.write("**Feature names :** {}".format(extract(data).keys()))
st.write("**Dataset summary**")
st.write(extract(data).describe())
#st.write("Data info:",transform(extract(data)).info())
#st.write(extract(data).dtypes.value_counts())


st.write('---')
description(transform(extract(data)))
#st.markdown("### Sélectionner les colonnes pour l'analyse")
st.markdown(" ##### Choisissez les attributs que vous voulez visualiser ")
selected=st.selectbox('**Features**', ["niveau_perturbation", "typologie","statut","impact_circulation","typologie/niveau_perturbation"])
#visualisation(transform(extract(data)))
#"""i = pd.DataFrame(transform(extract(data))['impact_circulation'].value_counts())#.map(dic)
#st.write(i)
#i["niveau"]=i.index
#st.write(i)


visualisation((transform(extract(data))),selected)
st.title("Preprocessing")
st.write("**Nous avons pensé qu'il serait utile de créer une nouvelle variable appelée 'Duree' qui calcule la durée de chaque travail de construction avant d'ajouter plus de visualisation.**")
if st.button('More Plots'):
    st.header('Autres visualisations utiles')
    
    #relation entre les variables :


    transform(extract(data))['impact_circulation']=transform(extract(data))['impact_circulation'].replace({'RESTREINTE':1,'SENS_UNIQUE':2,'BARRAGE_TOTAL':3,'IMPASSE':4},regex=True)
    #st.write(transform(extract(data))["duree(day)"])
    
    
    columns=["niveau_perturbation","typologie","statut","impact_circulation","cp_arrondissement","numero_stv","duree(day)"]
    dic_impact_circulation={1:'RESTREINTE',2:'SENS_UNIQUE',3:'BARRAGE_TOTAL',4:'IMPASSE'}
    dic_NIVEAU_PERTURBATION	={2:'Perturbant',1:"Tres perturbant"}
    dic_NUMERO_STV={9:'Nord-Ouest',10:'Nord-Est',11:'Centre',12:'Sud',13:'Sud-Ouest',14:'Sud-Est'}
    dic_TYPOLOGIE={1:'Ville',2:'Concessionnaire',3:'Prive'} 
    dic_STATUT={1:'	A venir',2:'En cours',3:'Suspendu',4:'Prolongé',5:'Terminé'}
        
    #plot_typologie_niveau_perturbation(transform(extract(data)),dic_NIVEAU_PERTURBATION,dic_TYPOLOGIE)
 

    plot_count_par_objet(transform(extract(data)),dic_TYPOLOGIE)
   

    corr(transform(extract(data)),columns)
  


st.write("**D'autres transformations utiles ont été effectuées, comme l'encodage de certaines variables catégorielles et la suppression d'autres moins utiles, puis nous avons rejoué les coordonnées, en les transformant de cartésiennes en polaires parce que les coordonnées y et x prises individuellement ne traduisent pas bien la position du chantier et parce que cela permet de mieux comprendre à quelle distance du centre se trouvent ces chantiers.**")
if st.button('Transform'):
    result = perprocessing(transform(extract(data)))
    st.write("Voici le résultat de nos transformations :")
    st.write(result.head())

st.title("Prévision de la durée des chantiers")
st.write("**Trouver les bons artisans, s’accorder sur le devis, choisir les matériaux… votre chantier est enfin prêt à démarrer. Oui, mais combien de temps va-t-il durer ?**")
st.write("**Bien que chaque type de travaux ait sa propre durée, nous allons tenter de prédire la durée de chaque construction ! Mais d'abord, nous devons déterminer le modèle qui nous aidera dans cette entreprise. Nous allons tester quatre modèles de régression bien connus : Régression linéaire, Lasso, Ridge et RandomForest.**")
if st.button('Tkjjhjransform'):
    result = load(data)
    st.write(result)
