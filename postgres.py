import os
import pandas as pd
import re
import json
import requests
from helpers.db import DB

db = DB()
url='https://opendata.paris.fr/api/records/1.0/search/?dataset=chantiers-perturbants&q=&rows=10000&facet=cp_arrondissement&facet=typologie&facet=maitre_ouvrage&facet=objet&facet=impact_circulation&facet=niveau_perturbation&facet=statut'
r = requests.get(url)
response=r.json()
df = pd.DataFrame(response["records"])
#engine=create_engine(db.uri)
df['fields'] = df['fields'].apply(json.dumps)
df['geometry'] = df['geometry'].apply(json.dumps)
df.to_sql(name='db_name',con=db.engine,if_exists='append',index=False) #,  dtype={"fields ": sqlalchemy.types.JSON,"geometry":sqlalchemy.types.JSON})

#conn=db.engine.connect()


