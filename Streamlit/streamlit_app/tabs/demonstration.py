import streamlit as st
import pandas as pd
import numpy as np

#Pour la modélisation
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import  SelectFromModel
import xgboost
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline


title = "Démonstration"
sidebar_name = "Démonstration"


@st.cache
def import_data(fichier):
    df=pd.read_csv(fichier, sep=';')
    df=df.drop(['CO type I (g/km)','HC (g/km)','HC (g/km)','NOX (g/km)','HC+NOX (g/km)','Particules (g/km)'],axis=1)
    df=df.dropna()
    df=df.drop_duplicates(keep = 'first')
    df=df.drop(df[df['Consommation urbaine (l/100km)'] == 0].index)
    df['gamme']=df['gamme'].replace('MOY-INF', 'MOY-INFER')
    df_reduit=df[['Carburant','Hybride','Puissance maximale (kW)','Boîte de vitesse','CO2 (g/km)','Carrosserie','gamme','masse vide euro min (kg)','masse vide euro max (kg)']]
    df_reduit=df_reduit.drop_duplicates(keep = 'first')
    data=df_reduit.copy()
    data['type_boite']= data['Boîte de vitesse'].str[0:1]
    data['nombre_vitesses']= data['Boîte de vitesse'].str[-1:]
    data['masse vide moyenne']=(data['masse vide euro min (kg)']+df['masse vide euro max (kg)'])/2
    data=data[['Carburant','Hybride','Puissance maximale (kW)','type_boite','nombre_vitesses','CO2 (g/km)','Carrosserie','gamme','masse vide moyenne']]
    data= data.astype({"nombre_vitesses": int})
    data_num=data.select_dtypes(include=np.number)
    data_cat=data.select_dtypes(exclude=np.number)
    data = data_num.join(pd.get_dummies(data_cat,prefix=data_cat.columns))
    return df_reduit, data


@st.cache(hash_funcs={XGBRegressor: id})
def fit_xgb_regression(xtrain, ytrain): 
    xgb = XGBRegressor(nthread=4,
                   objective='reg:squarederror',
                   reg_alpha=20,
                   reg_lambda=1.1, 
                   learning_rate=0.05, 
                   max_depth=7, 
                   min_child_weight=1.1, 
                   subsample=1, 
                   colsample_bytree=1, 
                   n_estimators=1000,
               random_state=42) 
    xgb.fit(xtrain, ytrain)
    
    sel = SelectFromModel(xgb)
    sel.fit(xtrain, ytrain)
    feature_list = list(xtrain.columns)
    selected_features=np.array(feature_list)[sel.get_support()]
    xtrain_SFM=xtrain[selected_features]
    
    xgbSFM=XGBRegressor(nthread=4,
                   objective='reg:squarederror',
                   reg_alpha=20,
                   reg_lambda=1.1, 
                   learning_rate=0.05, 
                   max_depth=7, 
                   min_child_weight=1.1, 
                   subsample=1, 
                   colsample_bytree=1, 
                   n_estimators=1000,
               random_state=42) 
    xgbSFM.fit(xtrain_SFM, ytrain)
    
    return xgb,xgbSFM,sel



@st.cache()
def fit_pipeline(xtrain,ytrain):
    model=make_pipeline(PolynomialFeatures(degree=4),
                   preprocessing.MinMaxScaler(),
                   Ridge(alpha=0.8))
    model.fit(xtrain,ytrain)
    return model


@st.cache(allow_output_mutation=True)
def fit_rf(xtrain,ytrain):
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(xtrain, ytrain)
    
    selrf = SelectFromModel(rf)
    selrf.fit(xtrain, ytrain)
    feature_list = list(xtrain.columns)
    selected_features=np.array(feature_list)[selrf.get_support()]
    xtrain_SFM=xtrain[selected_features]
    
    rfSFM = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rfSFM.fit(xtrain_SFM, ytrain)
    

    return rf,rfSFM,selrf
    
    
    
    
def run():

    st.title(title)

   
    df,data=import_data('/app/soutenance_co2py/Streamlit/Data/data2013.csv')
    
   # st.write(data.shape)
   # st.write(data.describe())
    
    
    
    #Création du jeu d'entraînement
    feats=data.drop(['CO2 (g/km)'],axis=1)
    target=data['CO2 (g/km)']
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)
    feature_list = list(X_train.columns)
    #st.write(X_test.describe())
    
    #Entrainement des modèles
    xgb,xgbSFM, sel_xgb= fit_xgb_regression(X_train, y_train)
    selected_features_FM_xgb = np.array(feature_list)[sel_xgb.get_support()]
    st.write('Le SelectFromModel du XgboostRegressor conserve les variables:',tuple(selected_features_FM_xgb))
    
    rf,rfSFM,sel_rf=fit_rf(X_train,y_train)
    selected_features_FM_rf = np.array(feature_list)[sel_rf.get_support()]
    st.write('Le SelectFromModel du RandomForestRegressor conserve les variables:',tuple(selected_features_FM_rf))    
    
    ridge_poly=fit_pipeline(X_train,y_train)


    #Formulaire de choix pour tester un véhicule   
    st.subheader('Informations du véhicule à tester')
    st.markdown("Pour utiliser cet outil, si vous ne connaissez pas une information, laissez le champs prérempli tel qu'il est, il prend en compte la moyenne des données observées jusque là.")
    with st.form('vehicule_infos',clear_on_submit=True) :
        choix_carrosserie = st.selectbox('Sélectionnez le type de véhicule :', 
                                         df['Carrosserie'].sort_values().unique(), 
                                         int(np.where(df['Carrosserie'].sort_values().unique()==df['Carrosserie'].mode()[0])[0][0]))
        choix_hybride=st.selectbox('Sélectionnez oui si le véhicule est hybride',
                                   df['Hybride'].sort_values().unique(), 
                                   int(np.where(df['Hybride'].sort_values().unique()==df['Hybride'].mode()[0])[0][0]))
        choix_gamme = st.selectbox('Sélectionnez la gamme du véhicule :', 
                                   df['gamme'].sort_values().unique(), 
                                   int(np.where(df['gamme'].sort_values().unique()==df['gamme'].mode()[0])[0][0]))
        choix_puiss = st.number_input('Saisissez la puissance du véhicule :', 
                                      float(df['Puissance maximale (kW)'].min()), 
                                      float(df['Puissance maximale (kW)'].max()), 
                                      float(df['Puissance maximale (kW)'].mean()))
        choix_carburant = st.selectbox('Sélectionnez le carburant :', 
                                       df['Carburant'].sort_values().unique(), 
                                       int(np.where(df['Carburant'].sort_values().unique()==df['Carburant'].mode()[0])[0][0]))
        choix_masse_min = st.number_input('Saisissez la masse minimale à vide du véhicule en kg :', 
                                          float(df['masse vide euro min (kg)'].min()-200), 
                                          float(df['masse vide euro min (kg)'].max()+200), 
                                          float(df['masse vide euro min (kg)'].mean()))       
        choix_masse_max =st.number_input("Saisissez la masse maximale à vide du véhicule en kg, si vous n'avez qu'une masse vide moyenne, mettez la même dans les deux champs de masse :", 
                                         float(df['masse vide euro min (kg)'].min()-200), 
                                         float(df['masse vide euro min (kg)'].max()+200), 
                                         float(choix_masse_min))              
        choix_boite = st.selectbox('Sélectionnez la boîte de vitesse du véhcule :', 
                                   df['Boîte de vitesse'].sort_values().unique(), 
                                   int(np.where(df['Boîte de vitesse'].sort_values().unique()==df['Boîte de vitesse'].mode()[0])[0][0]))         
        valeur_reelle=st.number_input('Si elle est connue, saisissez ici la valeur réelle des émissions de Co2 en g par km du véhicule, sinon laissez 0',
                                      float(0),
                                      float(max(df['CO2 (g/km)']+500)),
                                      float(0))
        
        button= st.form_submit_button("Calculer")
        
        
   # Compilation des choix
    liste_col = df.columns.to_series().drop('CO2 (g/km)')
    liste_val = np.array([choix_carburant, choix_hybride, round(choix_puiss,2), choix_boite, choix_carrosserie, choix_gamme,round(choix_masse_min,2),round(choix_masse_max,2)])
    df_demo = pd.DataFrame([liste_val], columns =liste_col,index=['Valeurs choisies'])
    st.write ("Les valeurs choisies par l'utilisateur sont : ",df_demo.T)   
    
    
   # Préparation du X_demo
          #Matrice vierge
    matrice=X_train.drop(X_train.index,axis=0)
    
        #Création de df_demo
    df_demo['type_boite']= df_demo['Boîte de vitesse'].str[0:1]
    df_demo['nombre_vitesses']= df_demo['Boîte de vitesse'].str[-1:]
    df_demo= df_demo.astype({"nombre_vitesses": int,'masse vide euro min (kg)':float,'masse vide euro max (kg)':float,'Puissance maximale (kW)':float})
    df_demo['masse vide moyenne']=(df_demo['masse vide euro min (kg)']+df_demo['masse vide euro max (kg)'])/2
    df_demo=df_demo.drop(['Boîte de vitesse','masse vide euro min (kg)','masse vide euro max (kg)'],axis=1)
    df_demo_num=df_demo.select_dtypes(include=np.number)
    df_demo_cat=df_demo.select_dtypes(exclude=np.number)
    df_demo = df_demo_num.join(pd.get_dummies(df_demo_cat,prefix=df_demo_cat.columns))
    
        #Remplissage de la matrice - le fillna ne permettait pas de garder le type uint8 indispensable pour les polynomial features
    X_demo=matrice.append(df_demo)
    for c in range(len(X_demo.columns)) :
        if X_demo.iloc[:,c].isnull().values.any() :
            X_demo.iloc[:,c]=X_demo.iloc[:,c].fillna(0)
            X_demo.iloc[:,c]=X_demo.iloc[:,c].astype('uint8')
       
    
    # Préparation du X_demo réduit via selectionfrommodel du XGBOOST
    X_demo_xgb_SFM=X_demo[selected_features_FM_xgb]
    
    
    # Préparation du X_demo réduit via selectionfrommodel du RandomForest
    X_demo_rf_SFM=X_demo[selected_features_FM_rf]
    
       
      
    # Application des modèles 
    y_xgb=xgb.predict(X_demo)
    y_xgb_SFM=xgbSFM.predict(X_demo_xgb_SFM)
    y_rf=rf.predict(X_demo)
    y_rf_SFM=rfSFM.predict(X_demo_rf_SFM)
    #y_ridge_poly=ridge.predict(X_demo_poly_mM)
    y_ridge_poly=ridge_poly.predict(X_demo)
    
    dico={'Valeur réelle (0 si inconnue)':round(valeur_reelle,2),
          'XGBoostRegressor':round(y_xgb[0],2),
          'XGBoostRegressor & SelectionFromModel':round(y_xgb_SFM[0],2),
          'RandomForestRegressor' :round(y_rf[0],2),
          'RandomForestRegressor & SelectionFromModel' : round(y_rf_SFM[0],2),
          'Ridge avec PolynomialFeatures' : round(y_ridge_poly[0],2)
          }

    st.subheader('Résultats pour le véhicule choisi : ')    
    result=pd.DataFrame(list(dico.items()),
                   columns=['Nom du modèle', 'Emission de CO2 en g/km'])
    #print 
    st.write(result)
    
       
    
   
