import streamlit as st
import pandas as pd
import numpy as np

#Pour la modélisation
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, f1_score
from sklearn import linear_model
from sklearn.linear_model import RidgeCV , LassoCV, lasso_path, Ridge
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
import xgboost
from xgboost import XGBRegressor

#Pour les graphs
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_file, show
from bokeh.plotting import figure
from numpy import histogram, linspace
from scipy.stats.kde import gaussian_kde
from bokeh.models import HoverTool, LinearAxis, Range1d, ColumnDataSource
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.annotations import BoxAnnotation

#Pour l'interprétabilité'
import streamlit.components.v1 as components
import random
import shap
shap.initjs()
import eli5
from skater.core.explanations import Interpretation
interpreter = Interpretation()
from skater.data import DataManager
from skater.model import InMemoryModel , DeployedModel


title = "Modélisation"
sidebar_name = "Modélisation"


@st.cache
def prepare_data(fichier):
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
    return data


@st.cache()
def fit_preprocess_poly(xtrain): 
    poly= PolynomialFeatures(degree=4)
    xtrain_poly = poly.fit_transform(xtrain)
           
            
    scaler = preprocessing.MinMaxScaler()
    xtrain_new=scaler.fit_transform(xtrain_poly)
    
    return poly,scaler,xtrain_poly,xtrain_new        

@st.cache()
def fit_ridge(xtrain, ytrain)   :         
             
    ridge = Ridge(alpha=0.8)
    ridge.fit(xtrain, ytrain)
    
    return ridge
    
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
      
    return xgb



def run():

    st.title(title)

    st.markdown(
        """
    Les variables conservées sont donc :              
    Carburant / Hybride / Puissance maximale (kW) / type_boite / nombre_vitesses / CO2 (g/km) / gamme / masse vide moyenne

    Les variables catégorielles sont traitées via un pd.getdummies. La table de travail contient les données :
        """
    )

    data=prepare_data('/app/soutenance_co2py/Streamlit/streamlit_app/Data/data2013.csv')
   # st.write(data.head(5))
    st.write(data.describe())
    

    st.markdown(
        """
    Premiers modèles : LinearRegression, Ridge et Lasso. 
    (cf [Notebook 1](https://github.com/DataScientest-Studio/Co2Py.v2/blob/main/Projet%20Co2Py%20iteration%201v2.ipynb) )
        """
    )
    
    st.markdown(
        """
    Préprocessing :  MinMaxScaler. 
    (cf [Notebook 2](https://github.com/DataScientest-Studio/Co2Py.v2/blob/main/Projet%20Co2Py%20iteration%202%263-new.ipynb))  
     Métriques : score et rmse
    
        """
    )
    
    st.markdown("Création des jeux de test et d'entraînement : 20% de données en test. La cible est 'CO2 (g/km)<br />'", unsafe_allow_html=True)    
    feats=data.drop(['CO2 (g/km)'],axis=1)
    target=data['CO2 (g/km)']
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=1)
    #st.write(X_test.describe())
    
    st.subheader('Le modèle Ridge avec Polynomial Features de degré 4') 
    
    #préparation des jeux d'entrainement et de test en polynome de degré 4 avec application d'un MinMaxScaler
    poly,scaler,X_train_poly,X_train_new=fit_preprocess_poly(X_train)
    X_test_poly = poly.transform(X_test)
    X_test_new=scaler.transform(X_test_poly)
    col_train=poly.get_feature_names(X_train.columns)
    col_test=poly.get_feature_names(X_test.columns)
    X_train_poly_mM=pd.DataFrame(data=X_train_new,columns=col_train,index=X_train.index)
    X_test_poly_mM=pd.DataFrame(data=X_test_new,columns=col_test,index=X_test.index)
    
    #entraintement du modèle
    ridge_poly=fit_ridge(X_train_poly_mM,y_train)
    
       
     #Application du modèle   
    coefficients=pd.DataFrame(data=ridge_poly.coef_, columns=['Coef'], index=col_test)
                
    y_train_predict = ridge_poly.predict(X_train_poly_mM)
    train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    train_score = ridge_poly.score(X_train_poly_mM,y_train)
        
    y_test_predict = ridge_poly.predict(X_test_poly_mM)
    test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    test_score = ridge_poly.score(X_test_poly_mM,y_test)
                
    ecart_rmse=test_rmse/train_rmse-1
    ecart_score=(test_score-train_score)*100
 
    st.write("""<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;margin-bottom:0;"><b>Résultats :</b></p>  
             <ul style="list-style-type:disc;line-height:15px;padding-left: 25px;"> 
             <li>Base d'apprentissage</u> : score = {:.2%}  & rmse = {:.2}.</li>""".format(train_score,train_rmse),"""  
             <li>Base de test : score = {:.2%}  & rmse = {:.2}.</li>""".format(test_score,test_rmse),"""  
             <li>Effet d'apprentissage : {:.2f} points sur le score & {:2.2%} sur l'erreur quadratique moyenne.</li>""".format(ecart_score,ecart_rmse)
               , unsafe_allow_html=True)
             
             
    st.markdown( '<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;margin-bottom:0;"><b>Quelques graphiques :</b></p>' , unsafe_allow_html=True)    
    fig, ax = plt.subplots(1,1, figsize=(10,5), sharex=True, sharey=True)

    sns.set(font_scale=1)
    ax.scatter(X_test_poly_mM["Puissance maximale (kW)"], y_test, label="Data points",color="blue")
    
    ax.scatter(X_test_poly_mM["Puissance maximale (kW)"], y_test_predict,
             color='magenta', lw=1, label="Predictions")
    ax.set_xlabel("Puissance maximale (kW)")
    ax.set_ylabel("Prédiction d'émission de CO2")
    ax.legend()
    st.pyplot(fig)
             
    Coefficients=coefficients.copy()
    Coefficients['Coef_abs']=np.abs(coefficients['Coef'])
    Coef_tri=Coefficients.sort_values(by = 'Coef_abs', ascending = False)        
    Coef_tri2=Coef_tri.head(15).sort_values(by = 'Coef', ascending = False) 

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    
    sns.set_style("white")
    ax=sns.barplot(y=Coef_tri2.index,
                x=Coef_tri2['Coef'],
                  palette="mako")
    ax.set_title('15 variables ayant le plus de poids dans le modèle',fontsize = 20)
    ax.set_xlabel('Coefficients en valeur relative',fontsize =20)  
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20)
    st.pyplot(fig)
             
             
             
             
             
    st.markdown("""<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;margin-bottom:0;"><b>Bilan :</b></p> 
                <ul style="list-style-type:disc;line-height:15px;padding-left: 25px;"> 
                <li>Avantages : Très bon résultat et effet de surapprentissage assez faible.</li>   
                <li>Inconvénients : 91 390 variables => très lourd si plus de données et impossibilité d'utiliser les outils d'interprétabilité (trop lourd)</li>
                <br />"""
                 , unsafe_allow_html=True)
   
             
             
   # st.write('test',train_score.style.format("{:.2f}"))
    st.subheader('Le modèle validé : XGBOOST Regressor')
  
    xgb = fit_xgb_regression(X_train, y_train)      
    
    xgb_train_pred = xgb.predict(X_train)
    xgb_test_pred = xgb.predict(X_test)
    train_score= xgb.score(X_train,y_train)
    test_score= xgb.score(X_test,y_test)
    
    #interprétabilité locale avec shap
    rand =random.randint(1, len(X_test)) 
    masker = shap.maskers.Independent(X_test, max_samples=1000)
    explanation = shap.TreeExplainer(xgb, masker)
    shap_values = explanation.shap_values(X_test.iloc[rand,:])
    graph_shap=shap.force_plot(explanation.expected_value, shap_values, X_test.iloc[rand,:])
        

    
    train_rmse= (np.sqrt(mean_squared_error(y_train, xgb_train_pred)))
    test_rmse= (np.sqrt(mean_squared_error(y_test, xgb_test_pred)))
    ecart_rmse=test_rmse/train_rmse-1
    ecart_score=(test_score-train_score)*100
        
      
    st.write("""<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;margin-bottom:0;"><b>Résultats :</b></p>  
             <ul style="list-style-type:disc;line-height:15px;padding-left: 25px;"> 
             <li>Base d'apprentissage : score = {:.2%}  & rmse = {:.2}.</li> """.format(train_score,train_rmse),"""  
             <li>Base de test : score = {:.2%}  & rmse = {:.2}.</li> """.format(test_score,test_rmse),"""  
             <li>Effet d'apprentissage : {:.2f} points sur le score & {:2.2%} sur l'erreur quadratique moyenne.</li> """.format(ecart_score,ecart_rmse)
             , unsafe_allow_html=True)    
    
    st.markdown('<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;margin-bottom:0;"><b>Quelques graphiques :</b></p>' , unsafe_allow_html=True)       
    fig, ax = plt.subplots(1,1, figsize=(10,5), sharex=True, sharey=True)

    # Je prends la masse vide moyenne en abscisse car c'est la variable la plus importante de l'arbre
    
    ax.scatter(X_test["masse vide moyenne"], y_test, label="Data points",color="blue")
    
    ax.scatter(X_test["masse vide moyenne"], xgb_test_pred,
             color='magenta', lw=1, label="Predictions")
    
    
    ax.plot([1450, 1450, 2700, 2700, 1450],[270, 400, 400, 270, 270],'darkorchid')
    plt.annotate('Zone de prédiction un peu moins fiable', xy=(1800, 400), xytext=(1320, 500), arrowprops={'facecolor':'darkorchid'} );
    
    ax.set_xlabel("masse vide moyenne")
    ax.set_ylabel("Prédiction d'émission de CO2")
    ax.legend();
    st.pyplot(fig)
    
    st.image('/app/soutenance_co2py/Streamlit/images/xgbsmall.png')
    
    
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    st.write('Interprétabilité locale sur la ligne {} de la base de test'.format(rand-1))
    st_shap(graph_shap)
    
    st.markdown("""<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;margin-bottom:0;"><b>Bilan :</b></p> 
                <ul style="list-style-type:disc;line-height:15px;padding-left: 25px;"> 
                <li>Avantages : Très bon résultat et effet de surapprentissage assez faible.</li>    
                <li>Inconvénients : Du fait du pd.getdummies il reste tout de même 36 variables.  </li> 
                """
                , unsafe_allow_html=True)
    st.markdown("""Pour la démonstration je prendrai aussi 2 modèles avec des sélections de variables
                &nbsp;
                &nbsp;""")
    
    
    
    st.subheader('Les autres modèles')
    st.markdown(
        """
    Tous les modèles testés sont détaillés dans le [Notebook 2](https://github.com/Nathaliegar/Soutenance_Co2Py/blob/main/Co2Py%20d%C3%A9doublonn%C3%A9%20notebook%202.ipynb)
    et le [Notebook 3](https://github.com/Nathaliegar/Soutenance_Co2Py/blob/main/Co2Py%20d%C3%A9doublonn%C3%A9%20notebook%203.ipynb).
    Les résultats sont ci-dessous :   
        """
    )  
    

    resultats=pd.read_excel('/app/soutenance_co2py/Streamlit/streamlit_app/Data/resultats.xlsx')
    st.dataframe(resultats)
