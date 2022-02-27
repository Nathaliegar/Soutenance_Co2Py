import streamlit as st
import pandas as pd

title = "Conclusion"
sidebar_name = "Conclusion"
from PIL import Image

def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    



    st.image(Image.open("../images/CO2Py.jpg"))


    st.title(title)

    st.markdown("---")    
    st.subheader("Difficultés rencontrées, limites de la modélisation")
    st.markdown("""
                <ul style="list-style-type:disc;line-height:15px;padding-left: 25px;"> 
                <li>Faible volume de données après dédoublonnage (moins de 6 000 lignes).</li>   
                <li>Changement régulier des normes de calcul des émissions de CO2</li>
                <li>Difficulté de compréhension des données pour les autres pays (normes, abbréviations,etc)</li>   
                """ 
                , unsafe_allow_html=True) 
 
    st.subheader("Pistes de réflexion")
    st.markdown("""
                <ul style="list-style-type:disc;line-height:15px;padding-left: 25px;"> 
                <li>Ajout des données des années précédentes en source pour affiner les modèles</li>
                <li>Essai des modèles de deep learning</li>                
                <li>Analyse des données européennes depuis 2015, avec probable nécessité de les enrichir via une autre source. (cf <a href="https://github.com/Nathaliegar/Soutenance_Co2Py/blob/main/Projet%20Co2Py%20European%20Data%20It1.ipynb" target="_blank">Notebook Europe</a>) </li>
                <li>Intégration des véhicules électriques</li>
                """  
                , unsafe_allow_html=True)    
    
     #[Notebook Europe](https://github.com/Nathaliegar/Soutenance_Co2Py/blob/main/Projet%20Co2Py%20European%20Data%20It1.ipynb).</li>  
 
     