import streamlit as st
import pandas as pd

title = "Projet CO2Py"
sidebar_name = "Introduction"
from PIL import Image

def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    



    st.image(Image.open("cd ../cd ../images/CO2Py.jpg"))


    st.title(title)

    st.markdown("---")    
    st.subheader("Contexte")
    st.markdown("""Le transport, avec l'industrie, est un des principaux émetteurs de CO2, gaz à effet de serre très fortement impliqué dans le réchauffement climatique.  
                Les émissions de CO2 des véhicules neufs sont donc des données importantes pour la protection de la planète, les déterminer dès la conception des véhicules pourrait aider les constructeurs dans leurs choix.""")    
 
    st.subheader("Objectif")
    st.markdown(
        """
        L'objectif de ce projet est donc de trouver un modèle pour prédire les émissions de CO2 des véhicules d'après leurs caractéristiques techniques, et ce dès la phase de conception, avant toute fabrication de prototype.
        """
    )

 
     
