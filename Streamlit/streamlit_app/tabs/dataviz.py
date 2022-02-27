import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_file, show
from bokeh.plotting import figure
from numpy import histogram, linspace
from scipy.stats.kde import gaussian_kde
from bokeh.models import HoverTool, LinearAxis, Range1d, ColumnDataSource
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.annotations import BoxAnnotation


title = ("Data exploration et visualisation")
sidebar_name = "Data visualisation"

def color_(x):
     Type=['GO','ES','EH','GN','GH','Bicarburation','FE']
     color=['magenta','blue','darkorange','darkorchid','green','darkturquoise','indigo']
     return color[Type.index(x)]
    
 
    
# Fonction pour charger les données 
@st.cache
def get_data():
    df=pd.read_csv('./Data/data2013.csv', sep=';')
    df=df.drop(['CO type I (g/km)','HC (g/km)','HC (g/km)','NOX (g/km)','HC+NOX (g/km)','Particules (g/km)'],axis=1)
    df=df.dropna()
    df=df.drop_duplicates(keep = 'first')
    df=df.drop(df[df['Consommation urbaine (l/100km)'] == 0].index)
    df['gamme']=df['gamme'].replace('MOY-INF', 'MOY-INFER')
    df.loc[(df["Carburant"]=='GP/ES')|(df["Carburant"]=='ES/GP')|
     (df["Carburant"]=='ES/GN')|(df["Carburant"]=='GN/ES'), 'Carburant_groupe'] = 'Bicarburation'  
    df.loc[(df["Carburant_groupe"]!='Bicarburation'), 'Carburant_groupe'] = df["Carburant"]
    df['color'] = df['Carburant_groupe'].apply(color_)
    return df


def run():

    st.title(title)
    st.markdown("---")  

    st.markdown(
        """
        Le jeu de donnée concerne les véhicules neufs homologués en France en 2013. 

        ## Préparation des données

        L'exploration des données a permis de gérer les données manquantes (suppression des colonnes avec plus de 90% de manquants, et des quelques lignes avec  les manquants restants), des doublons parfaits en ne conservant que la première ligne et des variables connues uniquement après mise sur route comme les émissions de particules.  
        Voici un aperçu de la table après ce traitement :
        """
    )

    
    # Chargement des données
    df = get_data()   
    
    
    st.write(df.head(5))
    st.write(df.describe())
    st.markdown(
        """
        ## Visualisation des données
        
        L'intégralité des explorations de données et Dataviz préparées est ici : 
        [Notebook 1](https://github.com/Nathaliegar/Soutenance_Co2Py/blob/main/Co2Py%20d%C3%A9doublonn%C3%A9%20notebook%201.ipynb)

       """
    )

    #st.markdown(<span style="color:darkorchid"> **""""Distribution des émissions de CO2"""**</span>)
   
    st.markdown('<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;"><b>Distribution des émissions de CO2 </b></p>', unsafe_allow_html=True)
    #st.pyplot(sns.displot(df['CO2 (g/km)'],bins=20,kind='hist',kde=True,color='Blue',height=5,aspect=2))
    
 
    liste = [("(x, y)", "(@x, @y)")] 
    pdf = gaussian_kde(df['CO2 (g/km)'])
    
    x = linspace(0,max(df['CO2 (g/km)'])+50,500)
    
    p = figure( x_range = (0, max(df['CO2 (g/km)'])+50), plot_height=300)
    
    
    intervalles=30
    hist, edges = np.histogram(df['CO2 (g/km)'], density=True, bins=intervalles)
    test, edges=np.histogram(df['CO2 (g/km)'], density=False, bins=intervalles)
    source = ColumnDataSource(data=dict(hist=hist,test=test,left = edges[:-1], right = edges[1:]))
    

    l=p.line(x, pdf(x),line_color="mediumblue")
    q=p.quad(top='hist', bottom=0, left = 'left', right = 'right', alpha=0.7, fill_color="mediumblue",line_color="white",source=source)
    hoverl=HoverTool(tooltips=liste,renderers=[l])
    hoverq = HoverTool(tooltips = [('probabilité ', '@hist'),('valeur','@test'),
                          ('abscisse', '$x')],renderers=[q])
    p.add_tools(hoverl)
    p.add_tools(hoverq)
    st.bokeh_chart(p, use_container_width=True)
        
    st.markdown('<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;"><b>Emissions de CO2 en fonction de la masse vide et du type de carburant </b></p>', unsafe_allow_html=True)    
    
    
    

    
    source1 = ColumnDataSource(df[(df["Hybride"]=="non")])
    source2 = ColumnDataSource(df[(df["Hybride"]=="oui")])
    
    hover = HoverTool(
            tooltips=[
                ("marque", "@Marque"),
                ("gamme", "@gamme"),
                ("carrosserie", "@Carrosserie")])
     
    p1 = figure(plot_width=750, plot_height=400,x_axis_label='Masse Vide max', y_axis_label='Emission de CO2')
    p1.circle(x='masse vide euro max (kg)',y='CO2 (g/km)',source = source1,color='color',size=2,legend_field='Carburant_groupe')
    
    box1 = BoxAnnotation(bottom = 280, 
                         top=590,
                        left = 1400,             
                        right =3150,             
                        fill_alpha =0.0,        
                        line_color = 'navy',
                        line_width=1,
                        line_alpha=1)     
    p1.add_layout(box1)
    
    
    p1.add_tools(hover)
    p1.legend.location = "top_left"
    tab1 = Panel(child=p1, title="Véhicules non hybrides")
    p1.legend.label_text_font_size = '8pt'
    p1.legend.background_fill_alpha = 0.0
    p1.legend.border_line_alpha= 0.0
    
    
    p2 = figure(plot_width=750, plot_height=400,x_axis_label='Masse Vide max', y_axis_label='Emission de CO2')
    p2.y_range=Range1d(0, 600)
    p2.circle(x='masse vide euro max (kg)',y='CO2 (g/km)',source = source2,color='color',size=2,legend_field='Carburant_groupe')
    p2.add_tools(hover)
    p2.legend.location = "top_left"
    p2.legend.label_text_font_size = '8pt'
    p2.legend.background_fill_alpha = 0.0
    p2.legend.border_line_alpha= 0.0
    tab2 = Panel(child=p2, title="Véhicules hybrides")
    tabs = Tabs(tabs=[ tab1, tab2 ])
    st.bokeh_chart(tabs, use_container_width=True)
    
    
    
    st.markdown('<p style="color:mediumblue; font-size: 20px;"><b>Relation entre puissance maximale et émissions de CO2</b></p>', unsafe_allow_html=True)
    fig, ax = plt.subplots( figsize=(10, 5), sharey=True)
    sns.scatterplot(x='Puissance maximale (kW)', y='CO2 (g/km)', data=df, color='mediumblue')
    plt.plot([520, 600, 600, 520, 520],[520, 520, 600, 600, 520],'navy', alpha = 0.6)
    plt.annotate('Valeurs extrêmes', xy=(520, 560), xytext=(300, 590), c='b', arrowprops={'facecolor':'b'})
    st.pyplot(fig)
    
    
   
    st.markdown('<p style="color:mediumblue; font-size: 20px;;margin-top: 1em;"><b>Analyse des corrélations</b></p>', unsafe_allow_html=True)
    matrice_correlation=df.corr()
    fig, ax = plt.subplots(figsize=(30,25))
    
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(matrice_correlation,annot=True,ax=ax,mask=mask,cmap=sns.color_palette("coolwarm", as_cmap=True),annot_kws={"size": 30})
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 30)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 30)
    st.pyplot(fig)

    st.markdown("""
                - Forte corrélation entre Puisssance administrative et Puissance Max 
                   - Puissance max est plus détaillée, c'est celle que je garde.
                - Forte corrélation entre masse vide min et masse vide max 
                   - Conservation de la moyenne des deux
                - Pour les vitesses, je sépare type de boîte de vitesse et nombre de vitesse
                """)
   