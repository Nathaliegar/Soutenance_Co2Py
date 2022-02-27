from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import intro, dataviz, modelisation, demonstration


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

#with open("style.css", "r") as f:
#    style = f.read()
 style="""
 h1,
h2,
h3,
h4 {
  color: #000000;
}

code {
  color: #1ec3bc;
}

#MainMenu {
  display: none;
}

div[data-testid="stDecoration"] {
  display: none;
}

footer {
  display: none;
}

/* Radio buttons */

.st-cc {
  color: black;
  font-weight: 500;
}

/* Sidebar */

section[data-testid="stSidebar"] > div {
  background-color: #10b8dd;
  padding-top: 2rem;
  padding-left: 1.5rem;
}

section[data-testid="stSidebar"] button[title="View fullscreen"] {
  display: none;
}

section[data-testid="stSidebar"] button[kind="icon"] {
  display: none;
}

section[data-testid="stSidebar"] .st-bk {
  background-color: white;
}

section[data-testid="stSidebar"] .st-c0 {
  background-color: black;
}

section[data-testid="stSidebar"] hr {
  margin-top: 30px;
  border-color: white;
  width: 50px;
}

section[data-testid="stSidebar"] h2 {
  color: white;
}

/* Images */

button[title="View fullscreen"] {
  display: none;
}

/* hr */

hr {
  width: 200px;
  border-width: 5px;
  border-color: #10b8dd;
  margin-top: 0px;
}

/* First Page */

section[tabindex="0"] .block-container {
  padding-top: 0px;
  padding-bottom: 0px;
}
 """


st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (dataviz.sidebar_name, dataviz),
        (modelisation.sidebar_name, modelisation),
        (demonstration.sidebar_name, demonstration)
    ]
)



def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Présenté par:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]
    
    tab.run()
    


if __name__ == "__main__":
    run()
