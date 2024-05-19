import streamlit as st
import os.path
from collections import OrderedDict
from streamlit_option_menu import option_menu
# Define TITLE, TEAM_MEMBERS and PROMOTION values, in config.py.
import config
import os

# Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
else:
    st.session_state.sidebar_state = 'auto'

st.set_page_config (
    page_title=config.TITLE,
    page_icon= "assets/faviconV2.png",
    initial_sidebar_state=st.session_state.sidebar_state
)

def is_running_in_docker():
    return os.path.exists('/.dockerenv')

if 'username' not in st.session_state:
    st.session_state.username = ""
    st.session_state.UserFirstName = ""
    st.session_state.UserLastName = ""
    st.session_state.token = ""
    st.session_state.UserAuthorization = 0

# Si l'application tourne avec Docker, session_state.docker == True 
# Sinon elle tourne localement, ==False
# En fonction de la valeur de varible précédente, le (pre) path est différent 
st.session_state.docker = os.path.exists('/.dockerenv')
if st.session_state.docker: 
    st.session_state.PrePath = ""
    st.session_state.users_db = "users_db"
    st.session_state.api_oauth = "api_oauth"
    st.session_state.api_predict = "api_predict"
    st.session_state.api_train = "api_train"
    st.session_state.api_flows = "api_flows"

else: 
    st.session_state.PrePath = "../"
    st.session_state.users_db = "localhost"
    st.session_state.api_oauth = "localhost"
    st.session_state.api_predict = "localhost"
    st.session_state.api_train = "localhost"
    st.session_state.api_flows = "localhost"


# Define the root folders depending on local/cloud run
# thisfile = os.path.abspath(__file__)
# if ('/' in thisfile): 
#     os.chdir(os.path.dirname(thisfile))

# Nécessaire pour la version windows 11
if st.session_state.docker: 
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Tabs in the ./tabs folder, imported here.
from tabs import intro, user_login, sales, training, production_start_up


with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# Add tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (user_login.sidebar_name, user_login),
        (sales.sidebar_name, sales),
        (training.sidebar_name, training),
        (production_start_up.sidebar_name, production_start_up),
        #(modelisation_seq2seq_tab.sidebar_name, modelisation_seq2seq_tab),
        #(game_tab.sidebar_name, game_tab ),
    ]
)


def run():

    # st.sidebar.image(
    #     "assets/logo_rakuten.png",
    #     width=270,
    # )
    iframe_code = """
    <div style="width:100%;height:0;padding-bottom:60%;position:relative;"><iframe src="https://giphy.com/embed/ZGdpXBABdYIqc4rMsE" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></div><p></p>
    """
    # Display the iframe in the sidebar using markdown
    st.sidebar.markdown(iframe_code, unsafe_allow_html=True)
    custom_css = """
    <style>
    .eczjsme4 {
        padding-top: 0px !important;
    }
    </style>
    """
    st.sidebar.markdown(custom_css, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"<span style='color:red; font-weight:bold;font-size:1.8em;'>{st.session_state.UserFirstName} {st.session_state.UserLastName}</span>", unsafe_allow_html=True)

    st.sidebar.write("")
    with st.sidebar:
        tab_name = option_menu(None, list(TABS.keys()),
                               # icons=['house', 'bi-binoculars', 'bi bi-graph-up', 'bi-chat-right-text','bi-book', 'bi-body-text'], menu_icon="cast", default_index=0,
                               icons=['house', 'binoculars', 'graph-up', 'search','book', 'chat-right-text','controller'], menu_icon="cast", default_index=0,
                               styles={"container": {"padding": "0!important","background-color": "#10b8dd", "border-radius": "0!important"},
                                       "nav-link": {"font-size": "1rem", "text-align": "left", "margin":"0em", "padding": "0em",
                                                    "padding-left": "0.2em", "--hover-color": "#eee", "font-weight": "400",
                                                    "font-family": "Source Sans Pro, sans-serif"}
                                        })
    # tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]
    tab.run()

if __name__ == "__main__":
    run()
