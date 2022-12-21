import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_chat import message as st_message
import requests
from streamlit_lottie import st_lottie
from streamlit_player import st_player

st.set_page_config(page_title='MonkeyPox Assistant',page_icon='🕊')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
	content:'Made with ❤️ by team NORA'; 
	visibility: visible ;
}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

Menu = option_menu(
    menu_title=None,
    options=["Home","MPX Detector","MPX Chatbot"],
    icons=["peace-fill","question-circle-fill","chat-left-dots-fill"],
    orientation="horizontal"
)

def lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def Home():
    html_temp = """
        <div style="background-color:teal ;padding:3px">
        <h2 style="color:white;text-align:center;">MonkeyPox Assistant</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    animation = "https://assets9.lottiefiles.com/packages/lf20_nmx1GJ.json"
    res_json = lottie_url(animation)
    st_lottie(res_json)

    info = '''Monkeypox (also called mpox by the WHO) is an infectious viral disease that can occur in humans and some other animals. Symptoms include fever, swollen lymph nodes, and a rash that forms blisters and then crusts over. The time from exposure to onset of symptoms ranges from five to twenty-one days. The duration of symptoms is typically two to four weeks. There may be mild symptoms, and it may occur without any symptoms being apparent. The classic presentation of fever and muscle pains, followed by swollen glands, with lesions all at the same stage, has not been found to be common to all outbreaks. Cases may be severe, especially in children, pregnant women or people with suppressed immune systems.'''
    st.info(info)

    st.markdown("----")

    facts = '<p style="text-align:center;"><img src="https://i.postimg.cc/5t5Lh6Cf/15574701425cd51bbeed46a.jpg" width="700"></p>'
    st.markdown(facts, unsafe_allow_html=True)

    st.markdown("----")

    st_player("https://youtu.be/9GziSwQTo4A")

    button = st.button("Click here to know about - Guidelines for management of monkeypox disease by Ministry of Health and Family welfare, Government of India")
    if button:
        import webbrowser
        webbrowser.open_new_tab("shorturl.at/ikHLZ")

def MPX_Detector():
    model = load_model('saved_model.h5')
    categories = ['Monkeypox', 'Measles', 'Normal', 'Chickenpox']

    string1 = '<img src="https://i.postimg.cc/XYyGMJRR/Add-a-little-bit-of-body-text.png" height="120" width="700">'
    st.markdown(string1, unsafe_allow_html=True)
    st.write("****")

    image_file = st.file_uploader('Upload Image',type=["png","jpg","jpeg"])

    if image_file:
        st.image(image_file, use_column_width=True)
        img = image.load_img(image_file,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y = model.predict(x)
        y = categories[np.argmax(y)]
        st.write("***")
        st.error(y)

def MPX_Chatbot():
    head = '<img src="https://i.postimg.cc/3J6ZdC5D/Chatbot.png" height="120" width="700">'
    st.markdown(head, unsafe_allow_html=True)
    st.write("***")

    if 'user_msg' not in st.session_state:
        st.session_state['user_msg'] = []

    if 'bot_msg' not in st.session_state:
        st.session_state['bot_msg'] = []

    question = st.text_input("Talk to the bot: ", key='input')

    suggestion = BOT.Suggestions()
    st.caption("Try asking: {}".format(suggestion))

    if question:
        answer = BOT.ASK(question)

        st.session_state['user_msg'].append(question)
        st.session_state['bot_msg'].append(answer)

    if st.session_state['bot_msg']:
        for i in range(len(st.session_state['bot_msg']) - 1, -1, -1):
            st_message(st.session_state['user_msg'][i], is_user=True, key=str(i) + '_user')
            st_message(st.session_state['bot_msg'][i], key=str(i))

if Menu=="MPX Chatbot":
    import BOT
    MPX_Chatbot()
elif Menu=="MPX Detector":
    MPX_Detector()
elif Menu=="Home":
    Home()
