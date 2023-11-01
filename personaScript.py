import openai
import pandas as pd
from ast import literal_eval
import streamlit as st
from streamlit_chat import message
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import distances_from_embeddings
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import json

with open("config.json") as f:
    config = json.load(f)
    key_vault_url = config["KEY_VAULT_URL"]
    deployment_gpt35 = config["gpt3.5"]
    deployment_embedding = config["embedding_model"]

credential = DefaultAzureCredential()

secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
azure_openai_endpoint = secret_client.get_secret("oai-hera-uksouth-endpoint").value
azure_openai_key = secret_client.get_secret("oai-hera-uksouth-key").value

openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = azure_openai_endpoint
openai.api_key = azure_openai_key

# PATH = "persona.csv"

st.markdown("Main Page")
st.sidebar.markdown("Main Page")


embeddedDataSet = pd.read_csv("customerServiceEmbeddings.csv")
personaList = []


def getContext(user_input, df):
    df["embeddings"] = df["embeddings"].apply(literal_eval)
    q_embedding = get_embedding(text=user_input, engine=deployment_embedding)

    df["distances"] = distances_from_embeddings(
        q_embedding, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    for i, row in df.sort_values("distances", ascending=True).iterrows():
        returns.append("text: " + row["text"])

    return returns, df.sort_values("distances", ascending=True)


@st.cache_data
def generate_response(df, system_prompt, user_prompt, context_length=5):
    st.session_state["messages"] = [{"role": "system", "content": system_prompt}]
    contexts, contextsdf = getContext(user_prompt, df)
    contexts = "\n\n###\n\n".join(contexts[0:context_length])

    prompt = (
        "\n\nContext: "
        + contexts
        + " \n\n---\n\nQuestion:"
        + user_prompt
        + "\n\nAnswer:"
    )

    response = openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        engine=deployment_gpt35,
        max_tokens=3000,
        temperature=0.3,
    )
    st.session_state["messages"].append({"role": "assistant", "content": response})
    return response["choices"][0]["message"]["content"].strip()


def getPersonaPrompt(name):
    personas = {}

    personas[
        "Alex"
    ] = "You are Alex Miller, a talented software engineer who loves tackling complex problems and building innovative solutions. A colleague approaches you for advice on a challenging project they're working on. They need your expertise to navigate a particularly tricky technical issue. They describe the problem and ask for your thoughts on potential solutions. Engage in a conversation with your colleague, offering your insights and suggestions to help them overcome the obstacle. Remember to demonstrate your deep knowledge of software engineering concepts and your enthusiasm for finding creative solutions."
    personas[
        "Sarah"
    ] = "Sarah, the Busy Professional, has relied heavily on her Sky mobile phone for her work since she got it a few weeks ago. She values a plan with unlimited data to access emails and documents constantly without worrying about going over. Being able to reliably access her emails and documents from anywhere is key, so strong network coverage is also a must."
    personas[
        "Mark"
    ] = "Mark is a busy 30-year-old professional who wants to make the most of his downtime by streaming the latest movies and TV shows, playing the latest games, and watching his favorite sports teams. He's looking for an affordable, reliable broadband connection and TV service that can keep up with his entertainment needs. He needs a fast connection with unlimited data and a wide range of channels, on-demand content, and streaming platforms."
    personas[
        "Luke"
    ] = "Luke, a 30-year-old married professional, is an experienced tech user and takes great pride in getting the best value for his money. He works long hours in the city and relies on being able to access fast, reliable broadband services when he's at home and away from the office. Recently, he has noticed that his Sky's fiber optic broadband services have not been running as quickly as he would like, and he's starting to grow very frustrated. He has increased the amount of money he spends each month for these services, but he's not seeing a change in speed. Luke has begun calling customer service with complaints about the slow internet speeds, as well as his dissatisfaction with customer service representatives who don't seem to take his concerns seriously. He wants to make sure he's getting the best service for his money."
    personas[
        "Vikram"
    ] = "You are Vikram Patel, a 75-year-old individual who is facing difficulties with your home Wi-Fi connection. The constant disruptions and lack of internet connectivity have been causing frustration and hindering your ability to stay connected with loved ones. Feeling overwhelmed by technology, you decide to seek assistance to resolve the issue. Engage in a conversation with a friendly tech support representative who offers to help. Clearly express your limitations and lack of technical knowledge while describing the problems you are facing with your Wi-Fi. Seek patient guidance, simple explanations, and step-by-step instructions to help troubleshoot the issue and regain a stable internet connection."

    return personas[name]


def home():
    PATH = "persona.csv"

    # def getPersona(PATH, idx):
    #     personaData = pd.read_csv(PATH, sep=";", encoding='cp1252')
    #     personaData.index = [x for x in range(1, len(personaData.values)+1)]

    #     return personaData.iloc[idx, :].to_json()

    # init_persona = getPersona(PATH, 16)

    st.title("User Persona Chat Assistant")
    st.markdown(
        "This is the User Persona Chat Assistant. It is capable of conversations with an user as users of sky products with vastly different buying habits and needs."
    )
    st.markdown(
        "This application leverages openAI GPT-3.5-Turbo model that has been fine-tuned with large scale customer service dataset embeddings from Text-Ada-Embedding-002."
    )

    clear_button = st.button("Refresh", key="clear")
    name = st.selectbox("Select a customer:", ("Sarah", "Luke", "Vikram"))
    names = ["Sarah", "Luke", "Vikram"]

    for item in names:
        if name == item:
            st.write("You are now chatting with", name)
            init_persona = getPersonaPrompt(item)

    system_prompt = f"""

    You are a customer who uses a lot of sky media company products. You have to do the following:
    - Learn what you can from the web on sky and their product offerings.
    - Learn how customers complain from the context provided.
    - Understand the customer you will impersonate from the following persona profile: {init_persona}
    - Do not under any circumstances use offensive language.
    - Format your answer in the following way: 
            - Respond as the customer, but dont mention it.
            - If you can't understand the question, please ask the user to rephrase the question.

"""

    if clear_button:
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["messages"] = [{"role": "system", "content": system_prompt}]

    text_container = st.container()
    response_container = st.container()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": system_prompt}]

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    with text_container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area("You:", key="input", height=100)
            submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = generate_response(
            embeddedDataSet, system_prompt, user_input, context_length=5
        )
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def main():
    home()


if __name__ == "__main__":
    main()
