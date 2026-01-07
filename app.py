import streamlit as st
import pandas as pd
from functools import reduce
from collections import Counter
from io import BytesIO
import joblib
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

st.header('RH da empresa X')

# Inicializa a sessão
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = {}

# Upload
with st.expander('Upload de Arquivos'):
    uploaded_file = st.file_uploader("Escolha um arquivo excel", type=['xls', 'xlsx', 'xlsm', 'xlsb'])

    if uploaded_file is not None:
        xls = pd.ExcelFile(BytesIO(uploaded_file.read()))
        filepaths = {uploaded_file.name: xls}
        st.session_state['uploaded_files'].update(filepaths)

    if st.session_state['uploaded_files']:
        remove_file = st.selectbox('Select a file to remove', list(st.session_state['uploaded_files']))
        if st.button('Remove File'):
            st.session_state['uploaded_files'].pop(remove_file)

data = {}
original_column_names = {}
df = {}

# parse string
for path, xls in st.session_state['uploaded_files'].items():
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name, dtype=str)
        original_column_names.update({col.lower().replace(' ', '_').replace(r'\W', ''): col for col in df.columns})
        df.columns = df.columns.str.replace(' ', '_').str.replace(r'\W', '')
        data[f'{path}_{sheet_name}'] = df

column_names = [col for df in data.values() for col in df.columns]

#tabela
st.dataframe(df)

#gemini
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Cria uma pasta para salvar dados
try:
    os.mkdir('data/')
except:
    #se ja existe
    pass

# Load chats passados
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar
with st.sidebar:
    st.write('# Historico de conversas')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Escolha um chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'Novo chat'),
            placeholder='_',
        )
    else:
        # primeira vez
        st.session_state.chat_id = st.selectbox(
            label='Escolha um chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'Novo chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    
    # inicializa o id
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'


# Mantem o cache da conversa atual
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []
    print('new_cache made')
st.session_state.model = genai.GenerativeModel('gemini-flash-latest')
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)

# Display
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

# User input
if prompt := st.chat_input('Escreva sua pergunta sobre o arquivo adicionado'):

    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')

    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )


    dados_texto = df.to_string(index=False)

    prompt_completo = f"""
    Você é um assistente de RH.

    Use APENAS os dados abaixo para responder à pergunta do usuário.
    Se a resposta não estiver nos dados, diga:
    "Não sei com base nos dados fornecidos."

    DADOS DA PLANILHA:
    {dados_texto}

    PERGUNTA DO USUÁRIO:
    {prompt}
    """

    response = st.session_state.chat.send_message(
        prompt_completo,
        stream=True,
    )
# Display assistente
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''

        for chunk in response:
            if hasattr(chunk, "text"):
                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    time.sleep(0.05)
                    message_placeholder.write(full_response + '▌')

        message_placeholder.write(full_response)

    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=full_response,
            avatar=AI_AVATAR_ICON,
        )
    )

    st.session_state.gemini_history = st.session_state.chat.history

    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )