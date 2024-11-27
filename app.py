import contextlib
import json
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from speak import database
from speak.chat_engine import chat
from speak.speech_to_text import transcribe  # Ensure this supports multi-language transcription
from speak.spell_check import grammar_coherence_correction
from speak.text_to_speech import generate_audio

# Add language options
LANGUAGE_OPTIONS = {
    "English": "en",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Tamil": "ta",
    "French": "fr",
    "German": "de",
    "Japanese": "ja"
}

def answers(audio_bytes, chat_identifier, selected_language):
    with open("tmp_file.wav", "wb") as file:
        file.write(audio_bytes)
    
    # Pass the selected language to the transcribe function
    audio_transcribe = transcribe(language=selected_language)  # Change to 'language'
    
    message_corrected = grammar_coherence_correction(audio_transcribe)
    database.insert_message(
        chat_id=chat_identifier, role="user", content=message_corrected, audio=audio_bytes,
    )
    all_messages = database.get_messages_by_chat_id(chat_identifier)
    all_messages = all_messages[::-1]
    clean_messages = []
    for msg in all_messages:
        msg_content = json.loads(msg[3])
        if msg[2] in ["system", "assistant"]:
            clean_messages.append(msg_content)
        elif msg[2] == "user":
            clean_messages.append({"role": "user", "content": msg_content["rewritten"]})
    
    response = chat(clean_messages)
    response_audio = generate_audio(response["content"])
    database.insert_message(
        chat_id=chat_identifier,
        role="assistant",
        content=response,
        audio=response_audio,
    )

# App configuration
st.set_page_config(
    page_title="TikTalk", page_icon="🧊", layout="wide",
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&display=swap');
        
        h1 {
            font-family: "IBM Plex Serif", serif;
            font-weight: 700;
            font-style: normal;
            text-align: center;
            color: #49def2;
            background: linear-gradient(to right, #5182ED, #8D75D1, #BE6B97, #D96570);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <h1>TickTalk : Personalized Language Buddy</h1>
""", unsafe_allow_html=True)

# Sidebar for Chat Management (Toggleable)
with st.sidebar:
    st.markdown(f"""
            <h2 style='text-align: left; color: #B784B7; font-weight: 600; font-size:24px;'>
                Chat Management
            </h2>
        """, unsafe_allow_html=True)
    open_sidebar = st.checkbox("", value=True)

    if open_sidebar:
        with st.expander("Select Chat"):
            chats = database.get_all_chats()
            names = [chat[1] for chat in chats]
            chat_id = None
            if names:
                selected_chat = st.selectbox("Select chat", names)
                chat_id = chats[names.index(selected_chat)][0]

        with st.expander("Create a New Chat"):
            name = st.text_input("Chat name", "")
            prompt = st.text_area("Chat Prompt", "", help="Write a prompt for the chatbot")
            if st.button("Create"):
                if name in names:
                    st.error("Chat already exists")
                else:
                    chat_id = database.insert_chat(name)
                    content = {"role": "system", "content": prompt}
                    database.insert_message(
                        chat_id=chat_id, role="system", content=content, audio="NULL"
                    )
                    st.success("Chat created")

        # Add horizontal line
        st.markdown("---")

        with st.expander("Delete Chat"):
            if st.button("Delete all messages"):
                if chat_id:
                    database.delete_messages_by_chat_id(chat_id)
                    st.success("All messages deleted")

            if st.button("Delete chat"):
                if chat_id:
                    database.delete_chat(chat_id)
                    st.success("Chat deleted")

# Chat layout and messages
st.markdown(
    """
    <style>
        .chat-container {
            width: 100%;
            padding: 10px;
        }
        .message-bubble {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: inline-block;
            max-width: 90%;
        }
        .user-bubble {
            background-color: #16325B;
            color: white;
            float: right;
            text-align: right;
            clear: both;
        }
        .assistant-bubble {
            background-color: #587DBD;
            color: white;
            float: left;
            text-align: left;
            clear: both;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Language selection for transcription
st.write("Select the language for your transcription:")
selected_language = st.selectbox("Choose language", options=LANGUAGE_OPTIONS.keys())
language_code = LANGUAGE_OPTIONS[selected_language]

if chat_id:
    st.markdown(f"""
        <h2 style='text-align: left; color: #B784B7; font-weight: 600;'>
            {selected_chat}
        </h2>
    """, unsafe_allow_html=True)
    
    st.write("Record your voice:")
    c1, c2 = st.columns([8, 2])
    
    with c1:
        if audio := mic_recorder(start_prompt="🎙️ Record", stop_prompt="⭕ Stop", key="recorder"):
            st.audio(audio["bytes"])
    
    with c2:
        if st.button("Send"):
            answers(audio["bytes"], chat_id, language_code)  # Pass selected language

    # Display chat messages
    messages = database.get_messages_by_chat_id(chat_id)
    for index, message in enumerate(messages):
        role = message[2]
        content = json.loads(message[3])
        audio = message[4]

        if role == "system":
            continue

        # Style user and assistant messages
        if role == "user":
            bubble_class = "user-bubble"
            bubble_content = content["original"]
            button_key = f"show_score_{index}"
        else:
            bubble_class = "assistant-bubble"
            bubble_content = content["content"]

        st.markdown(f"""
            <div class="chat-container">
                <div class="message-bubble {bubble_class}">
                    <strong>{role.capitalize()}:</strong><br>
                    {bubble_content}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.audio(audio)

        if role == "user":
            if st.button(f"Show Score", key=button_key):
                st.write(f"**Score:** {content.get('score', 'N/A')}")
                st.write(f"Original: {content.get('original', 'N/A')}")
                st.write(f"Grammar corrected: {content.get('grammar_corrected', 'N/A')}")
                st.write(f"Coherence corrected: {content.get('coherence_corrected', 'N/A')}")
                st.write(f"Suggestion: {content.get('rewritten', 'N/A')}")
    