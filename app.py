
import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="TechNova Support", page_icon="💬")
st.title("💬 TechNova Customer Support")
st.caption("Ask about orders, returns, warranty, or shipping.")

@st.cache_resource
def get_model():
    return ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key=st.secrets["NVIDIA_API_KEY"]
    )

if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("How can we help you?")

if question:
    st.session_state.history.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    try:
        system_prompt = """
        You are TechNova customer support.
        Mock orders:
        TN1001: Laptop Pro 14, Shipped, ETA 2 days.
        TN1002: Noise-cancel Earbuds, Processing, ETA 5 days.
        TN1003: Phone X, Delivered.
        Policies:
        Returns within 30 days in original packaging.
        Manufacturer warranty: 1 year.
        Shipping is free above Rs. 999.
        Standard delivery takes 3-5 days.
        Answer politely and do not invent order information.
        """
        chat_messages = [SystemMessage(content=system_prompt)]
        for item in st.session_state.history:
            if item["role"] == "user":
                chat_messages.append(HumanMessage(content=item["content"]))
            else:
                chat_messages.append(AIMessage(content=item["content"]))

        answer = get_model().invoke(chat_messages).content
        st.session_state.history.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
    except Exception:
        st.error("Sorry, support is temporarily unavailable. Please try again.")
