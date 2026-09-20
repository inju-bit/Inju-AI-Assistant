
# ======================================================
# IMPORT LIBRARIES
# ======================================================

# PYTHON STANDARD LIBRARY
# os reads environment variables.
# uuid generates unique conversation IDs.
import os
import uuid

# STREAMLIT
# Creates the web application and chat interface.
import streamlit as st

# LANGCHAIN CORE
# HumanMessage represents a message from the user.
from langchain_core.messages import HumanMessage

# LANGCHAIN OPENAI
# ChatOpenAI connects our chatbot to an OpenAI model.
from langchain_openai import ChatOpenAI

# LANGGRAPH
# create_react_agent creates our AI agent.
from langgraph.prebuilt import create_react_agent

# LANGGRAPH
# InMemorySaver stores conversation history in RAM.
from langgraph.checkpoint.memory import InMemorySaver

# PYTHON-DOTENV
# load_dotenv loads environment variables from .env.
from dotenv import load_dotenv


# ======================================================
# STEP 1: LOAD THE API KEY
# ======================================================

# Load variables from the local .env file.
load_dotenv()

# Read the API key from the environment.
# This works locally with .env and on Streamlit Cloud
# when OPENAI_API_KEY is configured as a root-level secret.
api_key = os.getenv("OPENAI_API_KEY")


# ======================================================
# STEP 2: CONFIGURE THE STREAMLIT PAGE
# ======================================================

# STREAMLIT FUNCTION
# Configure the page title, icon and layout.
st.set_page_config(
    page_title="Inju AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# STREAMLIT FUNCTIONS
# Display the application title and description.
st.title("🤖 Inju AI Assistant")

st.write(
    "Welcome! I am your AI assistant. "
    "Ask me questions and I will remember our conversation."
)


# ======================================================
# STEP 3: CHECK THE API KEY
# ======================================================

# PYTHON IF STATEMENT
# Stop the application if the API key is missing.
if not api_key:

    # STREAMLIT FUNCTION
    # Display an error on the website.
    st.error(
        "OpenAI API key is missing. "
        "Add OPENAI_API_KEY to your .env file "
        "or your Streamlit Cloud secrets."
    )

    # STREAMLIT FUNCTION
    # Stop executing the application.
    st.stop()


# ======================================================
# STEP 4: CREATE THE AI AGENT WITH MEMORY
# ======================================================

# STREAMLIT SESSION STATE
#
# Streamlit reruns the script when the user interacts
# with the application.
#
# st.session_state preserves variables between reruns
# within the same browser session.
#
# We create our agent only once per session.

if "agent_executor" not in st.session_state:

    # LANGCHAIN OPENAI
    # Create an instance of the ChatOpenAI class.
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=api_key
    )

    # PYTHON LIST
    # Create an empty list of tools.
    # You can add custom tools later.
    tools = []

    # LANGGRAPH
    # Create a memory/checkpointer object.
    memory = InMemorySaver()

    # LANGGRAPH
    # Create the agent using the model, tools and memory.
    agent_executor = create_react_agent(
        model,
        tools,
        checkpointer=memory
    )

    # STREAMLIT
    # Save the agent in session state so it survives reruns.
    st.session_state.agent_executor = agent_executor

    # PYTHON STANDARD LIBRARY
    # uuid.uuid4() generates a unique identifier.
    #
    # str() converts the identifier into a string.
    #
    # Each browser session gets its own conversation ID.
    st.session_state.thread_id = str(uuid.uuid4())


# ======================================================
# STEP 5: CREATE THE DISPLAYED CHAT HISTORY
# ======================================================

# STREAMLIT
# Create an empty list for displaying chat messages.
#
# This is separate from LangGraph's conversation memory.
# Streamlit uses this list to display chat bubbles.
#
# LangGraph uses its checkpointer to give previous
# messages to the AI model.

if "messages" not in st.session_state:

    # PYTHON LIST
    st.session_state.messages = []


# ======================================================
# STEP 6: CREATE THE CONVERSATION CONFIGURATION
# ======================================================

# PYTHON DICTIONARY
#
# "configurable" is a configuration key used by LangGraph.
#
# "thread_id" identifies the conversation to retrieve.
#
# Using the same thread_id allows the agent to remember
# the previous messages from this conversation.

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ======================================================
# STEP 7: CREATE THE SIDEBAR
# ======================================================

# STREAMLIT
# Create a sidebar with a new conversation button.

with st.sidebar:

    st.header("Chat settings")

    # STREAMLIT BUTTON
    if st.button("➕ New Chat"):

        # PYTHON STANDARD LIBRARY
        # Generate a new conversation ID.
        st.session_state.thread_id = str(uuid.uuid4())

        # Clear the displayed conversation history.
        st.session_state.messages = []

        # STREAMLIT
        # Rerun the application with the new conversation.
        st.rerun()

    st.divider()

    st.write("AI Model: GPT-4.1 mini")
    st.write("Memory: Enabled")


# ======================================================
# STEP 8: DISPLAY PREVIOUS CONVERSATIONS
# ======================================================

# PYTHON FOR LOOP
#
# Go through the messages in Streamlit session state.
#
# message is our own variable.
# Each message is a Python dictionary.

for message in st.session_state.messages:

    # STREAMLIT
    # Create a chat bubble for the message.
    #
    # message["role"] determines whether the bubble
    # belongs to the user or the assistant.

    with st.chat_message(message["role"]):

        # STREAMLIT
        # Display the message text.
        st.markdown(message["content"])


# ======================================================
# STEP 9: CREATE THE USER INPUT BOX
# ======================================================

# STREAMLIT FUNCTION
#
# st.chat_input() creates a text input box
# at the bottom of the chat interface.
#
# When the user submits a message, the function
# returns the text as a Python string.
#
# user_input is our own variable.

user_input = st.chat_input(
    "Ask me anything..."
)


# ======================================================
# STEP 10: PROCESS THE USER'S MESSAGE
# ======================================================

# PYTHON IF STATEMENT
# Run this block when the user submits a message.

if user_input:

    # STREAMLIT + PYTHON
    #
    # Save the user's message in the displayed chat history.
    #
    # .append() is a built-in Python list method.
    #
    # "role" and "content" are dictionary keys.

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # ==================================================
    # STEP 11: DISPLAY THE USER'S MESSAGE
    # ==================================================

    # STREAMLIT
    # Display the user's message in a chat bubble.

    with st.chat_message("user"):

        st.markdown(user_input)


    # ==================================================
    # STEP 12: GENERATE THE AI RESPONSE
    # ==================================================

    # STREAMLIT
    # Create an assistant chat bubble.

    with st.chat_message("assistant"):

        # STREAMLIT
        # Display a loading indicator while the AI responds.

        with st.spinner("Thinking..."):

            # PYTHON ERROR HANDLING
            # try/except handles errors from the API.
            try:

                # LANGCHAIN CORE
                # Convert the user's question into
                # a HumanMessage object.

                human_message = HumanMessage(
                    content=user_input
                )


                # ==========================================
                # STEP 13: CALL THE LANGGRAPH AGENT
                # ==========================================

                # LANGGRAPH
                #
                # invoke() runs the AI agent.
                #
                # {"messages": [human_message]}
                # sends the latest user message.
                #
                # config=config supplies the thread_id.
                #
                # LangGraph uses this conversation ID
                # to retrieve previous messages
                # from the memory/checkpointer.
                #
                # The result contains the updated agent
                # state, including its conversation messages.

                result = (
                    st.session_state.agent_executor.invoke(
                        {
                            "messages": [human_message]
                        },
                        config=config
                    )
                )


                # ==========================================
                # STEP 14: EXTRACT THE AI RESPONSE
                # ==========================================

                # PYTHON DICTIONARY ACCESS
                #
                # result["messages"] retrieves the
                # conversation messages from the agent state.
                #
                # [-1] selects the last message in the list.
                #
                # .content is an attribute of the
                # LangChain message object.
                #
                # For our text chatbot, it contains
                # the AI-generated answer.

                ai_response = result["messages"][-1].content


                # ==========================================
                # STEP 15: DISPLAY THE AI RESPONSE
                # ==========================================

                # STREAMLIT
                # Display the AI response in the chat bubble.

                st.markdown(ai_response)


                # ==========================================
                # STEP 16: SAVE THE DISPLAYED AI RESPONSE
                # ==========================================

                # STREAMLIT + PYTHON LIST
                #
                # Save the AI response so that it remains
                # visible when Streamlit reruns the script.

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response
                })


            # PYTHON EXCEPTION HANDLING
            #
            # Catch API or execution errors without crashing
            # the entire web application.

            except Exception:

                # STREAMLIT
                # Display a simple error message.

                st.error(
                    "Something went wrong. "
                    "Check your OpenAI API key, "
                    "API balance and internet connection."
                )
