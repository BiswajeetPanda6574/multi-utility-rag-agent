import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def switch_thread(thread_id):
    st.session_state["thread_id"] = str(thread_id)
    messages = load_conversation(thread_id)
    
    temp_messages = []
    for msg in messages:
        # Skip raw tool outputs and empty AI reasoning steps
        if not msg.content or isinstance(msg, ToolMessage):
            continue
        
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
        
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(thread_id), {})

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# We define CONFIG early so we can check the graph state before rendering chat input
CONFIG = {
    "configurable": {"thread_id": thread_key},
    "metadata": {"thread_id": thread_key},
    "run_name": "chat_turn",
}

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        st.sidebar.button(
            str(thread_id), 
            key=f"side-thread-{thread_id}", 
            on_click=switch_thread, 
            args=(thread_id,)
        )

# ============================ Main Layout ========================
st.title("📚 Multi-Utility RAG Agent")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# HITL: Check if the graph is paused waiting for approval
# ---------------------------------------------------------
current_state = chatbot.get_state(CONFIG)
is_paused = current_state.next and "tools" in current_state.next

if is_paused:
    st.warning("⚠️ The agent wants to execute a tool. Do you approve?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Tool"):
            with st.chat_message("assistant"):
                status_holder = {"box": None}

                def resume_stream():
                    # Pass None to resume the graph from where it paused
                    for message_chunk, _ in chatbot.stream(
                        None, 
                        config=CONFIG, 
                        stream_mode="messages"
                    ):
                        if isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(
                                    f"🔧 Using `{tool_name}` …", expanded=True
                                )
                            else:
                                status_holder["box"].update(
                                    label=f"🔧 Using `{tool_name}` …",
                                    state="running",
                                    expanded=True,
                                )

                        if isinstance(message_chunk, AIMessage) and message_chunk.content:
                            yield message_chunk.content

                ai_message = st.write_stream(resume_stream())

                if status_holder["box"] is not None:
                    status_holder["box"].update(
                        label="✅ Tool finished", state="complete", expanded=False
                    )

            if ai_message:
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": ai_message}
                )
            st.rerun()
            
    with col2:
        if st.button("❌ Reject Tool"):
            st.error("Tool execution rejected. Please start a New Chat to clear the state.")

else:
    # ---------------------------------------------------------
    # Normal Chat Input
    # ---------------------------------------------------------
    user_input = st.chat_input("Ask about your document or use tools")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            status_holder = {"box": None}

            def ai_only_stream():
                for message_chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

        if ai_message:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_message}
            )
        st.rerun()

doc_meta = thread_document_metadata(thread_key)
if doc_meta:
    st.caption(
        f"Document indexed: {doc_meta.get('filename')} "
        f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
    )

st.divider()
