import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

st.set_page_config(page_title="Telefonia RAG", layout="wide")
st.title("Telefonia RAG (sin LLM generativo)")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

full_reindex = st.checkbox("Reindexar completo", value=False)
if st.button("Indexar documentos"):
    try:
        with httpx.Client(timeout=300) as client:
            ingest_response = client.post(
                f"{API_BASE_URL}/ingest", params={"rebuild": str(full_reindex).lower()}
            )
            ingest_response.raise_for_status()
            ingest_data = ingest_response.json()
    except httpx.HTTPError as exc:
        st.error(f"No se pudo indexar: {exc}")
    else:
        st.success(
            "Indexación ejecutada."
        )
        st.json(ingest_data)

user_message = st.text_area("Tu consulta", height=120)

col1, col2 = st.columns(2)
send = col1.button("Enviar", type="primary")
reset = col2.button("Nueva conversación")

if reset:
    st.session_state.conversation_id = None
    st.success("Conversación reiniciada")

if send:
    if not user_message.strip():
        st.warning("Escribe una consulta primero.")
    else:
        def send_chat(conversation_id: str | None):
            payload = {"conversation_id": conversation_id, "content": user_message.strip()}
            with httpx.Client(timeout=60) as client:
                return client.post(f"{API_BASE_URL}/chat", json=payload)

        try:
            response = send_chat(st.session_state.conversation_id)
            if response.status_code == 404 and st.session_state.conversation_id:
                st.session_state.conversation_id = None
                response = send_chat(None)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            st.error(f"No se pudo consultar el backend: {exc}")
        else:
            st.session_state.conversation_id = data["conversation_id"]
            st.subheader("Respuesta")
            st.write(data["answer"])
            st.subheader("Citas")
            if not data["citations"]:
                st.info("Sin citas para esta respuesta.")
            for c in data["citations"]:
                st.markdown(
                    f"- **{c['source_path']}** (score: {c['score']:.4f})\n\n"
                    f"  {c['snippet']}"
                )

if st.session_state.conversation_id:
    st.divider()
    st.subheader("Historial")
    try:
        with httpx.Client(timeout=30) as client:
            history_response = client.get(
                f"{API_BASE_URL}/history/{st.session_state.conversation_id}"
            )
            history_response.raise_for_status()
            history_items = history_response.json()
    except httpx.HTTPError as exc:
        st.error(f"No se pudo obtener historial: {exc}")
    else:
        for item in history_items:
            with st.chat_message(
                "assistant" if item["role"] == "assistant" else "user"
            ):
                st.write(item["content"])
