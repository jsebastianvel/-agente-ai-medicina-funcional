import sys
from pathlib import Path

# Allow "streamlit run streamlit_app.py" to find the package without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agente_ai.graph.build_graph import graph
from agente_ai.ingestion.build_index import build_index
from agente_ai.observability.logger import start_trace
from agente_ai.rag.retriever import get_collection

ROUTE_LABELS = {
    "rag_qa": "consulta de conocimiento",
    "symptom_check": "chequeo de sintomas",
    "reject": "fuera de dominio",
}


@st.cache_resource
def ensure_index_built() -> bool:
    """Hosting platforms like Streamlit Community Cloud rebuild the container
    (and its filesystem) on every deploy, so the Chroma index is missing on
    first boot. Building it lazily here makes deploys self-healing without a
    separate CI "publish index" step. @st.cache_resource makes this run once
    per server process, not on every rerun."""
    if get_collection().count() == 0:
        build_index()
    return True


st.set_page_config(page_title="Agente de Medicina Funcional", page_icon="🩺")
st.title("Agente de Medicina Funcional")
st.caption(
    "Portafolio de IA — agente RAG multi-agente. No reemplaza el consejo de un "
    "profesional de la salud."
)

with st.spinner("Preparando la base de conocimiento..."):
    ensure_index_built()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Preguntame sobre medicina funcional...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            _run_id, trace = start_trace()
            result = graph.invoke({"query": query})
        answer = result.get("final_answer", "No pude generar una respuesta.")
        st.markdown(answer)

        route = result.get("route")
        tools = result.get("tool_calls_used") or []
        if route:
            route_label = ROUTE_LABELS.get(route, route)
            tools_label = ", ".join(tools) if tools else "ninguna"
            st.caption(f"Ruta: {route_label} · Herramientas usadas: {tools_label}")

        citations = result.get("citations", [])
        if citations:
            with st.expander("Fuentes"):
                for title in citations:
                    st.markdown(f"- {title}")

        if trace:
            with st.expander("Ver traza del agente"):
                for step in trace:
                    icon = "✅" if step["status"] == "ok" else "❌"
                    st.markdown(f"{icon} **{step['node']}** — {step['duration_ms']} ms")
                    if step.get("output"):
                        st.code(step["output"], language=None)
                    if step.get("error"):
                        st.error(step["error"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
