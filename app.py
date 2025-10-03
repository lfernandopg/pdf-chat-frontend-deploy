"""
Frontend Streamlit para RAG PDF Chat API
Conecta con el backend FastAPI para gestionar conversaciones con PDFs
"""
import os
import time
import requests
import streamlit as st
from datetime import datetime
from typing import Optional, List
import json


# --- Configuración de la API ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_TIMEOUT = 300  # 5 minutos para operaciones largas


# --- Configuración de idiomas ---
LANGUAGES = {
    "es": {
        "title": "💬 Chatea con tus PDFs",
        "subtitle": "Sube documentos PDF y haz preguntas usando IA con tecnología RAG",
        "upload_section": "📄 Gestión de Documentos",
        "upload_label": "Arrastra y suelta tus archivos PDF aquí",
        "upload_help": "Se permiten múltiples archivos. Máximo 50 páginas por PDF.",
        "chat_section": "💭 Conversación",
        "settings": "⚙️ Configuración",
        "advanced_settings": "🔧 Ajustes Avanzados",
        "model_selection": "🤖 Modelo de IA",
        "select_model": "Selecciona el modelo:",
        "embedding_model": "Modelo de Embeddings:",
        "process_button": "🚀 Crear Nuevo Chat",
        "add_docs_button": "➕ Añadir Documentos",
        "processing": "🔄 Procesando documentos...",
        "success": "✅ Operación exitosa",
        "error": "❌ Error",
        "error_api": "❌ Error de conexión con la API",
        "warning_upload": "⚠️ Por favor, crea un chat primero cargando documentos",
        "chat_placeholder": "Escribe tu pregunta sobre los documentos...",
        "thinking": "🤔 Analizando documentos...",
        "language": "🌐 Idioma",
        "stats_title": "📊 Estadísticas del Chat",
        "chat_id": "ID del Chat",
        "pages": "Documentos",
        "chunks": "Fragmentos procesados",
        "created_at": "Creado",
        "clear_chat": "🗑️ Limpiar Conversación",
        "delete_chat": "🗑️ Eliminar Chat Completo",
        "download_chat": "💾 Guardar Conversación",
        "download_btn": "Descargar",
        "chat_cleared": "✅ Conversación limpiada",
        "chat_deleted": "✅ Chat eliminado completamente",
        "no_messages": "⚠️ No hay mensajes para descargar",
        "about": "ℹ️ Acerca de",
        "about_text": "Esta aplicación utiliza tecnología RAG (Retrieval Augmented Generation) conectada a un backend FastAPI profesional para responder preguntas sobre documentos PDF.",
        "api_status": "Estado de la API",
        "api_online": "🟢 Online",
        "api_offline": "🔴 Offline",
        "temperature": "Temperatura",
        "max_tokens": "Tokens Máximos",
        "confirm_delete": "⚠️ ¿Estás seguro de eliminar el chat completo? Esta acción no se puede deshacer.",
        "yes": "Sí",
        "no": "No",
    },
    "en": {
        "title": "💬 Chat with your PDFs",
        "subtitle": "Upload PDF documents and ask questions using AI with RAG technology",
        "upload_section": "📄 Document Management",
        "upload_label": "Drag and drop your PDF files here",
        "upload_help": "Multiple files allowed. Maximum 50 pages per PDF.",
        "chat_section": "💭 Conversation",
        "settings": "⚙️ Settings",
        "advanced_settings": "🔧 Advanced Settings",
        "model_selection": "🤖 AI Model",
        "select_model": "Select model:",
        "embedding_model": "Embedding Model:",
        "process_button": "🚀 Create New Chat",
        "add_docs_button": "➕ Add Documents",
        "processing": "🔄 Processing documents...",
        "success": "✅ Operation successful",
        "error": "❌ Error",
        "error_api": "❌ API connection error",
        "warning_upload": "⚠️ Please create a chat first by uploading documents",
        "chat_placeholder": "Ask a question about the documents...",
        "thinking": "🤔 Analyzing documents...",
        "language": "🌐 Language",
        "stats_title": "📊 Chat Statistics",
        "chat_id": "Chat ID",
        "pages": "Documents",
        "chunks": "Processed chunks",
        "created_at": "Created",
        "clear_chat": "🗑️ Clear Conversation",
        "delete_chat": "🗑️ Delete Entire Chat",
        "download_chat": "💾 Save Conversation",
        "download_btn": "Download",
        "chat_cleared": "✅ Conversation cleared",
        "chat_deleted": "✅ Chat completely deleted",
        "no_messages": "⚠️ No messages to download",
        "about": "ℹ️ About",
        "about_text": "This application uses RAG (Retrieval Augmented Generation) technology connected to a professional FastAPI backend to answer questions about PDF documents.",
        "api_status": "API Status",
        "api_online": "🟢 Online",
        "api_offline": "🔴 Offline",
        "temperature": "Temperature",
        "max_tokens": "Max Tokens",
        "confirm_delete": "⚠️ Are you sure you want to delete the entire chat? This action cannot be undone.",
        "yes": "Yes",
        "no": "No",
    }
}


def get_text(key: str, lang: str) -> str:
    """Get translated text"""
    return LANGUAGES[lang].get(key, key)


def check_api_health() -> bool:
    """Check if API is reachable"""
    try:
        response = requests.get(
            f"{API_BASE_URL.replace('/api/v1', '')}/health",
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def create_chat(files: List, embedding_model: str, lang: str) -> Optional[dict]:
    """Create a new chat with uploaded documents"""
    try:
        files_data = [
            ('files', (file.name, file.getvalue(), 'application/pdf'))
            for file in files
        ]
        
        params = {"model_name": embedding_model} if embedding_model else {}
        
        response = requests.post(
            f"{API_BASE_URL}/chats/upload",
            files=files_data,
            params=params,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"{get_text('error', lang)}: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.Timeout:
        st.error(f"{get_text('error', lang)}: Timeout - El procesamiento tardó demasiado")
        return None
    except Exception as e:
        st.error(f"{get_text('error_api', lang)}: {str(e)}")
        return None


def check_chat_status(chat_id: str) -> Optional[dict]:
    """Check chat status"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/chats/{chat_id}/status",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def add_documents_to_chat(chat_id: str, files: List, lang: str) -> Optional[dict]:
    """Add documents to existing chat"""
    try:
        files_data = [
            ('files', (file.name, file.getvalue(), 'application/pdf'))
            for file in files
        ]
        
        response = requests.post(
            f"{API_BASE_URL}/chats/{chat_id}/add-documents",
            files=files_data,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"{get_text('error', lang)}: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"{get_text('error_api', lang)}: {str(e)}")
        return None


def ask_question(
    chat_id: str,
    prompt: str,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    lang: str
) -> Optional[dict]:
    """Ask a question to the chat"""
    try:
        payload = {
            "prompt": prompt,
            "llm_model_name": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chats/{chat_id}/ask",
            json=payload,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"{get_text('error', lang)}: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"{get_text('error_api', lang)}: {str(e)}")
        return None


def delete_chat(chat_id: str, lang: str) -> bool:
    """Delete entire chat"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/chats/{chat_id}",
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"{get_text('error_api', lang)}: {str(e)}")
        return False


def animate_text(text: str, placeholder):
    """Animate text output word by word"""
    words = text.split()
    displayed_text = ""
    for i, word in enumerate(words):
        displayed_text += word + " "
        if i % 3 == 0:
            time.sleep(0.05)
            placeholder.markdown(displayed_text + "▌")
    placeholder.markdown(displayed_text)


# --- Configuración de Streamlit ---
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personalizado ---
st.markdown("""
<style>
    h1 { color: white !important; }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stats-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar-section {
        background-color: rgba(250, 250, 250, 0.2);
        padding: 10px;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #808080;
    }
</style>
""", unsafe_allow_html=True)

# --- Estado de la sesión ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "chat_stats" not in st.session_state:
    st.session_state.chat_stats = None
if "language" not in st.session_state:
    st.session_state.language = "es"
if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "meta-llama/Llama-3.2-3B-Instruct:together"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 800

# --- Sidebar ---
with st.sidebar:
    # Language selector
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    language_options = {"🇺🇸 English": "en", "🇪🇸 Español": "es"}
    selected_lang_display = st.selectbox(
        get_text("language", st.session_state.language),
        options=list(language_options.keys()),
        index=list(language_options.values()).index(st.session_state.language)
    )
    st.session_state.language = language_options[selected_lang_display]
    lang = st.session_state.language
    st.markdown('</div>', unsafe_allow_html=True)
    
    # API Status
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader(get_text("api_status", lang))
    api_online = check_api_health()
    if api_online:
        st.markdown(f'<p class="status-online">{get_text("api_online", lang)}</p>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="status-offline">{get_text("api_offline", lang)}</p>', 
                   unsafe_allow_html=True)
        st.warning(f"API URL: {API_BASE_URL}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document upload section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header(get_text("upload_section", lang))
    
    uploaded_files = st.file_uploader(
        get_text("upload_label", lang),
        type=["pdf"],
        accept_multiple_files=True,
        help=get_text("upload_help", lang)
    )
    
    # Create new chat button
    if uploaded_files and st.session_state.chat_id is None:
        if st.button(get_text("process_button", lang), use_container_width=True):
            with st.spinner(get_text("processing", lang)):
                result = create_chat(
                    uploaded_files,
                    "sentence-transformers/all-MiniLM-L6-v2",
                    lang
                )
                if result:
                    st.session_state.chat_id = result["chat_id"]
                    st.session_state.chat_stats = result
                    st.session_state.messages = []
                    st.success(f"{get_text('success', lang)}: Chat ID {result['chat_id']}")
                    st.rerun()
    
    # Add documents to existing chat
    if uploaded_files and st.session_state.chat_id:
        if st.button(get_text("add_docs_button", lang), use_container_width=True):
            with st.spinner(get_text("processing", lang)):
                result = add_documents_to_chat(
                    st.session_state.chat_id,
                    uploaded_files,
                    lang
                )
                if result:
                    st.success(f"{get_text('success', lang)}: {result['new_documents_added']} documentos añadidos")
                    # Refresh stats
                    stats = check_chat_status(st.session_state.chat_id)
                    if stats and stats['exists']:
                        st.session_state.chat_stats = stats
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat statistics
    if st.session_state.chat_id and st.session_state.chat_stats:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader(get_text("stats_title", lang))
        
        stats = st.session_state.chat_stats
        st.metric(get_text("chat_id", lang), st.session_state.chat_id)
        st.metric(get_text("pages", lang), stats.get("documents_count", stats.get("documents_processed", 0)))
        st.metric(get_text("chunks", lang), stats.get("chunks_count", stats.get("chunks_created", 0)))
        
        if "created_at" in stats:
            created = datetime.fromisoformat(stats["created_at"].replace("Z", "+00:00"))
            st.info(f"📅 {get_text('created_at', lang)}: {created.strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced settings
    if st.session_state.chat_id:
        with st.expander(get_text("advanced_settings", lang)):
            st.subheader(get_text("model_selection", lang))
            
            llm_options = ["meta-llama/Llama-3.2-3B-Instruct:together",
                             "Qwen/Qwen2.5-7B-Instruct:together",
                             "marin-community/marin-8b-instruct:together"]
            
            st.session_state.selected_llm = st.selectbox(
                get_text("select_model", lang),
                llm_options,
                index=llm_options.index(st.session_state.selected_llm) if st.session_state.selected_llm in llm_options else 0
            )
            
            st.session_state.temperature = st.slider(
                get_text("temperature", lang),
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1
            )
            
            st.session_state.max_tokens = st.slider(
                get_text("max_tokens", lang),
                min_value=100,
                max_value=4000,
                value=st.session_state.max_tokens,
                step=100
            )
    
    # Chat management
    if st.session_state.chat_id:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(get_text("clear_chat", lang), use_container_width=True):
                st.session_state.messages = []
                st.success(get_text("chat_cleared", lang))
                st.rerun()
        
        with col2:
            if st.button(get_text("delete_chat", lang), use_container_width=True):
                st.session_state.show_delete_confirm = True
        
        # Delete confirmation
        if st.session_state.get("show_delete_confirm", False):
            st.warning(get_text("confirm_delete", lang))
            col1, col2 = st.columns(2)
            with col1:
                if st.button(get_text("yes", lang), use_container_width=True):
                    if delete_chat(st.session_state.chat_id, lang):
                        st.session_state.chat_id = None
                        st.session_state.chat_stats = None
                        st.session_state.messages = []
                        st.session_state.show_delete_confirm = False
                        st.success(get_text("chat_deleted", lang))
                        st.rerun()
            with col2:
                if st.button(get_text("no", lang), use_container_width=True):
                    st.session_state.show_delete_confirm = False
                    st.rerun()
        
        # Download conversation
        if st.session_state.messages:
            chat_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                label=get_text("download_chat", lang),
                data=chat_text,
                file_name=f"chat_{st.session_state.chat_id}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # About section
    with st.expander(get_text("about", lang)):
        st.info(get_text("about_text", lang))
        st.markdown("**Tecnologías:**")
        st.markdown("- 🚀 FastAPI Backend")
        st.markdown("- 🤖 LangChain RAG")
        st.markdown("- 🔍 FAISS Vector Store")
        st.markdown("- 🎨 Streamlit Frontend")

# --- Header ---
st.markdown(f"""
<div class="main-header">
    <h1>{get_text("title", lang)}</h1>
    <p style="font-size:1.2rem; margin-top:0.5rem; opacity:0.9;">
        {get_text("subtitle", lang)}
    </p>
</div>
""", unsafe_allow_html=True)

# --- Main chat area ---
if not st.session_state.chat_id:
    st.info(f"👈 {get_text('warning_upload', lang)}")
else:
    st.header(get_text("chat_section", lang))
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "metadata" in message:
                    with st.expander("📊 Metadata"):
                        st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input(get_text("chat_placeholder", lang)):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner(get_text("thinking", lang)):
                result = ask_question(
                    st.session_state.chat_id,
                    prompt,
                    st.session_state.selected_llm,
                    st.session_state.temperature,
                    st.session_state.max_tokens,
                    lang
                )
                
                if result:
                    answer = result["answer"]
                    animate_text(answer, placeholder)
                    
                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "model": result["model_used"],
                            "sources": result["sources_used"],
                            "timestamp": result["timestamp"]
                        }
                    })
                else:
                    placeholder.markdown(f"{get_text('error', lang)}: No se pudo obtener respuesta")

# --- Suggestions section ---
if st.session_state.chat_id:
    st.markdown("---")
    st.subheader("💡 " + ("Sugerencias" if lang == "es" else "Suggestions"))
    
    suggestions = [
        ("📝 Resume el documento", "📝 Summarize the document"),
        ("🔑 ¿Cuáles son los puntos clave?", "🔑 What are the key points?"),
        ("📊 Extrae datos importantes", "📊 Extract important data"),
        ("❓ Explica conceptos complejos", "❓ Explain complex concepts")
    ]
    
    cols = st.columns(4)
    for idx, (suggestion_es, suggestion_en) in enumerate(suggestions):
        suggestion = suggestion_es if lang == "es" else suggestion_en
        with cols[idx]:
            if st.button(suggestion, use_container_width=True, key=f"sug_{idx}"):
                # Simulate clicking on the suggestion
                st.session_state.messages.append({"role": "user", "content": suggestion[2:]})
                st.rerun()