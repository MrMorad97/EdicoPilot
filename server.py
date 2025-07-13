import streamlit as st
import tempfile
import os
from prompt import PROMPT_REGISTRY
from rag import StreamingChatPDF
from gpt2_inference import StreamingGPT2Chat
from LM import LightweightTransformerChatbot, DailyDialogTokenizer, ConversationContext, generate_response
import pickle
import torch
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import PromptTemplate
import time
from fineTunedGpt2 import FinetunedGPT2Chat

# Configure Streamlit
st.set_page_config(
    page_title="EduCopilot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_model" not in st.session_state:
    st.session_state.rag_model = StreamingChatPDF()
if "gpt2_model" not in st.session_state:
    st.session_state.gpt2_model = StreamingGPT2Chat()
if "lm_model" not in st.session_state:
    try:
        with open("dailydialog_tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        model = LightweightTransformerChatbot(
            vocab_size=tokenizer.vocab_size(),
            d_model=192,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            max_len=128,
            dropout=0.1,
            pad_token_id=tokenizer.pad_token_id
        )
        model.load_state_dict(torch.load("best_dailydialog_chatbot.pth", map_location='cpu'))
        model.eval()
        st.session_state.lm_model = model
        st.session_state.lm_tokenizer = tokenizer
        st.session_state.lm_context = ConversationContext(tokenizer)
    except Exception as e:
        st.error(f"Failed to load custom LM: {str(e)}")
        st.session_state.lm_model = None

# PDF state tracking
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "last_pdf_hash" not in st.session_state:
    st.session_state.last_pdf_hash = None
if "ft_gpt2_model" not in st.session_state:
    st.session_state.ft_gpt2_model = FinetunedGPT2Chat("./fine_tuned_gpt2")

# UI Setup
st.title("🎓 EduCopilot - Multi-Model Chat")

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    chat_mode = st.selectbox(
        "Chat Mode",
        ["RAG + Ollama", "GPT2", "Fine-tuned GPT2" ,"Custom LM"],
        index=0,
        key="chat_mode"
    )
    
    # Status indicator
    if st.session_state.chat_mode == "RAG + Ollama":
        if st.session_state.pdf_loaded:
            st.success("📄 PDFs loaded and ready")
        else:
            st.info("💡 Upload PDFs for document-based chat")
    
    # Prompt selection for RAG mode
    if st.session_state.chat_mode == "RAG + Ollama":
        prompt_names = ["Default"] + list(PROMPT_REGISTRY.keys())
        selected_prompt = st.selectbox(
            "Select Prompt Style", 
            prompt_names,
            index=0,
            key="prompt_selector"
        )
        
        if selected_prompt != "Default":
            st.markdown("**Active Prompt Template:**")
            st.code(PROMPT_REGISTRY[selected_prompt](), language="text")
        
        # PDF upload
        st.markdown("**📁 PDF Upload**")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents for context-aware responses"
        )
        
        # Show file info
        if uploaded_files:
            file_info = f"📊 {len(uploaded_files)} file(s) selected"
            total_size = sum(f.size for f in uploaded_files) / (1024*1024)  # MB
            st.caption(f"{file_info} ({total_size:.1f} MB)")
    
    # Model information
    with st.expander("🔧 Model Information"):
        if st.session_state.chat_mode == "RAG + Ollama":
            st.markdown("**TinyLlama Configuration:**")
            st.caption("• Using default Ollama settings")
            st.caption("• Standard chunk size: 1000 characters")
            st.caption("• Retrieval: 4 relevant documents")
            st.caption("• Context limit: 2000 characters")
            
            # Cache info
            if hasattr(st.session_state.rag_model, 'response_cache'):
                cache_size = len(st.session_state.rag_model.response_cache)
                st.caption(f"• Cached responses: {cache_size}")
        
        elif st.session_state.chat_mode == "GPT2":
            st.caption("• Using Hugging Face GPT-2")
            st.caption("• Optimized for conversational responses")
        
        elif st.session_state.chat_mode == "Custom LM":
            st.caption("• Custom trained transformer model")
            st.caption("• Trained on DailyDialog dataset")
            
        elif st.session_state.chat_mode == "Fine-tuned GPT2":
            st.caption("• Using your fine-tuned GPT-2 model")
            st.caption("• Based on Hugging Face GPT-2 base, fine-tuned on your dataset")

    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.rag_model.clear()
        st.session_state.pdf_loaded = False
        st.session_state.last_pdf_hash = None
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input and processing
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        style_enforced = False  # Track if we've enforced response style
        
        try:
            if st.session_state.chat_mode == "RAG + Ollama":
                # Handle PDF upload
                if uploaded_files:
                    current_hash = str(sorted([f.name + str(f.size) for f in uploaded_files]))
                    
                    if current_hash != st.session_state.last_pdf_hash:
                        with st.spinner("🔄 Processing PDFs..."):
                            temp_dir = tempfile.mkdtemp()
                            paths = []
                            for file in uploaded_files:
                                path = os.path.join(temp_dir, file.name)
                                with open(path, "wb") as f:
                                    f.write(file.getvalue())
                                paths.append(path)
                            
                            success = st.session_state.rag_model.ingest_pdfs(paths)
                            if success:
                                st.session_state.pdf_loaded = True
                                st.session_state.last_pdf_hash = current_hash
                                st.success("✅ PDFs processed successfully!")
                            else:
                                st.error("❌ Failed to process PDFs")
                
                # Handle custom prompts
                if st.session_state.prompt_selector != "Default":
                    template = PROMPT_REGISTRY[st.session_state.prompt_selector]()
                    prompt_template = PromptTemplate.from_template(template)
                    modified_prompt = prompt_template.format(question=prompt, context="")
                    st.info(f"Using {st.session_state.prompt_selector} prompt style")
                else:
                    modified_prompt = prompt
                    prompt_template = None
                
                # Stream response
                for chunk in st.session_state.rag_model.stream_response(
                    question=modified_prompt,
                    prompt_template=template if st.session_state.prompt_selector != "Default" else None
                ):
                    if isinstance(chunk, AIMessageChunk):
                        chunk = chunk.content
                    
                    full_response += str(chunk)
                    
                    # Enforce concise response style if needed
                    if (st.session_state.prompt_selector == "consice" and 
                        any(punct in full_response for punct in ['.','!','?']) and
                        not style_enforced):
                        full_response = '.'.join(full_response.split('.')[:1]) + '.'
                        response_placeholder.markdown(full_response)
                        style_enforced = True
                        break
                        
                    response_placeholder.markdown(full_response + "▌")
                
                # Enforce detailed response style if needed
                if (st.session_state.prompt_selector == "detailed" and 
                    len(full_response.split()) < 50 and
                    not style_enforced):
                    full_response += "\n\n[Additional details requested]"
            
            elif st.session_state.chat_mode == "GPT2":
                for chunk in st.session_state.gpt2_model.stream_response(prompt):
                    full_response += str(chunk)
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            elif st.session_state.chat_mode == "Custom LM":
                if st.session_state.lm_model is None:
                    full_response = "Custom LM model is not available. Please check the model files."
                else:
                    with st.spinner("🤖 Generating response..."):
                        response = generate_response(
                            st.session_state.lm_model,
                            st.session_state.lm_tokenizer,
                            st.session_state.lm_context,
                            prompt
                        )
                        full_response = str(response)
                response_placeholder.markdown(full_response)

            elif st.session_state.chat_mode == "Fine-tuned GPT2":
                for chunk in st.session_state.ft_gpt2_model.stream_response(prompt):
                    full_response += str(chunk)
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    "💡 **EduCopilot** uses TinyLlama with default settings for optimal balance of speed and quality. "
    "Upload PDFs for document-based Q&A or chat directly with the AI models."
)