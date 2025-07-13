# EducoPilot: Multi-Model Conversational AI System

A comprehensive conversational chatbot system that integrates multiple deep learning models including custom transformers, RAG systems, and fine-tuned GPT-2 models.

## 🚀 Features

- **Custom Transformer Model**: Lightweight model trained from scratch on DailyDialog corpus
- **RAG System**: Retrieval-Augmented Generation with TinyLlama and ChromaDB
- **Fine-tuned GPT-2**: Adapted pre-trained model for conversational tasks
- **Streamlit Interface**: Modern web interface for all models
- **PDF Document Support**: Upload and query PDF documents
- **Real-time Streaming**: Live response generation
- **Multi-model Switching**: Seamless switching between different AI models

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- Ollama (for local model serving)
- At least 8GB RAM (16GB recommended)
- GPU support (optional, for faster inference)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MrMorad97/EdicoPilot.git
cd EducoPilot
```

### 2. Create Virtual Environment

#### Windows:
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# Verify activation (should show venv path)
where python
```

#### Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

### 3. Install Dependencies

#### Windows:
```cmd
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements

# If you encounter SSL errors, try:
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements
```

#### Linux:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3-dev build-essential libssl-dev libffi-dev python3-setuptools

# Install all required packages
pip install -r requirements

# If you encounter permission errors, use:
pip install --user -r requirements
```

### 4. Install Ollama

#### Windows:
```cmd
# Method 1: Using winget (recommended)
winget install Ollama.Ollama

# Method 2: Manual download
# Visit https://ollama.ai/download
# Download and run the Windows installer

# Method 3: Using Chocolatey
choco install ollama

# Verify installation
ollama --version
```

#### Linux (Ubuntu/Debian):
```bash
# Method 1: Using curl (recommended)
curl -fsSL https://ollama.ai/install.sh | sh

# Method 2: Using apt (if available)
sudo apt update
sudo apt install ollama

# Method 3: Manual installation
wget https://ollama.ai/download/ollama-linux-amd64
sudo mv ollama-linux-amd64 /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama

# Verify installation
ollama --version
```

#### Linux (CentOS/RHEL/Fedora):
```bash
# Method 1: Using curl
curl -fsSL https://ollama.ai/install.sh | sh

# Method 2: Using dnf (Fedora)
sudo dnf install ollama

# Method 3: Using yum (CentOS/RHEL)
sudo yum install ollama

# Verify installation
ollama --version
```

### 5. Download Required Models

#### Windows:
```cmd
# Start Ollama service (run as Administrator)
ollama serve

# In a new terminal, pull models
ollama pull tinyllama
ollama pull mistral

# Verify models
ollama list
```

#### Linux:
```bash
# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Or run manually
ollama serve &

# Pull models
ollama pull tinyllama
ollama pull mistral

# Verify models
ollama list
```

## 🚀 Usage

### 1. Start Ollama Server

#### Windows:
```cmd
# Method 1: Run as Administrator (recommended)
ollama serve

# Method 2: Run as service
# Ollama should start automatically after installation
# Check if it's running:
ollama list

# Method 3: Run in background
start /B ollama serve
```

#### Linux:
```bash
# Method 1: Using systemctl (recommended)
sudo systemctl start ollama
sudo systemctl enable ollama
sudo systemctl status ollama

# Method 2: Run manually
ollama serve &

# Method 3: Run in background with nohup
nohup ollama serve > ollama.log 2>&1 &

# Verify it's running
ollama list
```

### 2. Launch the Application

#### Windows:
```cmd
# Activate virtual environment
venv\Scripts\activate.bat

# Launch the Streamlit app
streamlit run server.py

# Or specify port if 8501 is busy
streamlit run server.py --server.port 8502
```

#### Linux:
```bash
# Activate virtual environment
source venv/bin/activate

# Launch the Streamlit app
streamlit run server.py

# Or specify port if 8501 is busy
streamlit run server.py --server.port 8502

# Or run in background
nohup streamlit run server.py > streamlit.log 2>&1 &
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
EducoPilot/
├── best_dailydialog_chatbot.pth    # Custom transformer model weights
├── dailydialog_tokenizer.pkl       # Custom transformer tokenizer
├── dialogues_test.txt              # Test dialogue dataset
├── rag.py                          # RAG system implementation
├── fineTunedGpt2.py                # GPT-2 fine-tuning script
├── server.py                       # Streamlit web interface
├── scores.py                       # BERTScore evaluation
├── training_log.txt                # Training logs
├── requirements                    # Project dependencies
├── generate.py                     # Text generation script
├── visualize_bert.py               # BERT visualization utilities
├── prompt.py                       # Prompt engineering utilities
├── test.py                         # Testing script
├── commands                        # Command reference
├── gpt2_inference.py               # GPT-2 inference script
├── LM.py                           # Language model utilities
├── fine_tuned_gpt2/                # Fine-tuned GPT-2 model directory
├── embeddings_cache/               # Cached embeddings directory
├── chroma_db/                      # Chroma vector database
├── venv/                           # Virtual environment
└── README.md                       # This file
```

## 🎯 Model Modes

### 1. Custom LM Mode
- Uses the custom transformer model trained from scratch
- Fast inference, specialized for conversations
- Limited vocabulary but efficient

### 2. GPT-2 Base Mode
- Uses pre-trained GPT-2 model
- General-purpose responses
- Good baseline performance

### 3. Fine-tuned GPT-2 Mode
- Uses GPT-2 fine-tuned on DailyDialog corpus
- Best conversational performance
- Balanced between speed and quality

### 4. TinyLlama Mode (RAG)
- **Without PDF**: Direct chat with TinyLlama
- **With PDF**: Document-based responses using RAG system
- Factual and accurate responses based on uploaded documents

## 📊 Performance Metrics

| Model | BERTScore F1 | Response Time | Memory Usage |
|-------|-------------|---------------|--------------|
| Custom LM | 0.84 | ~150ms | 45MB |
| GPT-2 Base | 0.82 | ~250ms | 548MB |
| Fine-tuned GPT-2 | 0.86 | ~300ms | 548MB |
| RAG System | N/A | ~800ms | 1.2GB |

## 🔧 Configuration

### RAG System Settings
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 150 characters
- **Top-k Retrieval**: 4 documents
- **Score Threshold**: 0.4
- **Context Limit**: 2500 characters

### Model Parameters
- **Temperature**: 0.7 (adjustable in UI)
- **Max Tokens**: 150 (adjustable in UI)
- **Cache Size**: 50 responses

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**

   #### Windows:
   ```cmd
   # Check if Ollama is running
   ollama list
   
   # Restart Ollama (run as Administrator)
   ollama serve
   
   # Check Windows services
   services.msc
   # Look for "Ollama" service and start it
   ```

   #### Linux:
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Restart Ollama service
   sudo systemctl restart ollama
   sudo systemctl status ollama
   
   # Check logs
   sudo journalctl -u ollama -f
   ```

2. **CUDA/GPU Issues**

   #### Windows:
   ```cmd
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Force CPU usage if needed
   set CUDA_VISIBLE_DEVICES=""
   ```

   #### Linux:
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Force CPU usage if needed
   export CUDA_VISIBLE_DEVICES=""
   
   # Check NVIDIA drivers
   nvidia-smi
   ```

3. **Memory Issues**

   #### Windows:
   ```cmd
   # Clear cache
   rmdir /s /q embeddings_cache
   rmdir /s /q chroma_db
   
   # Check available memory
   wmic computersystem get TotalPhysicalMemory
   ```

   #### Linux:
   ```bash
   # Clear cache
   rm -rf embeddings_cache/*
   rm -rf chroma_db/*
   
   # Check available memory
   free -h
   
   # Clear system cache if needed
   sudo sync && sudo sysctl -w vm.drop_caches=3
   ```

4. **Port Already in Use**

   #### Windows:
   ```cmd
   # Find process using port 8501
   netstat -ano | findstr :8501
   
   # Kill process (replace PID with actual process ID)
   taskkill /PID <PID> /F
   
   # Or use different port
   streamlit run server.py --server.port 8502
   ```

   #### Linux:
   ```bash
   # Find process using port 8501
   lsof -ti:8501
   
   # Kill process
   lsof -ti:8501 | xargs kill -9
   
   # Or use different port
   streamlit run server.py --server.port 8502
   ```

5. **Permission Issues**

   #### Windows:
   ```cmd
   # Run as Administrator
   # Right-click Command Prompt/PowerShell and select "Run as Administrator"
   
   # Check file permissions
   icacls venv
   ```

   #### Linux:
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER venv/
   chmod +x venv/bin/activate
   
   # Install with user flag
   pip install --user -r requirements
   ```

6. **Python/Pip Issues**

   #### Windows:
   ```cmd
   # Reinstall pip
   python -m pip install --upgrade pip
   
   # Clear pip cache
   pip cache purge
   
   # Install with trusted hosts
   pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements
   ```

   #### Linux:
   ```bash
   # Reinstall pip
   pip install --upgrade pip
   
   # Clear pip cache
   pip cache purge
   
   # Install system dependencies
   sudo apt update && sudo apt install -y python3-dev build-essential
   ```

## 📝 Dependencies

Key dependencies include:
- **PyTorch**: Deep learning framework
- **Transformers**: Pre-trained models and tokenizers
- **LangChain**: RAG orchestration
- **ChromaDB**: Vector database
- **Streamlit**: Web interface
- **Ollama**: Local model server
- **FastEmbed**: High-speed embeddings
- **BERTScore**: Evaluation metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<h2>👥 Lead Developers</h2>

<div align="center" style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">

  <div style="width: 200px; padding: 15px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); text-align: center;">
    <img src="https://github.com/MrMorad97.png" width="100" height="100" style="border-radius: 50%; border: 2px solid #00c6ff;" alt="MrMorad97"/>
    <h3>Morad Boussagman</h3>
    <p><em>Lead Developer & AI Specialist</em></p>
    <a href="https://github.com/MrMorad97" target="_blank">@MrMorad97</a>
  </div>

  <div style="width: 200px; padding: 15px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); text-align: center;">
    <img src="https://github.com/sung55jinwoo.png" width="100" height="100" style="border-radius: 50%; border: 2px solid #00c6ff;" alt="sung55jinwoo"/>
    <h3>Ayoub Hammad</h3>
    <p><em>Lead Developer &  AI Specialist</em></p>
    <a href="https://github.com/sung55jinwoo" target="_blank">@sung55jinwoo</a>
  </div>

</div>


## 🙏 Acknowledgments

- DailyDialog dataset for training data
- Hugging Face for pre-trained models
- Streamlit for the web interface
- Ollama for local model serving

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section above

---

**Note**: Make sure to have sufficient disk space for model downloads and embeddings cache. The system requires approximately 2GB of free space for optimal operation. 
