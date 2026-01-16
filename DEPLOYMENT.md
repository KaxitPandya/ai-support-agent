# Deployment Guide

## Deploy to Streamlit Cloud

This application is ready for one-click deployment to Streamlit Cloud.

### Prerequisites
- GitHub account
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Deployment Steps

#### 1. Fork/Clone this Repository
```bash
git clone https://github.com/YOUR-USERNAME/knowledge-assistant.git
cd knowledge-assistant
```

#### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - **Repository**: `YOUR-USERNAME/knowledge-assistant`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click "Deploy"

#### 3. Configure Secrets

After deployment, add your OpenAI API key:

1. In Streamlit Cloud dashboard, click on your app
2. Go to **Settings** > **Secrets**
3. Add the following (copy from `.streamlit/secrets.toml.example`):

```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = "0.3"
OPENAI_MAX_TOKENS = "1024"

APP_NAME = "Knowledge Assistant"
APP_VERSION = "1.0.0"
DEBUG = "False"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = "384"

TOP_K_RESULTS = "5"
SIMILARITY_THRESHOLD = "0.3"
```

4. Click "Save"
5. Your app will automatically restart with the new secrets

#### 4. Access Your App

Your app will be available at:
```
https://YOUR-APP-NAME.streamlit.app
```

### Features Available

- 🎯 **Ticket Resolution**: Resolve customer support tickets using RAG
- 📚 **Document Upload**: Upload custom knowledge base documents
- 🔍 **RAG Inspector**: Test and debug the retrieval system
- 📊 **Analytics**: View ticket resolution statistics
- 🧪 **Testing**: Test with sample tickets

### API Endpoint

The FastAPI backend is also included but requires separate deployment if needed. For Streamlit Cloud, only the Streamlit UI will be deployed.

To run the FastAPI backend locally:
```bash
uvicorn src.main:app --reload
```

### Troubleshooting

**App won't start?**
- Check that your OpenAI API key is correctly set in Secrets
- View logs in Streamlit Cloud dashboard

**Out of memory?**
- The free tier has limited resources. Consider:
  - Using a smaller embedding model
  - Reducing `top_k_results` in secrets

**Slow first load?**
- First request downloads the embedding model (~80MB)
- Subsequent loads use cached model

### Local Development

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.env` file (copy from `env.example`):
   ```bash
   cp env.example .env
   ```
5. Add your OpenAI API key to `.env`
6. Run Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

### Environment Variables

See `env.example` for all available configuration options.

### Support

For issues, please create a GitHub issue in this repository.
