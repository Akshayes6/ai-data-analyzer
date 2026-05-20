# 📊 AI Data Analyzer

An AI-powered CSV analysis tool built with Streamlit. Upload any CSV file and instantly get statistics, interactive charts, natural language Q&A, and AI-generated insights.

---

## ✨ Features

- **📋 Summary** — Auto stats, missing values, column overview, and rule-based insights
- **📊 Charts** — Interactive Plotly visualizations (histogram, bar, scatter, box, line, heatmap)
- **💬 Ask a Question** — Ask anything in plain English; the AI writes Pandas code and runs it on your actual data for 100% accurate answers
- **🤖 AI Insights** — LLaMA-powered insights that highlight what's interesting in your dataset

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/ai-data-analyzer.git
cd ai-data-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate        # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```
> Get a free API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
ai-data-analyzer/
├── app.py              # Main Streamlit app
├── analyzer.py         # Pandas-based data analysis
├── llm_insights.py     # AI insights via Groq API
├── qa_engine.py        # Natural language Q&A engine
├── requirements.txt    # Python dependencies
├── .env                # Your API key (never committed)
└── .gitignore
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web UI |
| Pandas / NumPy | Data analysis |
| Plotly | Interactive charts |
| Groq API | LLM inference |
| LLaMA 3.1 8B | AI model |

---

## ⚠️ Note

Never share or commit your `.env` file. It is already excluded via `.gitignore`.
