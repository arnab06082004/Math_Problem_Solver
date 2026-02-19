# 🧠 Math Problem Solver AI

An AI-powered math assistant built with **Streamlit**, **LangChain 0.3**, and **Groq's LLaMA 3.3 70B**. It solves math problems, algebra, calculus, and word problems with clear step-by-step explanations.

---

## ✨ Features

- 📐 **Step-by-step solutions** for algebra, calculus, geometry, and word problems
- 🔢 **Python REPL** for accurate numeric calculations
- 🔁 **Fallback mechanism** — if the agent hits a limit, the reasoning chain answers directly
- 💬 **Persistent chat history** across the session
- 🧠 **ReAct agent** with a custom prompt that enforces structured Final Answers

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Agent Framework | LangChain 0.3 |
| Math Engine | PythonREPL (`langchain-experimental`) |
| Config | python-dotenv |

---

## 📁 Project Structure

```
math-solver-ai/
├── app.py              # Main Streamlit application
├── .env                # Environment variables (not committed)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/arnab06082004/Math_Problem_Solver
cd Math_Problem_Solver

```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run app.py
```

---

## 📦 requirements.txt

```
langchain==0.3.25
langchain-core==0.3.83
langchain-community==0.3.23
langchain-groq==0.3.8
langchain-experimental==0.3.4
langchain-text-splitters
langsmith
streamlit
python-dotenv
wikipedia
```

---

## 🧰 How It Works

The app uses a **LangChain ReAct agent** (`create_react_agent` + `AgentExecutor`) with two tools:

- **Reasoning Tool** — Uses an LCEL chain (`prompt | llm`) with a custom prompt to solve algebra, calculus, and word problems step by step, ending with a clear `Final Answer:`.
- **Math Tool** — Uses `PythonREPL` for pure numeric calculations like `print(45 * 12)`.

A **custom ReAct prompt** strictly enforces the `Thought → Action → Observation → Final Answer` format to prevent the agent from looping indefinitely.

If the agent fails or hits the iteration limit, a **direct fallback** to the reasoning chain ensures the user always receives an answer.

---

## 💡 Example Questions to Try

```
What is the derivative of x^3 + 2x?
Solve: 2x² - 5x + 3 = 0
What is 15% of 840?
A train travels 60mph for 2.5 hours. What is the distance?
Integrate x^2 + 3x with respect to x
```

---

## 📸 UI Overview

- **Chat window** — Conversational interface with full message history
- **Sidebar** — Clear chat button, active tools list, and example questions
- **Thought expander** — Agent's reasoning steps shown in real time via `StreamlitCallbackHandler`, collapsing once complete

---

## 🔒 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for accessing LLaMA models |

---

## ⚠️ Known Limitations

- The ReAct agent may occasionally struggle with very complex multi-step problems; the fallback chain handles these gracefully.
- `PythonREPL` executes arbitrary Python — do not deploy publicly without sandboxing.
- Groq API has rate limits on the free tier; heavy usage may result in temporary throttling.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).