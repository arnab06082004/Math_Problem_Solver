import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate as PT

# ─────────────────────────────────────────────────────────────
# Streamlit Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Math Solver AI", page_icon="🧠", layout="centered")

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=2048,
)

# ─────────────────────────────────────────────────────────────
# Reasoning Tool
# ─────────────────────────────────────────────────────────────
reasoning_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a math and reasoning assistant.

Rules:
1. Always solve step-by-step.
2. Use numbered steps.
3. Keep explanations clear and clean.
4. End with a final answer in one line starting with: Final Answer:

Question: {text}
"""
)

reasoning_chain = reasoning_prompt | llm

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=lambda x: reasoning_chain.invoke({"text": x}).content,
    description=(
        "Use this for algebra, derivatives, calculus, "
        "equations, and word problems. "
        "This tool gives full step-by-step explanation."
    ),
)

# ─────────────────────────────────────────────────────────────
# Python Math Tool
# ─────────────────────────────────────────────────────────────
python_repl = PythonREPL()

math_tool = Tool(
    name="Math Tool",
    func=python_repl.run,
    description=(
        "Use ONLY for pure numeric calculations like 45*12 or 100/4. "
        "Input must be valid Python expression. "
        "Always use print(). Example: print(2+2)"
    ),
)

# ─────────────────────────────────────────────────────────────
# Custom ReAct Prompt
# ─────────────────────────────────────────────────────────────
react_prompt = PT.from_template("""Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format STRICTLY — never deviate:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

STRICT RULES:
- You MUST end every response with "Final Answer:" followed by the answer.
- Do NOT repeat the same action more than once.
- After getting an Observation, immediately write the Final Answer.
- Never leave a response without a Final Answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────
tools = [reasoning_tool, math_tool]

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=4,
    early_stopping_method="generate",
    return_intermediate_steps=False,
)

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🧠 Math Problem Solver AI")
st.caption("Solve math, algebra, derivatives, and word problems step-by-step.")

with st.sidebar:
    st.header("⚙️ Controls")
    clear = st.button("🗑️ Clear Chat", use_container_width=True)
    st.divider()
    st.subheader("🧰 Tools Used")
    st.write("✅ Reasoning Tool (Algebra & Calculus)")
    st.write("✅ Python Tool (Numeric Only)")
    st.divider()
    st.subheader("💡 Try asking")
    st.code("Derivative of x^3 + 2x?")
    st.code("Solve 2x² - 5x + 3 = 0")
    st.code("What is 15% of 840?")
    st.code("A train at 60mph for 2.5hrs, distance?")

if "messages" not in st.session_state or clear:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi 👋 Ask me any math question!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Type your question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        thought_container = st.container()
        st_cb = StreamlitCallbackHandler(
            thought_container,
            expand_new_thoughts=True,
            collapse_completed_thoughts=True,
        )
        try:
            result = agent_executor.invoke({"input": query}, callbacks=[st_cb])
            response = result["output"]
            if not response.strip() or "Agent stopped" in response:
                # Fallback: call reasoning chain directly
                response = reasoning_chain.invoke({"text": query}).content
        except Exception as e:
            # Fallback: call reasoning chain directly
            try:
                response = reasoning_chain.invoke({"text": query}).content
            except Exception as e2:
                response = f"⚠️ Error: {str(e2)}"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})