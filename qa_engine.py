import pandas as pd
import numpy as np
import pandas.api.types as pat
from groq import Groq


def build_schema(df):
    lines = []
    for col in df.columns:
        if pat.is_numeric_dtype(df[col]):
            lines.append(
                f"  {col} (number): min={df[col].min()}, max={df[col].max()}, mean={round(df[col].mean(), 2)}"
            )
        else:
            examples = list(df[col].dropna().unique()[:5])
            lines.append(f"  {col} (text): examples = {examples}")
    return "\n".join(lines)


def ask_question(question: str, df: pd.DataFrame, api_key: str) -> dict:
    client = Groq(api_key=api_key)
    schema = build_schema(df)

    # ── Step 1: Generate Pandas code ──
    code_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=300,
        temperature=0.0,
        messages=[{
            "role": "system",
            "content": """You are a Pandas expert. Write simple correct Python code to answer questions about a DataFrame.

STRICT RULES:
- DataFrame is called 'df' and is already loaded
- Last line MUST be: result = ...
- Max 3 lines of code
- pandas (pd) and numpy (np) are available, no imports needed
- No markdown, no backticks, no explanation — ONLY Python code
- NEVER write: df.loc[...] = result (that assigns TO df, not to result)

GOOD examples:
  result = df[df['season'] == 2021].nlargest(1, 'runs')[['name','runs']]
  result = df.groupby('team')['runs'].mean().sort_values(ascending=False)
  result = df['runs'].mean()
  result = df.groupby('product')['total_revenue'].sum().sort_values(ascending=False).head(1)"""
        }, {
            "role": "user",
            "content": f"""DataFrame columns:
{schema}

Question: {question}

Write Pandas code (max 3 lines, last line must be: result = ...)"""
        }]
    )

    code = code_response.choices[0].message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    # ── Step 2: Run the code safely ──
    result = None
    error = None
    result_str = ""

    try:
        for dangerous in ["os", "subprocess", "open", "__import__", "eval", "exec("]:
            if dangerous in code:
                raise ValueError(f"Blocked unsafe code: '{dangerous}'")

        namespace = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, namespace)
        result = namespace.get("result")

        if result is None:
            error = "Code ran but 'result' was not set."
            result_str = error
        elif isinstance(result, pd.DataFrame):
            result_str = result.head(15).to_string(index=False)
        elif isinstance(result, pd.Series):
            result_str = result.head(15).to_string()
        else:
            result_str = str(result)

    except Exception as e:
        error = str(e)
        result_str = f"Error: {error}"

    # ── Step 3: Explain — ONLY using the exact result, no hallucination ──
    if error:
        explanation = f"Something went wrong: {error}. Try rephrasing your question."
    else:
        explain_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=150,
            temperature=0.0,   # 0 = no creativity, no made-up numbers
            messages=[{
                "role": "system",
                "content": """You explain Pandas results in plain English.

CRITICAL RULES:
- Use ONLY the numbers and names shown in the RESULT below
- Do NOT invent, estimate, or guess any numbers
- Do NOT add numbers that are not in the result
- If the result shows 58800, say 58800 — do not change it
- 2 sentences maximum"""
            }, {
                "role": "user",
                "content": f"""Question: "{question}"

RESULT (use only these exact numbers and names):
{result_str}

Explain this result in 2 sentences using only the values shown above."""
            }]
        )
        explanation = explain_response.choices[0].message.content.strip()

    return {
        "answer":     explanation,
        "code":       code,
        "result":     result,
        "result_str": result_str,
        "error":      error,
    }


def get_suggested_questions(df: pd.DataFrame) -> list:
    questions = []
    numeric_cols = [c for c in df.columns if pat.is_numeric_dtype(df[c])]
    text_cols    = [c for c in df.columns if not pat.is_numeric_dtype(df[c])]

    if numeric_cols:
        questions.append(f"Who has the highest {numeric_cols[0]}?")
        questions.append(f"What is the average {numeric_cols[0]}?")
    if len(numeric_cols) >= 2:
        questions.append(f"Show top 5 by {numeric_cols[0]}")
    if text_cols and numeric_cols:
        questions.append(f"Average {numeric_cols[0]} grouped by {text_cols[0]}")
    if text_cols:
        questions.append(f"How many unique {text_cols[0]}s are there?")

    return questions[:5]
