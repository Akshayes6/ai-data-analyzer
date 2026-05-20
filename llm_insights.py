from groq import Groq


def get_ai_insights(dataset_summary: str, api_key: str) -> str:
    """Sends a compact dataset summary to LLaMA and returns 5-7 specific insights."""
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=600,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": """You are a data analyst. Give exactly 5-7 insights about this dataset.
Rules:
- Start each insight with "•"
- Be SPECIFIC — mention actual column names and numbers
- 1-2 sentences per insight
- No introduction or conclusion — just the bullet points"""
            },
            {
                "role": "user",
                "content": f"Dataset:\n{dataset_summary}"
            }
        ]
    )
    return response.choices[0].message.content
