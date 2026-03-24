from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

def get_client():
    return Groq(api_key=GROQ_API_KEY)


def rewrite_question_with_history(question, chat_history):
    if not chat_history.strip():
        return question

    client = get_client()

    system_prompt = """
You rewrite follow-up questions into standalone questions.

Rules:
1. Use the chat history only to resolve references like "it", "he", "they", "that document", "the second one", etc.
2. Keep the meaning exactly the same.
3. Do not answer the question.
4. Return only the rewritten standalone question.
"""

    user_prompt = f"""
Chat History:
{chat_history}

Current Question:
{question}

Rewrite the current question as a standalone question.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten if rewritten else question


def generate_answer(question, standalone_question, context, chat_history):
    client = get_client()

    system_prompt = """
You are a conversational document QA assistant.

Rules:
1. Answer only from the provided document context.
2. Use chat history only to understand the conversation flow, not as a source of facts unless those facts are also supported by the document context.
3. If the answer is not present in the context, say: "I could not find that in the uploaded documents."
4. Keep the answer clear and concise.
5. At the end, mention source file names and page numbers when available.
"""

    user_prompt = f"""
Chat History:
{chat_history}

Original User Question:
{question}

Standalone Question Used For Retrieval:
{standalone_question}

Document Context:
{context}

Answer the original user question using only the document context.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content.strip()