from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Professional, human-friendly, grounded QA prompt
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert, helpful AI assistant. Answer the user's question clearly, accurately, and naturally "
        "based directly on the context provided below.\n\n"
        "- Base your answer directly on the retrieved context.\n"
        "- Format your response with clean bullet points or structured paragraphs for high human readability.\n"
        "- If only partial information is available in the context, provide what is known and clarify what is omitted.\n"
        "- If the topic is completely absent from the context, respond: 'I don't know based on the provided documents.'\n"
        "- Do not fabricate or invent facts outside the context.\n\n"
        "Context:\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}
