"""RAG base do curso, compatível com OpenAI (como no curso) e Gemini.

O provedor é detectado pelo tipo do cliente passado em ``llm_client``:

    from openai import OpenAI
    assistant = RAGBase(index=index, llm_client=OpenAI())

    from google import genai
    assistant = RAGBase(index=index, llm_client=genai.Client())

Também dá para forçar com ``provider="openai"`` ou ``provider="gemini"``.
Se ``model`` não for informado, usa o padrão do provedor (DEFAULT_MODELS).
"""

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''.strip()

USER_PROMPT_TEMPLATE = '''
Question:
{question}

Context:
{context}
'''.strip()

# Nome usado no curso, mantido por compatibilidade
PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE

DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.5-flash-lite",
}


def detect_provider(llm_client):
    module = type(llm_client).__module__

    if module.startswith("openai"):
        return "openai"
    if module.startswith("google.genai") or module.startswith("google.generativeai"):
        return "gemini"

    # fallback por duck typing (ex.: wrappers/mocks)
    if hasattr(llm_client, "responses"):
        return "openai"
    if hasattr(getattr(llm_client, "models", None), "generate_content"):
        return "gemini"

    raise ValueError(
        f"Não foi possível detectar o provedor do cliente {type(llm_client)!r}. "
        "Passe provider='openai' ou provider='gemini'."
    )


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model=None,
        provider=None,
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template

        self.provider = provider or detect_provider(llm_client)
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Provedor inválido: {self.provider!r}. "
                f"Use um de {list(DEFAULT_MODELS)}."
            )
        self.model = model or DEFAULT_MODELS[self.provider]

    def search(self, query, num_results=5):
        boost_dict = {'question': 3.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        if self.provider == "openai":
            return self._llm_openai(prompt)
        return self._llm_gemini(prompt)

    def _llm_openai(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

        return response.output_text

    def _llm_gemini(self, prompt):
        from google.genai import types

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
            ),
        )

        return response.text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer


class RAGVector(RAGBase):
    """RAG com busca vetorial (módulo 2): só o passo de search muda.

    ``embedder`` é qualquer objeto com ``.encode(texto)`` (ex.:
    SentenceTransformer) e ``index`` é um índice vetorial como o
    ``minsearch.VectorSearch``.
    """

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict,
        )
