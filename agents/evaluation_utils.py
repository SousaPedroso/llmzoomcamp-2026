"""Utilitários de avaliação (módulo 4), compatíveis com OpenAI (como no curso) e Gemini.

O provedor é detectado pelo tipo do
cliente passado e pode ser forçado com ``provider="openai"`` ou
``provider="gemini"``. Se ``model`` não for informado, usa o padrão do
provedor (``DEFAULT_MODELS``).

    from google import genai
    from evaluation_utils import llm_structured

    result, usage = llm_structured(genai.Client(), instructions, prompt, Questions)
    calc_price(usage)

O uso de tokens das duas APIs é normalizado em ``Usage`` (``input_tokens`` /
``output_tokens``), então ``calc_price`` e ``calc_total_price`` funcionam
igual para os dois provedores.
"""

import inspect
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints

from tqdm.auto import tqdm

from rag_helper import DEFAULT_MODELS, RAGBase, detect_provider


# ---------------------------------------------------------------------------
# Preços e uso de tokens
# ---------------------------------------------------------------------------

# USD por 1M de tokens (tier pago, sem cache/batch). No Gemini, os tokens de
# raciocínio (thinking) são cobrados como saída.
PRICES = {
    # OpenAI
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    # Gemini
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
}


@dataclass
class Usage:
    """Uso de tokens normalizado entre OpenAI e Gemini."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_openai(cls, usage, model=None):
        return cls(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            model=model,
        )

    @classmethod
    def from_gemini(cls, usage_metadata, model=None):
        if usage_metadata is None:
            return cls(model=model)

        # prompt + resultados de ferramentas internas contam como entrada;
        # resposta + thinking contam como saída
        input_tokens = (
            (usage_metadata.prompt_token_count or 0)
            + (usage_metadata.tool_use_prompt_token_count or 0)
        )
        output_tokens = (
            (usage_metadata.candidates_token_count or 0)
            + (usage_metadata.thoughts_token_count or 0)
        )
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )


def to_usage(usage, model=None):
    """Converte o objeto de uso de qualquer provedor em ``Usage``."""
    if isinstance(usage, Usage):
        return usage
    if hasattr(usage, "prompt_token_count"):  # Gemini: response.usage_metadata
        return Usage.from_gemini(usage, model or DEFAULT_MODELS["gemini"])
    if hasattr(usage, "input_tokens"):  # OpenAI: response.usage
        return Usage.from_openai(usage, model or DEFAULT_MODELS["openai"])
    raise TypeError(f"Tipo de uso não reconhecido: {type(usage)!r}")


def calc_price(usage, model=None):
    usage = to_usage(usage, model)
    model = model or usage.model

    if model not in PRICES:
        raise KeyError(
            f"Sem preço cadastrado para {model!r}. "
            "Adicione em evaluation_utils.PRICES."
        )

    prices = PRICES[model]
    input_cost = (usage.input_tokens / 1_000_000) * prices["input"]
    output_cost = (usage.output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


# ---------------------------------------------------------------------------
# Provedor / retry
# ---------------------------------------------------------------------------

def resolve_provider(client, provider=None, model=None):
    provider = provider or detect_provider(client)
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Provedor inválido: {provider!r}. Use um de {list(DEFAULT_MODELS)}."
        )
    return provider, model or DEFAULT_MODELS[provider]


def with_retry(f, max_retries=5, base_delay=2.0, max_delay=60.0):
    """Executa ``f()`` com backoff exponencial.

    Útil principalmente no free tier do Gemini, que devolve 429/503 com
    frequência quando há várias chamadas em paralelo.
    """
    for attempt in range(max_retries):
        try:
            return f()
        except Exception:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * 2 ** attempt, max_delay)
            time.sleep(delay + random.uniform(0, 1))


# ---------------------------------------------------------------------------
# Saída estruturada (geração de ground truth, LLM como juiz)
# ---------------------------------------------------------------------------

def llm_structured(
    client,
    instructions,
    user_prompt,
    output_type,
    model=None,
    provider=None,
):
    """Chama o LLM pedindo uma saída no formato do modelo Pydantic ``output_type``.

    Retorna ``(objeto_parseado, Usage)``.
    """
    provider, model = resolve_provider(client, provider, model)

    if provider == "openai":
        messages = [
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_prompt},
        ]

        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=output_type,
        )

        return response.output_parsed, Usage.from_openai(response.usage, model)

    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            response_schema=output_type,
        ),
    )

    parsed = response.parsed
    if parsed is None:
        # o SDK nem sempre preenche .parsed (ex.: JSON levemente fora do schema)
        parsed = output_type.model_validate_json(response.text)

    return parsed, Usage.from_gemini(response.usage_metadata, model)


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model=None,
    provider=None,
    max_retries=5,
):
    return with_retry(
        lambda: llm_structured(
            client,
            instructions,
            user_prompt,
            output_type,
            model=model,
            provider=provider,
        ),
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# RAG com rastreio de custo
# ---------------------------------------------------------------------------

class RAGWithUsage(RAGBase):

    def __init__(self, *args, max_retries=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def _track(self, usage):
        self.last_usage = usage
        self.usages.append(usage)

    def _llm_openai(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = with_retry(
            lambda: self.llm_client.responses.create(
                model=self.model,
                input=input_messages
            ),
            max_retries=self.max_retries,
        )

        self._track(Usage.from_openai(response.usage, self.model))
        return response.output_text

    def _llm_gemini(self, prompt):
        from google.genai import types

        response = with_retry(
            lambda: self.llm_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.instructions,
                ),
            ),
            max_retries=self.max_retries,
        )

        self._track(Usage.from_gemini(response.usage_metadata, self.model))
        return response.text

    def total_cost(self):
        return calc_total_price(self.usages)


# ---------------------------------------------------------------------------
# Agente (substitui o toyaikit do curso, funciona com os dois provedores)
# ---------------------------------------------------------------------------

_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation):
    origin = get_origin(annotation) or annotation
    return _JSON_TYPES.get(origin, "string")


def function_to_schema(func):
    """Gera ``name``, ``description`` e ``parameters`` (JSON Schema) a partir
    da assinatura, dos type hints e da docstring da função."""
    hints = get_type_hints(func)
    signature = inspect.signature(func)

    properties = {}
    required = []

    for name, param in signature.parameters.items():
        annotation = hints.get(name, str)
        prop = {"type": _json_type(annotation)}

        if prop["type"] == "array":
            item_args = get_args(annotation)
            if item_args:
                prop["items"] = {"type": _json_type(item_args[0])}

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": inspect.getdoc(func) or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def extract_tool_calls(messages):
    """Extrai a trajetória (chamadas de função) do histórico do agente.

    - OpenAI: itens com ``type == "function_call"``; ``arguments`` é string JSON.
    - Gemini: ``Content`` do modelo com ``part.function_call``; ``args`` já é dict.

    Retorna sempre ``[{"name": ..., "arguments": dict}, ...]``.
    """
    tool_calls = []

    for message in messages:
        if isinstance(message, dict):
            continue

        # OpenAI (Responses API)
        if getattr(message, "type", None) == "function_call":
            tool_calls.append({
                "name": message.name,
                "arguments": json.loads(message.arguments or "{}"),
            })
            continue

        # Gemini (types.Content)
        if getattr(message, "role", None) != "model":
            continue

        for part in message.parts or []:
            if part.function_call:
                tool_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args or {}),
                })

    return tool_calls


@dataclass
class AgentResult:
    last_message: str
    all_messages: list
    tool_calls: list
    usages: list = field(default_factory=list)

    @property
    def cost(self):
        return calc_total_price(self.usages)


class AgentRunner:
    """Loop agêntico do módulo 1, empacotado para avaliação.

    ``tools`` é uma lista de funções Python; o schema de cada uma é gerado
    a partir da assinatura e da docstring (como no toyaikit).

        runner = AgentRunner(
            llm_client=genai.Client(),
            tools=[search],
            instructions="You're a course teaching assistant...",
        )
        result = runner.loop("How do I run Ollama locally?")
        result.last_message, result.tool_calls, result.cost
    """

    def __init__(
        self,
        llm_client,
        tools: list[Callable],
        instructions: str,
        model=None,
        provider=None,
        max_iterations=10,
        max_retries=5,
    ):
        self.llm_client = llm_client
        self.provider, self.model = resolve_provider(llm_client, provider, model)
        self.instructions = instructions
        self.functions = {f.__name__: f for f in tools}
        self.schemas = [function_to_schema(f) for f in tools]
        self.max_iterations = max_iterations
        self.max_retries = max_retries

    def _execute(self, name, args) -> Any:
        if name not in self.functions:
            return {"error": f"Unknown function: {name}"}
        try:
            return self.functions[name](**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def loop(self, prompt):
        if self.provider == "openai":
            return self._loop_openai(prompt)
        return self._loop_gemini(prompt)

    def _loop_openai(self, prompt):
        tools = [{"type": "function", **schema} for schema in self.schemas]
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        usages = []
        last_message = ""

        for _ in range(self.max_iterations):
            response = with_retry(
                lambda: self.llm_client.responses.create(
                    model=self.model,
                    input=messages,
                    tools=tools,
                ),
                max_retries=self.max_retries,
            )
            usages.append(Usage.from_openai(response.usage, self.model))
            messages.extend(response.output)

            has_calls = False
            for item in response.output:
                if item.type == "function_call":
                    args = json.loads(item.arguments or "{}")
                    result = self._execute(item.name, args)
                    messages.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result),
                    })
                    has_calls = True
                elif item.type == "message":
                    last_message = item.content[0].text

            if not has_calls:
                break

        return AgentResult(
            last_message=last_message,
            all_messages=messages,
            tool_calls=extract_tool_calls(messages),
            usages=usages,
        )

    def _loop_gemini(self, prompt):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=self.instructions,
            tools=[{"function_declarations": self.schemas}],
        )
        messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        usages = []
        last_message = ""

        for _ in range(self.max_iterations):
            response = with_retry(
                lambda: self.llm_client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=config,
                ),
                max_retries=self.max_retries,
            )
            usages.append(Usage.from_gemini(response.usage_metadata, self.model))

            if not response.candidates or response.candidates[0].content is None:
                break  # resposta vazia (ex.: bloqueio ou MALFORMED_FUNCTION_CALL)

            # guarda a fala do modelo inteira (preserva thought signatures)
            model_content = response.candidates[0].content
            messages.append(model_content)

            function_calls = []
            texts = []
            for part in model_content.parts or []:
                if part.function_call:
                    function_calls.append(part.function_call)
                elif part.text and not part.thought:
                    texts.append(part.text)

            if texts:
                last_message = "".join(texts)

            if not function_calls:
                break

            # todas as respostas do turno vão num único Content com role="user"
            response_parts = []
            for call in function_calls:
                result = self._execute(call.name, dict(call.args or {}))
                response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id,
                        name=call.name,
                        response={"result": result},  # precisa ser dict
                    )
                ))
            messages.append(types.Content(role="user", parts=response_parts))

        return AgentResult(
            last_message=last_message,
            all_messages=messages,
            tool_calls=extract_tool_calls(messages),
            usages=usages,
        )


# ---------------------------------------------------------------------------
# Paralelismo
# ---------------------------------------------------------------------------

def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
