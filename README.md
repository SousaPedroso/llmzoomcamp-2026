# llmzoomcamp-2026

Sessão 2026 do LLMZoomcamp da Data Talks Club.

## Agentes

Sem o agente, uma chamada de função é feita à mão e só funciona para uma chamada: você envia a pergunta, executa a função pedida, devolve o resultado e recebe a resposta. Esse fluxo falha quando o modelo precisa buscar várias vezes ou quando a primeira busca não encontra a resposta, porque não dá para saber antes quantas chamadas serão necessárias. O agente resolve isso com um loop que chama o modelo e executa as ferramentas até ele responder sem pedir nenhuma função. Quem decide quantas buscas fazer é o modelo, não o código. Ele é formado por três partes: instruções (papel e comportamento), ferramentas (aqui, só o search) e memória (o histórico com prompts, saídas do modelo e resultados das ferramentas). Com isso o modelo consegue se corrigir sozinho. Por exemplo, ao buscar "Olama" e não achar nada, ele tenta de novo com "Ollama". Frameworks como LangChain, PydanticAI e OpenAI Agents SDK são versões desse mesmo padrão.

Dado que aqui está sendo usado o gemini, em contrapartida a openai no curso, algumas adaptações foram feitas para que o código funcione com o gemini. Abaixo estão as principais mudanças tanto conceituais quanto de código.

### Equivalências

| Conceito | OpenAI (Responses API) | Gemini (google-genai) |
|---|---|---|
| Instruções de sistema | `instructions=...` ou mensagem com `role: "developer"` | `config.system_instruction` |
| Declaração de ferramenta | `{"type": "function", "name": ..., "parameters": ...}` | `{"function_declarations": [{"name": ..., "parameters": ...}]}` |
| Saída do modelo | `response.output` (lista de itens) | `response.candidates[0].content.parts` |
| Identificar chamada de função | `item.type == "function_call"` | `part.function_call` não é `None` |
| Argumentos da chamada | `json.loads(call.arguments)` (string JSON) | `call.args` (já é `dict`) |
| ID da chamada | `call.call_id` | `call.id` |
| Resposta da função | `{"type": "function_call_output", "call_id": ..., "output": json_str}` | `types.Part(function_response=types.FunctionResponse(id=..., name=..., response={...}))` |
| Formato do resultado | string (normalmente `json.dumps(...)`) | `dict` (ex.: `{"result": resultado}`) |
| Papel da resposta da função | item solto na lista de mensagens | `types.Content(role="user", parts=[...])` |
| Guardar fala do modelo no histórico | `messages.extend(response.output)` | `messages.append(response.candidates[0].content)` |
| Texto final do modelo | item `type == "message"` → `item.content[0].text` | `part.text` (ignorando `part.thought`) |
| Condição de parada | nenhum item `function_call` na saída | nenhuma `part.function_call` na saída |

### Pontos de atenção no Gemini

- **Thought signatures:** nos modelos Gemini 3, adicione o `Content` do modelo inteiro ao histórico. Não reconstrua a fala do modelo manualmente, senão as assinaturas se perdem.
- **Chamadas paralelas:** o modelo pode pedir várias chamadas de função no mesmo turno. Por isso a necessidade de percorrer todas as `parts`, não só `parts[0]`.
- **Um turno de resposta:** todas as respostas de função de um mesmo turno vão juntas em **um único** `Content` com `role="user"`.
- **`response` precisa ser dict:** se a função retorna uma lista, embrulhe em `{"result": lista}`.

### Esqueleto do loop

#### OpenAI

```python
while True:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages,
        tools=[search_tool],
    )
    messages.extend(response.output)

    has_calls = False
    for item in response.output:
        if item.type == "function_call":
            messages.append(make_call(item))  # function_call_output
            has_calls = True
        elif item.type == "message":
            print(item.content[0].text)

    if not has_calls:
        break
```

#### Gemini

```python
for it in range(MAX_ITERATIONS):
    response = genai_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            tools=[search_tool],
        ),
    )
    model_content = response.candidates[0].content
    messages.append(model_content)

    calls = []
    for part in model_content.parts or []:
        if part.function_call:
            calls.append(part.function_call)
        elif part.text and not part.thought:
            print(part.text)

    if not calls:
        break

    messages.append(types.Content(
        role="user",
        parts=[make_call(call) for call in calls],  # Part(function_response=...)
    ))
```


