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




## Busca Vetorial

Realizar a busca por um texto, mesmo que buscando a palavra num documento, isso não considera a semântica do texto, apenas a presença da palavra. A busca vetorial, por outro lado, considera o significado do texto e retorna resultados mais relevantes. Para isso, é necessário transformar o texto em vetores numéricos (embeddings) que representam o significado do texto.

Os embeddings são gerados por modelos de linguagem treinados para capturar o significado semântico do texto. Cada documento é transformado em um vetor de alta dimensão, e esses vetores são armazenados em um índice vetorial. Quando uma pergunta é feita, ela também é transformada em um vetor, e a busca é realizada comparando esse vetor com os vetores dos documentos no índice.

A busca semântica é feita em duas etapas: primeiro, o texto da pergunta é transformado em um vetor (embedding) e, em seguida, esse vetor é comparado com os vetores dos documentos para encontrar os mais próximos. A proximidade entre vetores é medida por uma métrica de similaridade, como a distância euclidiana ou a similaridade do cosseno.

A busca vetorial, inicialmente se deu sem persistência dos dados, via [min search](./busca_vetorial/vector_search.ipynb).  Após, seguimos com o uso de persistência, em sqlite, conforme o [sqlite_vector_search](./busca_vetorial/sqlite_vector_search.ipynb). Por fim, com uma persistência mais robusta, via [pgvector](./busca_vetorial/pg_vector_sarch.ipynb).


## Avaliação

Avaliar a resposta do modelo é uma tarefa complexa, pois envolve aspectos subjetivos e objetivos. A avaliação pode ser feita de várias maneiras, incluindo métricas automáticas, revisões humanas e testes de desempenho. De modo geral, pode ser offlinete (numa base de dados de perguntas e respostas) ou online (avaliando a resposta do modelo em tempo real, geralmente por feedback de usuários).

A avaliação pode medir a função de busca (retornar os documentos corretos), bem como o uso de ferramentas do agente e o RAG (se o agente gera boas respostas).

### Busca

Para avaliar a busca/pesquisa, precisamos determinar um dataset de referência. Este dataset precisa de perguntas em que sabemos quais os documentos são a resposta correta. Geralmente, esse conjunto é denominado como 'gold standard' ou 'ground truth'.

Algumas formas de gerar o conjunto de referência incluem:
- Anotação manual olhando os documentos e escrevendo as perguntas.
- Coletar as perguntas dos usuários e rotular as respostas corretas.
- Geração sintética de perguntas e respotas a partir de um LLM (usada aqui no curso).

#### Métricas

Para avaliar a busca, as principais métricas são a HIT Rate (taxa de acerto) e a MRR (Mean Reciprocal Rank). A HIT Rate mede a proporção de perguntas em que pelo menos um dos documentos retornados é relevante (ideia semelhante a um recall). A MRR mede a posição do primeiro documento relevante na lista de resultados, sendo mais sensível à ordem dos resultados, semelhando-se a uma precisão ponderada.

### RAG/Agentes

A avaliação de agentes exige mais complexidade, por availiar o pipeline como um todo. Isto é, não apenas a busca, mas também a capacidade do agente de usar ferramentas, gerar respostas e interagir com o usuário. Dentre as formas de avaliação, uma das principais é a partir de ground truth questions.

A avaliação por **ground truth** envolve a criação de um conjunto de perguntas e respostas esperadas, que são usadas para testar o agente. O agente é executado com essas perguntas, e suas respostas são comparadas com as respostas esperadas, se estão próximas. Assim, basicamente o fluxo se dá por A -> B -> A', onde A é a resposta esperada, B é a questão gerado por essa resposta e A' é a resposta do agente. Se A' for próxima de A, o agente está funcionando bem.

Outra forma de avaliação também possível se dá por RAG, através de utilizar um LLM como juiz. Forma também offline, tal como com **ground truth**, mas aqui a ideia é verificar se as respostas são boas o suficiente. A ideia se assemelha ao feito por **ground truth**, mas aqui o juiz é um LLM, que avalia se a resposta do agente é boa o suficiente, sem necessariamente ter uma resposta esperada. A avaliação por LLM juiz pode ser feita de forma automática, mas também é recomendado envolver revisão humana para garantir a qualidade das avaliações.

Para a avaliação via RAG, também aproveitamos da mesma configuração das outras avaliações, com o fluxo A -> B -> A', mas aqui nesse caso o A' vem de um agente, ao invés de um pipeline RAG. Assim, aqui também usamos um juiz LLM para avaliar se a resposta do agente é boa o suficiente, sem necessariamente ter uma resposta esperada. A avaliação por LLM juiz pode ser feita de forma automática, mas também é recomendado envolver revisão humana para garantir a qualidade das avaliações. Também nesse caso, pode ser considerado o uso de trajetória, em que basicamente é olhar se o agente está fazendo o que deveria fazer, se está usando as ferramentas corretas, se está buscando corretamente, etc. A ideia é que o juiz LLM avalie a trajetória do agente, e não apenas a resposta final.
