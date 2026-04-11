# Synalinks API Reference

## Core Classes

### synalinks.DataModel

Base class for defining structured I/O schemas. Inherits from Pydantic BaseModel.

```python
class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")
    context: str = synalinks.Field(default="", description="Optional context")

# Methods
model.get_schema()  # Returns JSON schema dict
model.prettify_json()  # Returns formatted JSON string
model.prettify_schema()  # Returns formatted schema string
model.get("field_name")  # Get field value
```

### synalinks.Field

Field descriptor with metadata for LLM prompts.

```python
synalinks.Field(
    description="Field description for LLM",  # Required for LLM understanding
    default=None,  # Default value
    default_factory=list,  # Factory for mutable defaults
)
```

### synalinks.JsonDataModel

Runtime data model with actual values. Created when DataModel is instantiated.

### synalinks.SymbolicDataModel

Schema-only placeholder used during graph construction. No actual data.

```python
# Metaclass operations create SymbolicDataModel
combined_schema = Query + Answer  # SymbolicDataModel
isinstance(combined_schema, synalinks.SymbolicDataModel)  # True
```

---

## Module Classes

### synalinks.Module

Base class for all modules. Encapsulates state and computation.

```python
class MyModule(synalinks.Module):
    def __init__(self, name=None, description=None, trainable=True):
        super().__init__(name=name, description=description, trainable=trainable)

    async def call(self, inputs, training=False):
        # Core computation
        return outputs

    async def compute_output_spec(self, inputs, training=False):
        # Define output schema (optional, auto-inferred if not implemented)
        return symbolic_output

    def get_config(self):
        return {"name": self.name, ...}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
```

**Properties:**
- `module.trainable` - Whether variables are updated during training
- `module.trainable_variables` - List of trainable variables
- `module.variables` - All variables
- `module.name` - Module name
- `module.description` - Module description

### synalinks.Input

Entry point for programs. Defines expected input schema.

```python
inputs = synalinks.Input(data_model=Query)
```

### synalinks.Generator

Core module for LLM generation with structured output.

```python
outputs = await synalinks.Generator(
    data_model=Answer,  # Or schema=Answer.get_schema()
    language_model=lm,
    prompt_template=None,  # Custom prompt template
    instructions="Be concise",  # MUST be a string (not a list!)
    examples=None,  # Few-shot examples
    return_inputs=False,  # Include inputs in output
    use_inputs_schema=False,  # Include input schema in prompt
    use_outputs_schema=False,  # Include output schema in prompt
    streaming=False,  # Enable streaming output
)(inputs)
```

### synalinks.ChainOfThought

Generator with automatic thinking step.

```python
outputs = await synalinks.ChainOfThought(
    data_model=Answer,  # Output schema
    language_model=lm,
    return_inputs=True,
)(inputs)
# Output includes "thinking" field automatically
```

### synalinks.Decision

Single-label classification for routing.

```python
decision = await synalinks.Decision(
    question="What type of query is this?",
    labels=["factual", "opinion", "creative"],
    language_model=lm,
)(inputs)
# Returns data model with "label" field
```

### synalinks.Branch

Conditional routing based on decision.

```python
outputs = await synalinks.Branch(
    question="Difficulty level?",
    labels=["easy", "hard"],
    branches=[module_for_easy, module_for_hard],
    language_model=lm,
    return_decision=True,  # Include decision in output
)(inputs)
# Returns tuple of outputs; non-selected branches return None
```

### synalinks.SelfCritique

Self-evaluation with reward score.

```python
critique = await synalinks.SelfCritique(
    language_model=lm,
    return_reward=True,  # Include reward float
    return_inputs=True,
)(previous_output)
# Output includes "critique" and "reward" fields
```

### synalinks.Identity

Pass-through module (no-op).

```python
outputs = await synalinks.Identity()(inputs)
```

---

## Program Classes

### synalinks.Program

Main container for module DAGs.

```python
program = synalinks.Program(
    inputs=inputs,
    outputs=outputs,
    name="my_program",
    description="Program description",
    trainable=True,
)

# Execution
result = await program(input_data)  # Single inference
results = await program.predict(batch, batch_size=32)  # Batch inference

# Training
program.compile(reward=..., optimizer=..., metrics=[...])
history = await program.fit(x=..., y=..., epochs=10, validation_split=0.2)
metrics = await program.evaluate(x=..., y=..., batch_size=32)

# Saving
program.save("path.json")
program = synalinks.Program.load("path.json")
program.save_variables("vars.json")
program.load_variables("vars.json")

# Building
await program.build(InputDataModel)  # Explicit build

# Inspection
program.modules  # List of modules
program.summary()  # Print summary
```

### synalinks.Sequential

Linear chain of modules.

```python
program = synalinks.Sequential(
    [
        synalinks.Input(data_model=Query),
        synalinks.Generator(data_model=Thinking, language_model=lm),
        synalinks.Generator(data_model=Answer, language_model=lm),
    ],
    name="sequential_chain",
)
```

---

## Model Classes

### synalinks.LanguageModel

LLM wrapper supporting multiple providers via LiteLLM.

```python
lm = synalinks.LanguageModel(
    model="ollama/mistral",  # Provider/model format
    # model="openai/gpt-4o-mini",
    # model="anthropic/claude-3-sonnet-20240229",
    # model="gemini/gemini-2.5-pro",
    # model="groq/llama3-8b-8192",
    # model="mistral/codestral-latest",
    # model="xai/grok-code-fast-1",
    # model="azure/<your_deployment_name>",
    api_base=None,    # Optional endpoint override
    timeout=600,      # Timeout in seconds (default 600)
    retry=5,          # Number of retries with exponential backoff (default 5)
    fallback=None,    # Fallback LanguageModel if this one fails
    caching=False,    # Enable caching of LM calls
)
```

**Supported providers:** ollama, openai, anthropic, gemini, mistral, groq, xai, azure, hosted_vllm.

### synalinks.EmbeddingModel

Embedding model wrapper.

```python
em = synalinks.EmbeddingModel(
    model="ollama/mxbai-embed-large",
    # model="openai/text-embedding-3-small",
)
```

---

## Operations (synalinks.ops)

### Concatenation

```python
result = await synalinks.ops.concat(x1, x2, name="combined")
# Or use operator: result = x1 + x2
```

### Masking Operations

```python
# Keep only specified fields
result = await synalinks.ops.in_mask(x, mask=["answer"])
# Remove specified fields
result = await synalinks.ops.out_mask(x, mask=["thinking"])
# Regex pattern matching
result = await synalinks.ops.in_mask(x, pattern="^input_")
result = await synalinks.ops.out_mask(x, pattern="name$")
```

### Aggregation Operations

```python
# Group similar fields into lists (e.g. answer, answer_1 -> answers: [...])
result = await synalinks.ops.factorize(x)
# Expand lists into individual fields (inverse of factorize)
result = await synalinks.ops.decompose(x)
```

### Logical Operations

```python
result = await synalinks.ops.logical_and(x1, x2)  # Or: x1 & x2
result = await synalinks.ops.logical_or(x1, x2)   # Or: x1 | x2
result = await synalinks.ops.logical_xor(x1, x2)  # Or: x1 ^ x2
result = await synalinks.ops.logical_not(x)       # Or: ~x
```

---

## Rewards (synalinks.rewards)

### Built-in Rewards

```python
synalinks.rewards.ExactMatch(in_mask=["answer"])
synalinks.rewards.CosineSimilarity(embedding_model=em, in_mask=["field"])
synalinks.rewards.LMAsJudge(language_model=lm)
synalinks.rewards.ProgramAsJudge(program=judge_program)
```

### Custom Rewards

```python
@synalinks.saving.register_synalinks_serializable()
async def custom_reward(y_true, y_pred):
    return float(y_true.get("x") == y_pred.get("x"))

program.compile(reward=synalinks.rewards.MeanRewardWrapper(fn=custom_reward))
```

---

## Optimizers (synalinks.optimizers)

```python
synalinks.optimizers.RandomFewShot()
synalinks.optimizers.OMEGA(language_model=lm, embedding_model=em)
```

---

## Callbacks (synalinks.callbacks)

```python
synalinks.callbacks.ProgramCheckpoint(
    filepath="best.json",
    monitor="val_reward",
    mode="max",
    save_best_only=True,
)
```

---

## Utilities (synalinks.utils)

```python
synalinks.utils.plot_program(program, to_folder=".", show_schemas=True, show_trainable=True)
synalinks.utils.plot_history(history, to_folder=".")
synalinks.utils.plot_metrics_with_mean_and_std(metrics_list, to_folder=".")
synalinks.utils.plot_metrics_comparison_with_mean_and_std(metrics_dict, to_folder=".")

@synalinks.utils.register_synalinks_serializable()
def my_function(): ...
```

---

## Configuration

```python
synalinks.enable_logging()  # Debug logging
synalinks.enable_observability()  # Tracing (Arize Phoenix compatible)
synalinks.set_seed(42)  # Reproducibility
synalinks.clear_session()  # Clear global state
```

---

## Datasets (synalinks.datasets)

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
input_model = synalinks.datasets.gsm8k.get_input_data_model()
output_model = synalinks.datasets.gsm8k.get_output_data_model()
```
