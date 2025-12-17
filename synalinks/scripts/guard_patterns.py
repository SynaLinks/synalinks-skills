#!/usr/bin/env python3
"""Input and Output Guard patterns using XOR operator.

Usage:
    uv run -- python scripts/guard_patterns.py

This script demonstrates:
1. Input guards to block invalid requests
2. Output guards to filter unsafe responses
3. XOR (^) operator for computation bypass
4. OR (|) operator for result merging
"""

import asyncio
import synalinks

synalinks.enable_logging()


class ConversationalInputGuard(synalinks.Module):
    """Input guard that blocks messages containing blacklisted words."""

    def __init__(
        self,
        blacklisted_words=None,
        warning_message="I'm unable to comply with your request",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words or []
        self.warning_message = warning_message

    async def call(self, inputs, training=False):
        """Return warning message if blocked, None otherwise."""
        if not synalinks.is_chat_messages(inputs):
            raise ValueError("Input guard works only for ChatMessages")

        if not inputs or not inputs["messages"]:
            return None

        content = inputs["messages"][-1]["content"]
        content_lower = content.lower()

        if any(bw.lower() in content_lower for bw in self.blacklisted_words):
            return synalinks.ChatMessage(
                role="assistant",
                content=self.warning_message,
            )
        return None

    async def compute_output_spec(self, inputs, training=False):
        """Define output schema."""
        if not synalinks.is_chat_messages(inputs):
            raise ValueError("Input guard works only for ChatMessages")
        return synalinks.ChatMessage.to_symbolic_data_model(name=self.name)

    def to_config(self):
        """Serialization config for Module subclasses."""
        return {
            "name": self.name,
            "description": self.description,
            "blacklisted_words": self.blacklisted_words,
            "warning_message": self.warning_message,
        }


class ConversationalOutputGuard(synalinks.Module):
    """Output guard that replaces responses containing blacklisted words."""

    def __init__(
        self,
        blacklisted_words=None,
        warning_message="I'm unable to comply with your request",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words or []
        self.warning_message = warning_message

    async def call(self, inputs, training=False):
        """Return warning message if output should be blocked, None otherwise."""
        if not synalinks.is_chat_message(inputs):
            raise ValueError("Output guard works only for ChatMessage")

        if not inputs:
            return None

        content = inputs["content"]
        content_lower = content.lower()

        if any(bw.lower() in content_lower for bw in self.blacklisted_words):
            return synalinks.ChatMessage(
                role="assistant",
                content=self.warning_message,
            )
        return None

    async def compute_output_spec(self, inputs, training=False):
        """Define output schema."""
        if not synalinks.is_chat_message(inputs):
            raise ValueError("Output guard works only for ChatMessage")
        return synalinks.ChatMessage.to_symbolic_data_model(name=self.name)

    def to_config(self):
        """Serialization config for Module subclasses."""
        return {
            "name": self.name,
            "description": self.description,
            "blacklisted_words": self.blacklisted_words,
            "warning_message": self.warning_message,
        }


async def build_input_guarded_program(language_model):
    """Build a chatbot with input guard.

    Logic flow:
    1. Check input for blacklisted words
    2. If warning exists: XOR makes inputs None, bypassing generator
    3. Return warning (if blocked) OR answer (if allowed)
    """
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Check input
    warning_msg = await ConversationalInputGuard(
        blacklisted_words=["forbidden", "blocked"],
    )(inputs)

    # XOR: if warning exists, inputs becomes None (bypassing generator)
    guarded_inputs = warning_msg ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(
        language_model=language_model,
    )(guarded_inputs)

    # OR: return warning if it exists, otherwise return answer
    outputs = warning_msg | answer

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="input_guarded_chatbot",
        description="A chatbot with input guard",
    )


async def build_output_guarded_program(language_model):
    """Build a chatbot with output guard.

    Logic flow:
    1. Generate response
    2. Check output for blacklisted words
    3. XOR + OR: if warning exists, replace answer with warning
    """
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    answer = await synalinks.Generator(
        language_model=language_model,
    )(inputs)

    # Check output
    warning_msg = await ConversationalOutputGuard(
        blacklisted_words=["corn", "dangerous"],
    )(answer)

    # XOR + OR: if warning exists, replace answer with warning
    outputs = (answer ^ warning_msg) | warning_msg

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="output_guarded_chatbot",
        description="A chatbot with output guard",
    )


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    # Test input guard
    print("=== Input Guard Demo ===")
    input_program = await build_input_guarded_program(lm)
    synalinks.utils.plot_program(input_program, to_folder=".")

    # Blocked input
    result1 = await input_program(
        synalinks.ChatMessages(
            messages=[{"role": "user", "content": "Tell me about forbidden topics"}]
        )
    )
    print(f"Blocked input result: {result1.prettify_json()}")

    # Allowed input
    result2 = await input_program(
        synalinks.ChatMessages(
            messages=[{"role": "user", "content": "What is the capital of France?"}]
        )
    )
    print(f"Allowed input result: {result2.prettify_json()}")

    # Test output guard
    print("\n=== Output Guard Demo ===")
    output_program = await build_output_guarded_program(lm)

    result3 = await output_program(
        synalinks.ChatMessages(
            messages=[{"role": "user", "content": "Tell me a story about corn"}]
        )
    )
    print(f"Output guard result: {result3.prettify_json()}")


if __name__ == "__main__":
    asyncio.run(main())
