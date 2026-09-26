#!/usr/bin/env python3
"""Decision models example.

Usage:
    TYPESAFE_API_KEY=... uv run -- python scripts/decision_model.py

Run log:
    none captured (needs a TypeSafe API key).

Demonstrates:
1. A DecisionModel answering a data model of questions (bool, Literal, Rating)
2. Decision with min_confidence: abstains (returns None) when unsure
3. Branch routed by a decision model, branches written by a language model
4. RubricsAsJudge grading weighted criteria in one decision model call
"""

import asyncio
from typing import Literal

import synalinks

synalinks.enable_logging()


class Ticket(synalinks.DataModel):
    message: str = synalinks.Field(description="The customer message")


class Triage(synalinks.DataModel):
    is_billing: bool = synalinks.Field(description="Is the ticket about billing?")
    urgency: Literal["low", "medium", "high"] = synalinks.Field(
        description="How urgent is the ticket?",
    )
    frustration: synalinks.Rating = synalinks.Field(
        description="How frustrated is the customer, from 1 (calm) to 5 (angry)?",
    )


class Reply(synalinks.DataModel):
    reply: str = synalinks.Field(description="The reply to the customer")


async def main():
    # Reads TYPESAFE_API_KEY from the environment.
    decision_model = synalinks.DecisionModel(model="typesafe/jev-latest")
    language_model = synalinks.LanguageModel(model="ollama/mistral")

    ticket = Ticket(message="I was charged twice for my order. Fix this today!")

    # 1. Every field is a question, answered in one call.
    inputs = synalinks.Input(data_model=Ticket)
    outputs = await synalinks.Generator(
        data_model=Triage,
        decision_model=decision_model,
        instructions="Triage the support tickets of an online shop.",
    )(inputs)
    triage = synalinks.Program(inputs=inputs, outputs=outputs, name="triage")
    result = await triage(ticket)
    print(result.prettify_json() if result else "Triage failed")

    # 2. A decision the model is not sure enough about is not taken.
    inputs = synalinks.Input(data_model=Ticket)
    outputs = await synalinks.Decision(
        question="Which team should handle the ticket?",
        labels=["billing", "technical", "sales"],
        decision_model=decision_model,
        min_confidence=0.7,
    )(inputs)
    router = synalinks.Program(inputs=inputs, outputs=outputs, name="team")
    result = await router(ticket)
    print(result.prettify_json() if result else "Not sure enough: escalate to a human")

    # 3. Route cheap, write expensive: the decision model picks the branch,
    # the branches keep their language model.
    inputs = synalinks.Input(data_model=Ticket)
    (billing, technical) = await synalinks.Branch(
        question="Which team should handle the ticket?",
        labels=["billing", "technical"],
        branches=[
            synalinks.Generator(
                data_model=Reply,
                language_model=language_model,
                instructions="Reply as the billing team.",
            ),
            synalinks.Generator(
                data_model=Reply,
                language_model=language_model,
                instructions="Reply as the technical support team.",
            ),
        ],
        decision_model=decision_model,
        return_decision=False,
    )(inputs)
    support = synalinks.Program(
        inputs=inputs, outputs=billing | technical, name="support"
    )
    result = await support(ticket)
    print(result.prettify_json() if result else "No branch selected")

    # 4. Grade weighted criteria in a single decision model call.
    reward = synalinks.rewards.RubricsAsJudge(
        rubrics=[
            {"name": "polite", "description": "The reply is polite.", "weight": 1},
            {
                "name": "actionable",
                "description": "The reply says what happens next.",
                "weight": 2,
            },
        ],
        decision_model=decision_model,
    )
    grades = await reward.program(
        [None, Reply(reply="Sorry! We refunded the duplicate charge today.")]
    )
    print(grades.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
