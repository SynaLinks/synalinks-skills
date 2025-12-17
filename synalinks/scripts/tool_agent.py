#!/usr/bin/env python3
"""Function-calling agent with tools.

Usage:
    uv run -- python scripts/tool_agent.py

This script demonstrates:
1. Defining tools with @register_synalinks_serializable
2. FunctionCallingAgent for autonomous tool use
3. Execution trajectory tracking
"""

import asyncio
import synalinks

synalinks.enable_logging()


class Query(synalinks.DataModel):
    """Input data model."""
    query: str = synalinks.Field(description="The user query")


class FinalAnswer(synalinks.DataModel):
    """Final answer from agent."""
    reasoning: str = synalinks.Field(description="How the answer was derived")
    answer: str = synalinks.Field(description="The final answer")


@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression: Mathematical expression like '2 + 2' or '(10 * 5) / 2'.
            Only numbers and +, -, *, /, (, ), . operators allowed.

    Returns:
        dict with 'result' and 'log' keys
    """
    # Validate input
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {
            "result": None,
            "log": "Error: Invalid characters. Only numbers and +,-,*,/,() allowed.",
        }

    try:
        result = eval(expression, {"__builtins__": None}, {})
        return {"result": round(float(result), 4), "log": "Success"}
    except Exception as e:
        return {"result": None, "log": f"Error: {e}"}


@synalinks.utils.register_synalinks_serializable()
async def get_weather(city: str):
    """Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        dict with weather information
    """
    # Simulated weather data
    weather_data = {
        "paris": {"temp": 18, "condition": "cloudy"},
        "london": {"temp": 15, "condition": "rainy"},
        "tokyo": {"temp": 22, "condition": "sunny"},
        "new york": {"temp": 20, "condition": "partly cloudy"},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return {
            "city": city,
            "temperature": data["temp"],
            "condition": data["condition"],
            "log": "Success",
        }
    return {
        "city": city,
        "temperature": None,
        "condition": None,
        "log": f"Weather data not available for {city}",
    }


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    # Define tools
    tools = [
        synalinks.Tool(calculate),
        synalinks.Tool(get_weather),
    ]

    # Build agent
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=FinalAnswer,
        tools=tools,
        language_model=lm,
        max_iterations=5,
        autonomous=True,
        return_inputs_with_trajectory=True,
        instructions=[
            "Use tools when needed to answer the question",
            "For math questions, always use the calculate tool",
            "Explain your reasoning in the response",
        ],
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="multi_tool_agent",
        description="Agent with calculator and weather tools",
    )

    # Visualize
    synalinks.utils.plot_program(agent, to_folder=".", show_schemas=True)

    # Test calculations
    print("=== Math Question ===")
    result1 = await agent(Query(query="What is (25 * 4) + 17?"))
    print(result1.prettify_json())

    # Test weather
    print("\n=== Weather Question ===")
    result2 = await agent(Query(query="What's the weather like in Tokyo?"))
    print(result2.prettify_json())

    # Test combined
    print("\n=== Combined Question ===")
    result3 = await agent(
        Query(query="If the temperature in Paris is doubled, what would it be?")
    )
    print(result3.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
