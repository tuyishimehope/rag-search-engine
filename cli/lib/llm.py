import os

from dotenv import load_dotenv


def call_llm(content):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": content,
        }
    ]
    response = client.chat.completions.create(
        model="openrouter/free", messages=messages)
    output = response.choices[0].message.content
    usage = response.usage
    if usage is not None:
        print("Prompt tokens: ", usage.prompt_tokens)
        print("Response tokens: ", usage.completion_tokens)
    else:
        print("Token usage unavailable")
    return output
