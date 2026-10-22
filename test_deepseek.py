import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有读取到 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),
)

response = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    messages=[
        {
            "role": "user",
            "content": "请只回复：连接成功",
        }
    ],
)

print(response.choices[0].message.content)