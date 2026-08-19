import asyncio
import os
import sys

# Đảm bảo Python hiểu thư mục src là một package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import ssl
ssl.create_default_context = ssl._create_unverified_context

# Tự động load file .env ở thư mục gốc
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip().strip("'").strip('"')


from tap_agent.gemini_provider import GeminiProvider
from tap_agent.agent_harness import AgentHarness
from tap_agent.coding_session import CodingSession
from tap_agent.tools import AVAILABLE_TOOLS, execute_tool

async def main():
    prompt_path = os.path.join(os.path.dirname(__file__), 'tap_agent', 'system_prompt.md')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # Khởi tạo Gemini model
    provider = GeminiProvider(
        model="gemini-3.1-flash-lite", # Hoặc model bạn muốn dùng
        api_key=os.environ.get("GEMINI_API_KEY"),
        system_instruction=system_prompt
    )
    
    # Gom các thành phần lại vào Harness
    harness = AgentHarness(
        provider=provider,
        tools=AVAILABLE_TOOLS,
        tool_executor=execute_tool
    )
    
    # Mở session chat
    session = CodingSession(harness=harness)
    
    print("Agent is ready. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("Agent is thinking...")
        try:
            reply = await session.prompt(user_input)
            print(f"\nAgent: {reply.content}")
        except Exception as e:
            print(f"\n[Lỗi Hệ Thống]: Quá trình xử lý thất bại.")
            print(f"Chi tiết: {e}")
            print("Vui lòng đợi vài giây và thử lại (có thể do API Rate Limit).")

if __name__ == "__main__":
    asyncio.run(main())
