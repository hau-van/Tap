import asyncio
import json
import os

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.tap_coding.tools import create_bash_tool


class MockToolCall:
    def __init__(self, tool_id: str, arguments: dict):
        self.id = tool_id
        self.arguments = arguments

async def main():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    bash_tool = create_bash_tool(cwd=".")
    
    tools = [{
        "type": "function",
        "function": {
            "name": bash_tool["name"],
            "description": bash_tool["description"],
            "parameters": bash_tool["input_schema"]
        }
    }]
    
    # 2. KHAI BÁO RÕ KIỂU DỮ LIỆU CHO MESSAGES
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "Bạn là một Coding Agent. Hãy sử dụng bash tool khi cần tương tác với hệ điều hành."},
        {"role": "user", "content": "Hãy chạy lệnh kiểm tra thư mục hiện tại có những file nào (dùng ls -la hoặc dir)."}
    ]
    
    print("🤖 Agent đang suy nghĩ...")
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools, # type: ignore (Thêm dòng này nếu Pylance vẫn cảnh báo biến tools)
        tool_choice="auto"
    )
    
    response_msg = response.choices[0].message
    
    # 3. CHUYỂN OBJECT THÀNH DICT TRƯỚC KHI APPEND
    # Hàm model_dump(exclude_none=True) sẽ loại bỏ các trường null, đúng định dạng API cần
    messages.append(response_msg.model_dump(exclude_none=True)) 
    
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            if tool_call.function.name == "bash":
                raw_args = tool_call.function.arguments
                parsed_args = json.loads(raw_args)
                
                print(f"⚙️ LLM quyết định gọi Tool: bash")
                
                tc = MockToolCall(tool_id=tool_call.id, arguments=parsed_args)
                result = await bash_tool["executor"](tc)
                
                print(f"✅ Lệnh chạy xong. Exit code: {result['ok']}")
                
                # 4. ĐẢM BẢO CONTENT LÀ STRING
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": str(result["content"]) 
                })
                
        print("🤖 Agent đang đọc kết quả và phân tích...")
        final_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        print("\n=== CÂU TRẢ LỜI CUỐI CÙNG CỦA AGENT ===")
        print(final_response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())