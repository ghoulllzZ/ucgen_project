import os
from openai import OpenAI
from openai import AuthenticationError, APIError, APIConnectionError, RateLimitError


def init_gpt_client():
    """初始化GPT客户端（验证API密钥）"""
    # 从环境变量读取API密钥（安全方式，避免硬编码）
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误：未配置OPENAI_API_KEY环境变量！")
        print("  👉 Windows设置：set OPENAI_API_KEY=你的密钥")
        print("  👉 Mac/Linux设置：export OPENAI_API_KEY=你的密钥")
        return None

    # 初始化客户端（如需使用国内代理，添加base_url参数）
    try:
        client = OpenAI(api_key=api_key)
        # 可选：添加代理地址（示例）
        # client = OpenAI(api_key=api_key, base_url="https://你的代理域名/v1")
        return client
    except Exception as e:
        print(f"❌ 客户端初始化失败：{str(e)}")
        return None


def interactive_chat():
    """启动交互式GPT对话"""
    # 初始化客户端
    client = init_gpt_client()
    if not client:
        return

    # 初始化对话上下文（保留多轮对话记忆）
    messages = [
        {"role": "system", "content": "你是一个友好的助手，回答简洁明了，易于理解。"}
    ]

    print("🎉 GPT交互式对话已启动！")
    print("💡 输入问题即可对话，输入 'exit'/'quit' 退出程序")
    print("-" * 50)

    while True:
        # 获取用户输入
        user_input = input("\n你：").strip()

        # 退出逻辑
        if user_input.lower() in ["exit", "quit", "退出", "结束"]:
            print("👋 对话结束，再见！")
            break

        # 空输入跳过
        if not user_input:
            print("⚠️ 请输入有效内容！")
            continue

        # 将用户输入加入上下文
        messages.append({"role": "user", "content": user_input})

        try:
            # 调用GPT API
            print("🤖 GPT正在思考...")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 轻量模型，性价比高
                messages=messages,
                temperature=0.7,  # 回答多样性（0-1，值越高越随机）
                timeout=10  # 超时时间15秒
            )

            # 解析GPT回复
            gpt_reply = response.choices[0].message.content.strip()
            print(f"\nGPT：{gpt_reply}")

            # 将GPT回复加入上下文（保留多轮对话记忆）
            messages.append({"role": "assistant", "content": gpt_reply})

            # 可选：限制上下文长度（避免tokens超限）
            # 当对话轮次过多时，可删除早期的非系统消息
            if len(messages) > 20:  # 保留系统消息+最近19轮对话
                messages = [messages[0]] + messages[-19:]

        except AuthenticationError:
            print("❌ 错误：API密钥无效/过期，请检查密钥！")
            break
        except RateLimitError:
            print("❌ 错误：API调用超限/额度不足，请稍后重试！")
        except APIConnectionError:
            print("❌ 错误：网络连接失败（检查网络/代理）！")
        except APIError:
            print("❌ 错误：GPT服务器返回异常，请稍后重试！")
        except Exception as e:
            print(f"❌ 未知错误：{str(e)}")


if __name__ == "__main__":
    interactive_chat()