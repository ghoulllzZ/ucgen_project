import os
from openai import OpenAI
from openai import AuthenticationError, APIError, APIConnectionError, RateLimitError


def test_gpt_api():
    """验证GPT API调用是否成功"""
    # 1. 配置API密钥（推荐通过环境变量设置，避免硬编码泄露）
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # 若未设置环境变量，可临时在这里填写（仅测试用，生产环境务必用环境变量）
        api_key = ""
        print("❌ 错误：未配置OPENAI_API_KEY环境变量，请先设置")
        return False

    try:
        # 2. 初始化客户端
        client = OpenAI(api_key=api_key)

        # 3. 发起测试请求（调用最基础的gpt-3.5-turbo模型）
        print("🔍 正在测试GPT API调用...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 测试用轻量模型，成本低
            messages=[
                {"role": "user", "content": "请回复'API调用成功'，仅验证连通性"}
            ],
            temperature=0,  # 固定输出，便于验证
            timeout=10  # 超时时间10秒
        )

        # 4. 解析并输出结果
        if response and response.choices:
            reply = response.choices[0].message.content.strip()
            print(f"✅ API调用成功！GPT回复：{reply}")
            print(f"📌 响应ID：{response.id} | 消耗tokens：{response.usage.total_tokens}")
            return True

    except AuthenticationError:
        print("❌ 错误：API密钥无效或已过期，请检查密钥是否正确")
    except RateLimitError:
        print("❌ 错误：API调用次数超限或额度不足，请检查账户额度")
    except ConnectionError:
        print("❌ 错误：网络连接失败，无法访问OpenAI服务器（可能需要代理）")
    except APIError:
        print("❌ 错误：OpenAI服务器返回异常，请稍后重试或检查模型名称是否正确")
    except Exception as e:
        print(f"❌ 未知错误：{str(e)}")

    return False


if __name__ == "__main__":
    # 执行验证
    test_gpt_api()