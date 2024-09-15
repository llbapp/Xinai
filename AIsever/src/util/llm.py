
from openai import OpenAI
from src.config.config import url
#记得修改url，不知道为什么我无法引用config里的参数
# url = "https://u429062-8653-35f1c9ab.cqa1.seetacloud.com:8443/v1"

def load_llm():
        """
        加载llm模型，参数base_url是api网址，在config文件里更改
        """
        openai = OpenAI(
            api_key='YOUR_API_KEY',
            base_url=url
        )
        model_name = "internlm2"
        return openai, model_name
    

if __name__ == '__main__':
    openai, model_name = load_llm(url)
    response = openai.chat.completions.create(
      model="internlm2",
      messages=[
        {"role": "user", "content": "hi"},
      ],
        temperature=0.8,
        top_p=0.8
    )
    message_content = response.choices[0].message.content
    print(message_content)