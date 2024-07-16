# main.py
from langchain.llms.base import LLM
from model import initialize_model_and_tokenizer

class custom_llm(LLM):
  
    model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
    model1, tokenizer = initialize_model_and_tokenizer(model_name)
    def _call(self, prompt, stop=None, run_manager=None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        result = self.model1.generate(input_ids=inputs.input_ids, max_new_tokens=20)
        result = self.tokenizer.decode(result[0], skip_special_tokens=True)
        return result

    @property
    def _llm_type(self) -> str:
        return "custom"

# 实例化并测试CustomLLM
a_custom_llm = custom_llm()
response = a_custom_llm._call("你好，请问有什么我可以帮助你的？")
print(response)
