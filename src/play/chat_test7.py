from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
model_id = "THUDM/chatglm3-6b"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
import os

# 设置环境变量
os.environ["HUGGINGFACEHUB_API_TOKEN"] ='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'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto" ,
    trust_remote_code=True
    #attn_implementation="flash_attention_2", # if you have an ampere GPU
)

pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_new_tokens=100,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                )   
llm = HuggingFacePipeline(pipeline=pipe)

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_huggingface import ChatHuggingFace

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

chat_model = ChatHuggingFace(llm=llm)

res = chat_model.invoke(messages)
print(res.content)