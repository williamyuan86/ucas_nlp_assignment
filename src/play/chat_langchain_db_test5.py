from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from transformers import BitsAndBytesConfig,AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain import HuggingFacePipeline

# 配置量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)


# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")
model = AutoModelForCausalLM.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat", config=quantization_config)
pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_length=1024)
local_llm = HuggingFacePipeline(pipeline=pipe)


db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db_chain = SQLDatabaseChain(llm=local_llm, db=db, verbose=True)
result = db_chain.run("SELECT COUNT(*) FROM employees")
print(result)