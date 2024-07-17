from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline,BitsAndBytesConfig

from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



model_id = "THUDM/chatglm3-6b"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

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

# 定义消息和提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
         ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | llm

with_message_history = RunnableWithMessageHistory(chain, 
                                                  get_session_history,
                                                  input_messages_key="messages"
)

config = {"configurable": {"session_id": "abc"}}


for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(1)
    print(r, end="|")