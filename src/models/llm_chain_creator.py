from langchain import LLMChain

from langchain import PromptTemplate

from cust_llm import cust_llm

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=cust_llm)