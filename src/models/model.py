from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def initialize_model_and_tokenizer(model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # 加载模型和分词器
    model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer