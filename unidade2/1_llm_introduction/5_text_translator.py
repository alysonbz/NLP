from transformers import pipeline

prompt = "I want you to stay, Till I'm in the grave, Till I rot away, dead and buried, Till I'm in the casket you carry"

llm = pipeline("translation_en_to_es",
               model="Helsinki-NLP/opus-mt-en-es")
output = llm(prompt, clean_up_tokenization_spaces=True)

print(output[0]["translation_text"])