from transformers import pipeline

context = "Lucas goes to school every day of the week. He has many subjects to go to " \
          "each school day: English and History."
question = "What subjects does Lucas study?"
llm = pipeline(task="question-answering",
               model="t5-small")

output = llm(question=question, context=context)

print(output["answer"])
