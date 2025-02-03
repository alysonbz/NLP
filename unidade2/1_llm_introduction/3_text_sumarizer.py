from transformers import pipeline

prompt = "My name is Bob. Each day I drive my kids to school. My daughter goes to a school that’s far from our house. It takes 30 minutes to get there. " \
         "Then I drive my son to his school. It’s close to my job. My daughter is in the sixth grade and my son is in the second. They are both good students. My daughter usually sings her favorite songs while I drive. My son usually sleeps. "\
        "I arrive at the office at 8:30 AM. I say good morning to all my workmates then I get a big cup of hot coffee." \
        " I turn on my computer and read my email. Some days I have a lot to read. Soon I need another cup of coffee."
llm = pipeline(task="summarization",
               model="t5-small")

output = llm(prompt, max_length=50, clean_up_tokenization_spaces=True)

print(output[0]["summary_text"])
