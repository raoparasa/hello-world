import dspy

lm = dspy.LM('ollama_chat/gemma3:4b', api_base="http://localhost:11434")
dspy.configure(lm=lm)

class QA(dspy.Signature):
    """Given the question, generate the answer"""
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

predict = dspy.Predict(QA)
prediction = predict(question="who scored the final goal in football world cup finals in 2014?")
print(prediction.answer)

print(lm.inspect_history(1))