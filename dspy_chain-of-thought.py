import dspy

lm = dspy.LM('ollama_chat/gemma3:4b', api_base="http://localhost:11434")
dspy.configure(lm=lm)

class QA(dspy.Signature):
    """Given the question, generate the answer"""
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

question = "who provided the assist for the final winning goal in football world cup finals in 2014?"
# predict = dspy.Predict(QA)
# prediction = predict(question=question)
# Answer: Germany's Neuer ... not even close - was supposed to be Andre Schurrle

generate_answer = dspy.ChainOfThought(QA)
prediction = generate_answer(question=question)
# Answer: Lionel Messi ... not correct - was supposed to be Andre Schurrle
# DOES NOT WORK, no such attribute... print(prediction.rationale)

print(prediction)
print(prediction.answer)
#print(lm.inspect_history(1))

multi_step_question = "what is the capital of the birth state of the person who provided the assist for the Mario Gotze's in football world cup finals in 2014?"
output = generate_answer(question=multi_step_question)
print(output)