import dspy

lm = dspy.LM('ollama_chat/gemma3:4b', api_base="http://localhost:11434")
dspy.configure(lm=lm)

predict = dspy.Predict("question -> answer")
prediction = predict(question="who scored the final goal in football world cup finals in 2014?")
print(prediction.answer)

print(lm.inspect_history(1))