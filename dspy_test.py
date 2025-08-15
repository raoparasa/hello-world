import dspy

lm = dspy.LM('ollama_chat/gemma3:4b', api_base="http://localhost:11434")
dspy.configure(lm=lm)
response = lm("Say this is a test!", temperature=0.7)
print(response)