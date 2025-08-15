import dspy

lm = dspy.LM('ollama_chat/gemma3:4b', api_base="http://localhost:11434")
dspy.configure(lm=lm)

class QA(dspy.Signature):
    """Given the question, generate the answer"""
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class ChainOfThoughtModule(dspy.Module):
    def __init__(self, callbacks=None):
        self.cot = dspy.ChainOfThought(QA)

    def forward(self, question):
        return self.cot(question=question)
    

multi_step_question = "what is the capital of the birth state of the person who provided the assist for the Mario Gotze's in football world cup finals in 2014?"
cot = ChainOfThoughtModule()
output = cot(question=multi_step_question)
print(output)