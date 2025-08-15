import dspy

##lm = dspy.LM('ollama_chat/qwen3:4b', api_base="http://localhost:11434")
lm = dspy.LM('gemini/gemini-2.5-flash', api_key='AIzaSyCKWBUF4M-Oat76LiejyOjvPzoEL5kzR28')
dspy.configure(lm=lm)

class QA(dspy.Signature):
    """Given the question, generate the answer"""
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class DoubleChainOfThoughtModule(dspy.Module):
    def __init__(self, callbacks=None):
        self.cot1 = dspy.ChainOfThought("question -> step_by_step_thought")
        self.cot2 = dspy.ChainOfThought("question, thought -> one_word_answer")

    def forward(self, question):
        thought = self.cot1(question=question).step_by_step_thought
        answer = self.cot2(question=question, thought=thought).one_word_answer
        return dspy.Prediction(thought=thought, answer=answer)
    

multi_step_question = "what is the capital of the birth state of the person who provided the assist for the Mario Gotze's in football world cup finals in 2014?"
doubleCot = DoubleChainOfThoughtModule()
output = doubleCot(question=multi_step_question)
print(output)

## Output from Gemma 3 Model  
## Prediction(
##    thought="1. **Mario Gotze's Birth State:** Mario Gotze was born in Bochum, Germany.\n2. **Bochum's Location:** Bochum is located in the state of North Rhine-Westphalia (Nordrhein-Westfalen).\n3. **Capital of North Rhine-Westphalia:** The capital of North Rhine-Westphalia is Düsseldorf.\n\nTherefore, the capital of the birth state of the person who provided the assist for the Mario Gotze's in football world cup finals in 2014 is Düsseldorf.",
##    answer='Düsseldorf'
## )
##
## Output from Llama2 Model
## Prediction(
##    thought="Here are the steps I took to arrive at my answer: \n1. Identified the player's birthplace from available sources \n2. Determined the country where the World Cup final was held in 2014 \n3. Found the capital of that country \nTherefore, the capital of the birth state of the person who provided the assist is Berlin, Germany.",       
##    answer='Berlin'
## )
##
## Output from Qwen3 Model
## Prediction(
##    thought="Here are the steps I took to arrive at my answer: \n1. Identified the player's birthplace from available sources \n2. Determined the country where the World Cup final was held in 2014 \n3. Found the capital of that country \nTherefore, the capital of the birth state of the person who provided the assist is Berlin, Germany.",       
##    answer='Berlin'
## )
##
## Output from Gemini 2.5 Pro
## Prediction(
##    thought="1.  The question asks for the capital of the birth state of the person who assisted Mario Götze's goal in the 2014 FIFA World Cup final.\n2.  First, I need to identify the person who provided the assist. The goal was scored by Mario Götze for Germany against Argentina. The assist was provided by André Schürrle.\n3.  Next, I need to find André Schürrle's birth state. A quick search reveals he was born in Ludwigshafen, Germany.\n4.  Then, I need to determine which German state Ludwigshafen is in. Ludwigshafen is located in the state of Rhineland-Palatinate (Rheinland-Pfalz).\n5.  Finally, I need to find the capital of the state of Rhineland-Palatinate. The capital city of Rhineland-Palatinate is Mainz.\n6.  Therefore, the final answer is Mainz.",
##    answer=None
## )
##
## Output from Gemini 2.5 Flash
## Prediction(                                                                                                                                                                  
##    thought="1.  **Identify the goal:** Mario Götze's goal in the 2014 FIFA World Cup Final.\n2.  **Identify the assist provider:** André Schürrle provided the assist for Mario Götze's goal.\n3.  **Identify the birth state of the assist provider:** André Schürrle was born in Ludwigshafen am Rhein, which is in the German state of Rhineland-Palatinate (Rheinland-Pfalz).\n4.  **Identify the capital of that state:** The capital of Rhineland-Palatinate is Mainz.",
##    answer='Mainz'
## )
##