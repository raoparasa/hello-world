import dspy

lm = dspy.LM('ollama_chat/qwen3:4b', api_base="http://localhost:11434")
dspy.configure(lm=lm)

from pydantic import BaseModel, Field

class AnswerConfidence(BaseModel):
    answer: str = Field("Answer. 1-5 words.")
    confidence: float = Field("Your confidence between 0-1.")

class QAWithConfidence(dspy.Signature):
    """Given user's question, answer it and also give your confidence value"""
    question = dspy.InputField()
    answer: AnswerConfidence = dspy.OutputField()
    
predict = dspy.TypedChainOfThought(QAWithConfidence)

question = "who provided the assist for the goal in football world cup finals in 2014?"
output = predict(question=question)
print(output.answer)
print(output.answer.answer)
print(output.answer.confidence)
print(type(output.answer.confidence))

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