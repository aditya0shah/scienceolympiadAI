from flask import Flask, render_template, request
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.callbacks import get_openai_callback
import openai
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAPI_API_KEY')

app = Flask(__name__)




llm = OpenAI(
    temperature = 0, 
    openai_api_key = OPENAI_API_KEY,
    model_name='text-davinci-003'
)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'spent a total of {cb.total_tokens} tokens')
    return result

conversation_sum = ConversationChain(
    llm=llm, 
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)
)

print(count_tokens(conversation_sum, 
             """You get a perfect score on every single Science Olympiad anatomy and physiology test you take. You have never gotten a question wrong. You ONLY answer questions related
to the anatomy and physiology event in Science Olympiad, whether that be content questions, practice questions, explanations, or any other request out of you that pertains to 
anatomy and physiology in science olympiad. If request does not pretain to anatomy and physiology in science olympiad, respond with "I am only an expert in anatomy/physiology please 
just ask me about that."""))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    chatbot_response = count_tokens(conversation_sum, user_input)


    return render_template('index.html', user_input=user_input, chatbot_response=chatbot_response)

if __name__ == '__main__':
    app.run(debug=True)