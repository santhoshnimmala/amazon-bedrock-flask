import boto3
from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")
llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": 2048}
model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

@app.route("/")
def chat_page():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input")
    response = model.predict(input=user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
