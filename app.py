from flask import Flask, render_template, request, jsonify
from my_model_inference import get_responses, N_RESPONSE
from utils import preprocess

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/ask", methods=['POST'])
def ask():
    message = str(request.form['messageText'])
    message = preprocess([message])[0]
    response = get_responses(message, N_RESPONSE)

    while True:
        if message in ['bye', 'quit', 'exit']:
            response = 'Hope to see you soon! Bye!'
            print(response)
            return jsonify({'status': 'OK', 'answer': response})
        elif len(message.strip()) == 0:
            response = 'Please type in text message!'
            print(response)
            return jsonify({'status': 'OK', 'answer': response})
        else:
            res = ''
            for i in range(len(response)):
                res += "Suggestion[" + str(i+1) + ']: ' + response[i][0] + '<br>'
            res += '<br>< If there is NO desired suggestions, RETYPE again! Otherwise, type in new message. >'
            print(res)
            return jsonify({'status': 'OK', 'answer': res})


if __name__ == "__main__":
    # Point your browser to http://localhost:5000/ (http://127.0.0.1:5000/)
    app.run()
