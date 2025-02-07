from flask import Flask, jsonify, request, render_template
import markdown
import os
from src import qa

app = Flask(__name__, static_folder="", template_folder="")

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in the same directory

@app.route('/run_script', methods=['POST'])
def run_script():
    data = request.json
    user_input = data.get("text", "")

    # Convert input text (Markdown) to HTML
    html_output = markdown.markdown(qa(user_input))

    return jsonify({'html': html_output})

if __name__ == '__main__':
    app.run(debug=True)
