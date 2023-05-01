import yaml
import subprocess
from flask import Flask, jsonify, request, render_template_string
import os
os.environ['PATH'] = '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'

command = "python bot_v7.py {' '.join(arg_list)}"


app = Flask(__name__)

@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        with open('config.yaml', 'r') as file:
            data = yaml.safe_load(file)
        return jsonify(data)
    elif request.method == 'POST':
        data = request.form['config']
        # Do something with the config_value (e.g. write it to a file)
        return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

@app.route('/bot_v7.py', methods=['POST'])
def run_bot():
    fields = request.form
    arg_list = [f"--{key}='{value}'" for key, value in fields.items()]
    command = f"python bot_v7.py {' '.join(arg_list)}"
    try:
        process_output = subprocess.check_output(command, universal_newlines=True, shell=True)
    except subprocess.CalledProcessError:
        process_output = "Error running bot"
    return render_template_string("<pre>{{ output }}</pre>", output=process_output)

if __name__ == '__main__':
    app.run(debug=True)