from flask import Flask, render_template, Response
import pickle
import os

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logs')
def view_logs():
    log_file_path = os.path.join(app.root_path, 'trading_system.log')
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        log_content = f"Failed to read log file: {e}"
    return render_template('logs.html', logs=log_content)

if __name__ == '__main__':
    app.run(debug=True)