from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
#from soc_lc_2 import crew, get_compliance_status
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_credentials', methods=['POST'])
def save_credentials():
    data = request.json
    session['aws_access_key'] = data.get('aws_access_key')
    session['aws_secret_key'] = data.get('aws_secret_key')
    session['aws_region'] = data.get('aws_region', 'us-east-1')
    session['admin_account_id'] = data.get('admin_account_id')
    
    return jsonify({"status": "success", "message": "Credentials saved successfully"})

@app.route('/run_compliance_check', methods=['POST'])
def run_compliance_check():
    try:
        # Set environment variables from session
        os.environ['AWS_ACCESS_KEY_ID'] = session.get('aws_access_key')
        os.environ['AWS_SECRET_ACCESS_KEY'] = session.get('aws_secret_key')
        os.environ['AWS_DEFAULT_REGION'] = session.get('aws_region')
        
        # Run compliance check
        result = get_compliance_status(session.get('admin_account_id'))
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/run_full_analysis', methods=['POST'])
def run_full_analysis():
    try:
        # Set environment variables from session
        os.environ['AWS_ACCESS_KEY_ID'] = session.get('aws_access_key')
        os.environ['AWS_SECRET_ACCESS_KEY'] = session.get('aws_secret_key')
        os.environ['AWS_DEFAULT_REGION'] = session.get('aws_region')
        
        # Run the full CrewAI workflow
        result = crew.kickoff()
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True) 