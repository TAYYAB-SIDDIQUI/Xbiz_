from flask import Flask,request,jsonify

app=Flask(__name__)

@app.route("/")
def index():
    return jsonify(message="api2 is running"),200

@app.route("/status")
def status():
    return jsonify(message="Status 2 OK"),200

@app.route("/login")
def login():
    return jsonify(message="login 2 required"),201

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5001)