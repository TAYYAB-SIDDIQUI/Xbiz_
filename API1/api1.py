from flask import Flask, request, jsonify

app = Flask(__name__)

# Store multiple records instead of just one
data = [
    {"id": 1, "name": "alice", "age": 23}
]

@app.route("/")
def index():
    return jsonify(message="api1 is running"), 200

@app.route("/status")
def status():
    return jsonify(message="Status OK"), 200

@app.route("/login")
def login():
    return jsonify(message="login required"), 401

@app.route('/read', methods=["GET"])
def Read():
    return jsonify(data), 200

@app.route('/update/<int:item_id>', methods=["PUT"])
def Update(item_id):
    update_data = request.get_json()
    for item in data:
        if item["id"] == item_id:
            item.update(update_data)
            return jsonify(message=f"Record {item_id} updated", updated=item), 200
    return jsonify(message=f"No record with id {item_id} found"), 404

@app.route('/create', methods=["POST"])
def Create():
    add_data = request.get_json()
    print(add_data)
    if not add_data:
        return jsonify(message="No data provided"), 400
    
    data.append(add_data)
    return jsonify(message="created !!!!", new_data=add_data), 201

@app.route('/delete/<int:item_id>', methods=["DELETE"])
def Delete(item_id):
    global data
    before_count = len(data)
    data = [item for item in data if item.get("id") != item_id]
    if len(data) == before_count:
        return jsonify(message=f"No record with id {item_id} found"), 404
    return jsonify(message=f"Record with id {item_id} deleted"), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
