import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import base64
from io import BytesIO
from PIL import Image
from bson import ObjectId

# ------------------ DATABASE CONNECTION ------------------
def get_database():
    CONNECTION_URL = os.env.get("CONNECTION_URL")
    if not CONNECTION_URL:
        raise Exception("Missing CONNECTION_URL environment variable")
    client = MongoClient(CONNECTION_URL)
    return client['AttendEase']

# ------------------ FACE ENCODING FUNCTIONS ------------------
def getEncodings():
    """Retrieve all stored face encodings from database"""
    dbname = get_database()
    collection_name = dbname["encodings"]
    items = collection_name.find({})
    
    known_images = []
    encodings = []
    
    for i in items:
        i.pop("_id")
        for name, encoding in i.items():
            known_images.append(name)
            encodings.append(np.array(encoding))
    
    return known_images, encodings

def get_face_embedding(img_file):
    """Extract face embedding using DeepFace with better error handling"""
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().timestamp()}.jpg"
        
        if hasattr(img_file, 'save'):
            img_file.save(temp_path)
        else:
            # If it's already a file path
            temp_path = img_file
        
        print(f"Processing image: {temp_path}")
        
        # Get face embedding using DeepFace with multiple detectors
        try:
            # Try with default detector first
            embedding_objs = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet",
                enforce_detection=True,
                detector_backend="opencv"
            )
        except:
            print("OpenCV detector failed, trying with RetinaFace...")
            try:
                embedding_objs = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=True,
                    detector_backend="retinaface"
                )
            except:
                print("RetinaFace failed, trying with MTCNN...")
                embedding_objs = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=False,  # More lenient
                    detector_backend="mtcnn"
                )
        
        # Clean up temp file
        if os.path.exists(temp_path) and hasattr(img_file, 'save'):
            os.remove(temp_path)
        
        if embedding_objs and len(embedding_objs) > 0:
            print("Face embedding extracted successfully")
            return embedding_objs[0]["embedding"]
        
        print("No face embedding found")
        return None
        
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path) and hasattr(img_file, 'save'):
            os.remove(temp_path)
        return None

def update_face(imgName, addImg):
    """Add new face encoding to database"""
    try:
        embedding = get_face_embedding(addImg)
        
        if embedding is None:
            return False
        
        # Store in database
        pair = {imgName: embedding}
        dbname = get_database()
        collection_name = dbname["encodings"]
        collection_name.insert_one(pair)
        return True
        
    except Exception as e:
        print("Error while updating face:", e)
        return False

def compare_faces(baseImg, similarity_threshold=0.55):
    """Compare uploaded face with all stored faces using cosine similarity"""
    try:
        # Get embedding for uploaded image
        test_embedding = get_face_embedding(baseImg)
        
        if test_embedding is None:
            print("Failed to get embedding for uploaded image")
            return False
        
        # Get all known encodings
        known_images, encodings = getEncodings()
        
        print(f"Found {len(encodings)} stored faces:")
        for i, name in enumerate(known_images):
            print(f"  {i+1}. {name}")
        
        if not encodings:
            print("No known face encodings found in database")
            return False
        
        # Calculate cosine similarity with all known faces
        test_embedding = np.array(test_embedding)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
        
        max_similarity = 0
        matched_name = None
        
        print(f"Comparing against {len(encodings)} stored faces...")
        
        for i, known_encoding in enumerate(encodings):
            known_encoding = np.array(known_encoding)
            known_encoding = known_encoding / np.linalg.norm(known_encoding)  # Normalize
            
            # Calculate cosine similarity (higher is better)
            similarity = np.dot(test_embedding, known_encoding)
            
            print(f"Similarity with {known_images[i]}: {similarity:.3f}")
            
            if similarity > max_similarity:
                max_similarity = similarity
                matched_name = known_images[i]
        
        print(f"Best match: {matched_name} with similarity: {max_similarity:.3f}")
        print(f"Threshold: {similarity_threshold}")
        
        # Check if best match is above similarity threshold
        if max_similarity >= similarity_threshold:
            # Clean the name (remove file extension and ID prefixes)
            clean_name = matched_name.split(".")[0]
            if "_" in clean_name:
                # If it has employee_id_name format, extract just the name
                parts = clean_name.split("_")
                if len(parts) > 1:
                    clean_name = "_".join(parts[1:])  # Take everything after first underscore
            
            print(f"Face matched! Returning: {clean_name}")
            return clean_name
        
        print(f"No match found. Best similarity {max_similarity:.3f} below threshold {similarity_threshold}")
        return False
        
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False

# ------------------ ATTENDANCE FUNCTION ------------------
def update_attendance(id, status):
    """Record attendance in database"""
    IST = pytz.timezone('Asia/Kolkata')
    now = datetime.now(IST)
    momentDate = now.strftime("%d/%m/%Y")
    momentTime = now.strftime("%H:%M:%S")
    db = get_database()
    collection_name = db[momentDate]
    data = {"id": id, "status": status, "date": momentDate, "time": momentTime}
    try:
        collection_name.insert_one(data)
        return True
    except Exception as e:
        print("Error updating attendance:", e)
        return False

# ------------------ FLASK APP SETUP ------------------
app = Flask(__name__)
CORS(app)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = 'attendease-secret-key-change-in-production'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
jwt = JWTManager(app)

@app.route('/face_match', methods=['POST'])
def face_match():
    """Match uploaded face with stored faces and update attendance"""
    try:
        # Check for both 'file1' (old format) and 'image' (new format)
        file_key = 'image' if 'image' in request.files else 'file1'
        
        if file_key not in request.files:
            return jsonify({"success": False, "message": "No file provided"}), 400
            
        uploaded_file = request.files.get(file_key)
        employee_id = request.form.get('employee_id')
        name = request.form.get('name')
        
        # Match face
        print(f"=== FACE MATCH DEBUG ===")
        print(f"Uploaded file: {uploaded_file.filename}")
        print(f"File size: {len(uploaded_file.read())} bytes")
        uploaded_file.seek(0)  # Reset file pointer
        
        response = compare_faces(uploaded_file)
        print(f"Face comparison result: {response}")
        print(f"========================")
        
        if response:
            # Update attendance
            status = "Present"
            attendance_success = update_attendance(response, status)
            
            if attendance_success:
                return jsonify({
                    "success": True, 
                    "message": "Face matched and attendance marked successfully",
                    "employee_name": response
                })
            else:
                return jsonify({
                    "success": False, 
                    "message": "Face matched but failed to update attendance"
                })
        else:
            return jsonify({
                "success": False, 
                "message": "Face not recognized. Please try again."
            })
            
    except Exception as e:
        print(f"Error in face_match: {e}")
        return jsonify({"success": False, "message": "Face matching failed"}), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    """Add new face to database"""
    try:
        # Check for both 'file1' (old format) and 'image' (new format)
        file_key = 'image' if 'image' in request.files else 'file1'
        
        if file_key not in request.files:
            return jsonify({"success": False, "message": "No file provided"}), 400
            
        uploaded_file = request.files.get(file_key)
        employee_id = request.form.get('employee_id')
        name = request.form.get('name')
        
        # Use employee_id and name if provided, otherwise use filename
        if employee_id and name:
            imgName = f"{employee_id}_{name}"
        else:
            imgName = uploaded_file.filename.split(".")[0]
        
        response = update_face(imgName, uploaded_file)
        
        if response:
            return jsonify({
                "success": True, 
                "message": "Face registered successfully",
                "employee_name": imgName
            })
        else:
            return jsonify({
                "success": False, 
                "message": "Failed to register face. Please ensure face is clearly visible."
            })
            
    except Exception as e:
        print(f"Error in add_face: {e}")
        return jsonify({"success": False, "message": "Face registration failed"}), 500

# ------------------ AUTHENTICATION ENDPOINTS ------------------
@app.route('/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'password', 'role']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "message": f"{field} is required"}), 400
        
        # Check if user already exists
        db = get_database()
        users_collection = db["users"]
        
        if users_collection.find_one({"email": data['email']}):
            return jsonify({"success": False, "message": "Email already registered"}), 400
        
        # Hash password
        hashed_password = generate_password_hash(data['password'])
        
        # Create user document
        user_doc = {
            "name": data['name'],
            "email": data['email'],
            "password": hashed_password,
            "role": data['role'],
            "post": data.get('post', ''),
            "phone": data.get('phone', ''),
            "created_at": datetime.now().isoformat()
        }
        
        # Insert user
        result = users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
        # Create access token
        access_token = create_access_token(identity=user_id)
        
        # Return user data (without password)
        user_doc.pop('password')
        user_doc['id'] = user_id
        user_doc.pop('_id', None)
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "token": access_token,
            "user": user_doc
        }), 201
        
    except Exception as e:
        print(f"Error in signup: {e}")
        return jsonify({"success": False, "message": "Registration failed"}), 500

@app.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            return jsonify({"success": False, "message": "Email and password are required"}), 400
        
        # Find user
        db = get_database()
        users_collection = db["users"]
        user = users_collection.find_one({"email": data['email']})
        
        if not user:
            return jsonify({"success": False, "message": "Invalid email or password"}), 401
        
        # Check password
        if not check_password_hash(user['password'], data['password']):
            return jsonify({"success": False, "message": "Invalid email or password"}), 401
        
        # Check role if specified
        if data.get('role') and user['role'] != data['role']:
            return jsonify({"success": False, "message": f"Invalid credentials for {data['role']} login"}), 401
        
        # Create access token
        access_token = create_access_token(identity=str(user['_id']))
        
        # Return user data (without password)
        user.pop('password')
        user['id'] = str(user['_id'])
        user.pop('_id')
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "token": access_token,
            "user": user
        })
        
    except Exception as e:
        print(f"Error in login: {e}")
        return jsonify({"success": False, "message": "Login failed"}), 500

@app.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    """Verify JWT token"""
    try:
        user_id = get_jwt_identity()
        db = get_database()
        users_collection = db["users"]
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        user.pop('password')
        user['id'] = str(user['_id'])
        user.pop('_id')
        
        return jsonify({"success": True, "user": user})
        
    except Exception as e:
        print(f"Error in verify_token: {e}")
        return jsonify({"success": False, "message": "Token verification failed"}), 500

# ------------------ EMPLOYEE MANAGEMENT ENDPOINTS ------------------
@app.route('/employees', methods=['GET'])
@jwt_required()
def get_employees():
    """Get all employees"""
    try:
        db = get_database()
        users_collection = db["users"]
        employees = list(users_collection.find({"role": "employee"}))
        
        # Format employee data
        formatted_employees = []
        for emp in employees:
            emp.pop('password', None)
            emp['id'] = str(emp['_id'])
            emp.pop('_id')
            formatted_employees.append(emp)
        
        return jsonify({
            "success": True,
            "employees": formatted_employees
        })
        
    except Exception as e:
        print(f"Error in get_employees: {e}")
        return jsonify({"success": False, "message": "Failed to fetch employees"}), 500

@app.route('/employees', methods=['POST'])
@jwt_required()
def create_employee():
    """Create new employee (Admin only)"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "message": f"{field} is required"}), 400
        
        # Check if email already exists
        db = get_database()
        users_collection = db["users"]
        
        if users_collection.find_one({"email": data['email']}):
            return jsonify({"success": False, "message": "Email already registered"}), 400
        
        # Hash password
        hashed_password = generate_password_hash(data['password'])
        
        # Create employee document
        employee_doc = {
            "name": data['name'],
            "email": data['email'],
            "password": hashed_password,
            "role": "employee",
            "post": data.get('post', ''),
            "phone": data.get('phone', ''),
            "created_at": datetime.now().isoformat()
        }
        
        # Insert employee
        result = users_collection.insert_one(employee_doc)
        employee_id = str(result.inserted_id)
        
        # Return employee data (without password)
        employee_doc.pop('password')
        employee_doc['id'] = employee_id
        employee_doc.pop('_id', None)
        
        return jsonify({
            "success": True,
            "message": "Employee created successfully",
            "employee": employee_doc
        }), 201
        
    except Exception as e:
        print(f"Error in create_employee: {e}")
        return jsonify({"success": False, "message": "Failed to create employee"}), 500

@app.route('/employees/<employee_id>', methods=['PUT'])
@jwt_required()
def update_employee(employee_id):
    """Update employee information"""
    try:
        data = request.get_json()
        
        db = get_database()
        users_collection = db["users"]
        
        # Prepare update data
        update_data = {}
        allowed_fields = ['name', 'post', 'phone']
        
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
        
        if not update_data:
            return jsonify({"success": False, "message": "No valid fields to update"}), 400
        
        # Update employee
        result = users_collection.update_one(
            {"_id": ObjectId(employee_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            return jsonify({"success": False, "message": "Employee not found"}), 404
        
        # Get updated employee
        employee = users_collection.find_one({"_id": ObjectId(employee_id)})
        employee.pop('password', None)
        employee['id'] = str(employee['_id'])
        employee.pop('_id')
        
        return jsonify({
            "success": True,
            "message": "Employee updated successfully",
            "employee": employee
        })
        
    except Exception as e:
        print(f"Error in update_employee: {e}")
        return jsonify({"success": False, "message": "Failed to update employee"}), 500

@app.route('/employees/<employee_id>', methods=['DELETE'])
@jwt_required()
def delete_employee(employee_id):
    """Delete employee"""
    try:
        db = get_database()
        users_collection = db["users"]
        
        result = users_collection.delete_one({"_id": ObjectId(employee_id)})
        
        if result.deleted_count == 0:
            return jsonify({"success": False, "message": "Employee not found"}), 404
        
        return jsonify({
            "success": True,
            "message": "Employee deleted successfully"
        })
        
    except Exception as e:
        print(f"Error in delete_employee: {e}")
        return jsonify({"success": False, "message": "Failed to delete employee"}), 500

# ------------------ ATTENDANCE ENDPOINTS ------------------
@app.route('/attendance', methods=['GET'])
@jwt_required()
def get_attendance():
    """Get attendance records"""
    try:
        employee_id = request.args.get('employee_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        db = get_database()
        
        # Get all collection names (dates)
        collection_names = db.list_collection_names()
        attendance_collections = [name for name in collection_names if '/' in name]  # Date format: DD/MM/YYYY
        
        all_attendance = []
        
        for collection_name in attendance_collections:
            collection = db[collection_name]
            query = {}
            
            if employee_id:
                query['id'] = employee_id
            
            records = list(collection.find(query))
            for record in records:
                record.pop('_id', None)
                all_attendance.append(record)
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_attendance = []
            for record in all_attendance:
                record_date = datetime.strptime(record['date'], "%d/%m/%Y")
                
                if start_date:
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    if record_date < start:
                        continue
                
                if end_date:
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                    if record_date > end:
                        continue
                
                filtered_attendance.append(record)
            
            all_attendance = filtered_attendance
        
        return jsonify({
            "success": True,
            "attendance": all_attendance
        })
        
    except Exception as e:
        print(f"Error in get_attendance: {e}")
        return jsonify({"success": False, "message": "Failed to fetch attendance"}), 500

# ------------------ GEOFENCE ENDPOINTS ------------------
@app.route('/geofence', methods=['GET'])
@jwt_required()
def get_geofence():
    """Get geofence configuration"""
    try:
        db = get_database()
        geofence_collection = db["geofence"]
        geofence = geofence_collection.find_one()
        
        if not geofence:
            # Return default geofence if none exists
            default_geofence = {
                "latitude": 28.6139,
                "longitude": 77.2090,
                "radius": 100,
                "name": "Office Location"
            }
            return jsonify({
                "success": True,
                "geofence": default_geofence
            })
        
        geofence.pop('_id', None)
        return jsonify({
            "success": True,
            "geofence": geofence
        })
        
    except Exception as e:
        print(f"Error in get_geofence: {e}")
        return jsonify({"success": False, "message": "Failed to fetch geofence"}), 500

@app.route('/geofence', methods=['PUT'])
@jwt_required()
def update_geofence():
    """Update geofence configuration (Admin only)"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'radius']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "message": f"{field} is required"}), 400
        
        db = get_database()
        geofence_collection = db["geofence"]
        
        geofence_data = {
            "latitude": float(data['latitude']),
            "longitude": float(data['longitude']),
            "radius": float(data['radius']),
            "name": data.get('name', 'Office Location'),
            "updated_at": datetime.now().isoformat()
        }
        
        # Upsert geofence configuration
        geofence_collection.replace_one({}, geofence_data, upsert=True)
        
        return jsonify({
            "success": True,
            "message": "Geofence updated successfully",
            "geofence": geofence_data
        })
        
    except Exception as e:
        print(f"Error in update_geofence: {e}")
        return jsonify({"success": False, "message": "Failed to update geofence"}), 500

@app.route('/', methods=['GET'])
def home():
    return 'AttendEase APP API is Running Successfully! ✅'

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "OK", "timestamp": datetime.now().isoformat()})

# ------------------ PASSWORD CHANGE ENDPOINTS ------------------
@app.route('/employees/<employee_id>/password', methods=['PUT'])
@jwt_required()
def change_employee_password(employee_id):
    """Change employee password"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('currentPassword') or not data.get('newPassword'):
            return jsonify({"success": False, "message": "Current password and new password are required"}), 400
        
        db = get_database()
        users_collection = db["users"]
        
        # Find the employee
        employee = users_collection.find_one({"_id": ObjectId(employee_id)})
        
        if not employee:
            return jsonify({"success": False, "message": "Employee not found"}), 404
        
        # Verify current password
        if not check_password_hash(employee['password'], data['currentPassword']):
            return jsonify({"success": False, "message": "Current password is incorrect"}), 401
        
        # Hash new password
        new_hashed_password = generate_password_hash(data['newPassword'])
        
        # Update password
        result = users_collection.update_one(
            {"_id": ObjectId(employee_id)},
            {"$set": {"password": new_hashed_password}}
        )
        
        if result.matched_count == 0:
            return jsonify({"success": False, "message": "Employee not found"}), 404
        
        return jsonify({
            "success": True,
            "message": "Password changed successfully"
        })
        
    except Exception as e:
        print(f"Error in change_employee_password: {e}")
        return jsonify({"success": False, "message": "Failed to change password"}), 500

@app.route('/admin/<admin_id>/password', methods=['PUT'])
@jwt_required()
def change_admin_password(admin_id):
    """Change admin password"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('currentPassword') or not data.get('newPassword'):
            return jsonify({"success": False, "message": "Current password and new password are required"}), 400
        
        db = get_database()
        users_collection = db["users"]
        
        # Find the admin
        admin = users_collection.find_one({"_id": ObjectId(admin_id), "role": "admin"})
        
        if not admin:
            return jsonify({"success": False, "message": "Admin not found"}), 404
        
        # Verify current password
        if not check_password_hash(admin['password'], data['currentPassword']):
            return jsonify({"success": False, "message": "Current password is incorrect"}), 401
        
        # Hash new password
        new_hashed_password = generate_password_hash(data['newPassword'])
        
        # Update password
        result = users_collection.update_one(
            {"_id": ObjectId(admin_id), "role": "admin"},
            {"$set": {"password": new_hashed_password}}
        )
        
        if result.matched_count == 0:
            return jsonify({"success": False, "message": "Admin not found"}), 404
        
        return jsonify({
            "success": True,
            "message": "Password changed successfully"
        })
        
    except Exception as e:
        print(f"Error in change_admin_password: {e}")
        return jsonify({"success": False, "message": "Failed to change password"}), 500

# ------------------ MAIN ENTRY POINT ------------------
if __name__ == '__main__':
    port = int(os.env.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)