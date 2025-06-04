import os
import json
import re
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from dotenv import load_dotenv
from functools import wraps
import requests

# Import your backend functions
from storyboard_generator import (
    initialize_database, text_enhancer, save_story_to_file,
    generate_audio_narration, generate_comic_images,
    create_storyboard_with_audio, save_to_mysql,
    get_story_history, get_story_details,
    extract_characters_from_story,
    generate_consistent_image_prompt,
    generate_comic_images_with_consistency,
    save_character_styles,
    get_character_styles,
    analyze_story_for_character_consistency,
    save_story_with_character_styles,
    # Recreate + update
    recreate_story_assets,
    update_story_data,
    # For fallback
    fallback_image_generation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database configuration
mysql_host = os.getenv("MYSQL_HOST", "localhost")
mysql_user = os.getenv("MYSQL_USER", "root")
mysql_password = os.getenv("MYSQL_PASSWORD", "")
mysql_db = os.getenv("MYSQL_DB", "story_database")

# OpenAI API key for direct image generation
openai_api_key = os.getenv("OPENAI_API_KEY")


# Helper function for database connections
def get_db_connection():
    return pymysql.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_db,
        cursorclass=pymysql.cursors.DictCursor
    )


# --------------------------
# Initialize Database
# --------------------------
def initialize_database_with_users():
    """Create database and tables if they don't exist."""
    try:
        conn = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password)
        cursor = conn.cursor()
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {mysql_db}")
        cursor.execute(f"USE {mysql_db}")

        # Create story_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS story_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                story_outline TEXT,
                enhanced_story TEXT,
                audio_urls JSON,
                image_urls JSON,
                pdf_url VARCHAR(255),
                pptx_url VARCHAR(255),
                user_id INT,
                character_styles JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Database initialization successful")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")


# Call the initializer
initialize_database_with_users()


# --------------------------
# Authentication Helpers
# --------------------------
def register_user(username, email, password):
    """Register a new user."""
    try:
        conn = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db)
        cursor = conn.cursor()

        # Check username
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return False, "Username already exists"

        # Check email
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return False, "Email already exists"

        # Hash password
        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                       (username, email, hashed_password))
        conn.commit()
        user_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return True, user_id
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return False, f"Database error: {str(e)}"


def authenticate_user(username, password):
    """Authenticate a user by username/password."""
    try:
        conn = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            return True, user
        return False, None
    except Exception as e:
        logging.error(f"Error authenticating user: {e}")
        return False, None


def get_user_by_id(user_id):
    """Retrieve user details by ID."""
    try:
        conn = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            return {'id': user_id, 'username': 'Unknown', 'email': 'unknown@example.com', 'created_at': None}
        return user
    except Exception as e:
        logging.error(f"Error getting user: {e}")
        return {'id': user_id, 'username': 'User', 'email': 'error@example.com', 'created_at': None}


# --------------------------
# Login Required Decorator
# --------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# --------------------------
# Sidebar Helpers
# --------------------------
def get_user_stories(user_id, limit=10):
    """Retrieve the most recent stories for a user."""
    try:
        conn = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db)
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # Check if user_id column exists
        cursor.execute("SHOW COLUMNS FROM story_data LIKE 'user_id'")
        if not cursor.fetchone():
            # If user_id doesn't exist, just get the most recent stories
            query = """
                SELECT id, title, SUBSTRING(story_outline, 1, 100) AS preview,
                       created_at, pdf_url, pptx_url
                FROM story_data
                ORDER BY created_at DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))
        else:
            # Filter by user_id
            query = """
                SELECT id, title, SUBSTRING(story_outline, 1, 100) AS preview,
                       created_at, pdf_url, pptx_url
                FROM story_data
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            cursor.execute(query, (user_id, limit))

        stories = cursor.fetchall()
        cursor.close()
        conn.close()
        return stories
    except Exception as e:
        logging.error(f"Error retrieving user story history: {e}")
        return []


def get_sidebar_stories(user_id=None):
    """Get stories for the sidebar (limit=10)."""
    if user_id:
        return get_user_stories(user_id, limit=10)
    return []


# --------------------------
# Saving Story to MySQL
# --------------------------
def save_to_mysql_with_user(title, original_story, enhanced_story, audio_files, image_files, pdf_path, pptx_path,
                            user_id):
    """Save a new story to MySQL with a user association."""
    logging.info("Saving story data to MySQL (with user).")
    audio_json = json.dumps(audio_files)
    image_json = json.dumps(image_files)

    # Fix here: Handle pdf_path and pptx_path correctly whether they're strings or tuples
    if isinstance(pdf_path, tuple):
        pdf_local = os.path.abspath(pdf_path[0]) if pdf_path[0] and os.path.exists(pdf_path[0]) else ""
    else:
        pdf_local = os.path.abspath(pdf_path) if pdf_path and os.path.exists(pdf_path) else ""

    if isinstance(pptx_path, tuple):
        pptx_local = os.path.abspath(pptx_path[1]) if len(pptx_path) > 1 and pptx_path[1] and os.path.exists(
            pptx_path[1]) else ""
    else:
        pptx_local = os.path.abspath(pptx_path) if pptx_path and os.path.exists(pptx_path) else ""

    try:
        conn = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db)
        cursor = conn.cursor()

        # Check if pptx_url column exists
        cursor.execute("SHOW COLUMNS FROM story_data LIKE 'pptx_url'")
        if not cursor.fetchone():
            # Add the column if it doesn't exist
            cursor.execute("ALTER TABLE story_data ADD COLUMN pptx_url VARCHAR(255)")
            logging.info("Added pptx_url column to story_data table")

        insert_query = """
            INSERT INTO story_data
            (title, story_outline, enhanced_story, audio_urls, image_urls, pdf_url, pptx_url, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query,
                       (title, original_story, enhanced_story,
                        audio_json, image_json, pdf_local, pptx_local, user_id))
        conn.commit()
        record_id = cursor.lastrowid
        logging.info(f"MySQL insert successful with ID: {record_id}")
        cursor.close()
        conn.close()
        return record_id
    except Exception as e:
        logging.error(f"Error inserting data into MySQL: {e}")
        return None


# --------------------------
# Flask Routes
# --------------------------
@app.route('/')
def index():
    user_id = session.get('user_id')
    stories = get_sidebar_stories(user_id)
    is_logged_in = 'user_id' in session
    username = session.get('username', '')
    return render_template('index.html', stories=stories, is_logged_in=is_logged_in, username=username)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please provide both username and password', 'error')
            return redirect(url_for('login'))

        success, user = authenticate_user(username, password)
        if success:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html', stories=[])


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))

        if not re.match(r'^[a-zA-Z0-9_]+', username):
            flash('Username can only contain letters, numbers, and underscores', 'error')
            return redirect(url_for('register'))

        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            flash('Please enter a valid email address', 'error')
            return redirect(url_for('register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
            return redirect(url_for('register'))

        success, result = register_user(username, email, password)
        if success:
            session['user_id'] = result
            session['username'] = username
            flash('Account created successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash(f'Registration failed: {result}', 'error')

    return render_template('register.html', stories=[])


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))


@app.route('/create', methods=['GET', 'POST'])
@login_required
def create_story():
    """Create a new story route with an optional checkbox to enhance the story."""
    user_id = session.get('user_id')
    stories = get_sidebar_stories(user_id)

    if request.method == 'POST':
        title = request.form.get('title')
        story_outline = request.form.get('story_outline')
        enhance = request.form.get('enhance') == 'on'
        character_styles_json = request.form.get('character_styles', '{}')
        try:
            character_styles = json.loads(character_styles_json)
        except:
            character_styles = {}

        # Consistency level
        consistency_level = float(request.form.get('consistency_level', 0.8))

        if not title or not story_outline:
            flash('Title and story outline are required!', 'error')
            return redirect(url_for('create_story'))

        # If "Enhance" checkbox is checked, run text_enhancer
        if enhance:
            story = text_enhancer(story_outline)
        else:
            story = story_outline

        # Save story to local files
        txt_path, json_path = save_story_to_file(story, title)

        # Generate audio & images
        audio_files, story_dir = generate_audio_narration(story, title)
        if character_styles and len(character_styles) > 0:
            image_files = generate_comic_images_with_consistency(
                story, title, story_dir, character_styles, consistency_level
            )
        else:
            image_files = generate_comic_images(story, title, story_dir)

        # Create PDF and PowerPoint
        pdf_path, pptx_path = create_storyboard_with_audio(image_files, audio_files, story, title, story_dir)

        # Save to database
        if character_styles and len(character_styles) > 0:
            record_id = save_story_with_character_styles(
                title, story_outline, story, audio_files, image_files,
                pdf_path, pptx_path, character_styles, user_id
            )
        else:
            record_id = save_to_mysql_with_user(
                title, story_outline, story, audio_files, image_files, pdf_path, pptx_path, user_id
            )

        if record_id:
            flash('Story created successfully!', 'success')
            return redirect(url_for('view_story', story_id=record_id))
        else:
            flash('Failed to create story. Please try again.', 'error')

    return render_template('create_story.html', stories=stories)


@app.route('/story/<int:story_id>')
@login_required
def view_story(story_id):
    """View a specific story route."""
    user_id = session.get('user_id')
    stories = get_sidebar_stories(user_id)
    story = get_story_details(story_id)

    if not story:
        flash('Story not found!', 'error')
        return redirect(url_for('index'))

    # Check if story belongs to current user
    if story.get('user_id') and story['user_id'] != user_id:
        flash('You do not have permission to view this story', 'error')
        return redirect(url_for('index'))

    return render_template('view_story.html', story=story, stories=stories, story_id=story_id)


@app.route('/story/<int:story_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_story(story_id):
    """Edit an existing story, optionally regenerating images and audio."""
    user_id = session.get('user_id')
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM story_data WHERE id = %s AND user_id = %s", (story_id, user_id))
            story = cursor.fetchone()
    except Exception as e:
        logging.error(f"Error retrieving story: {e}")
        flash('Failed to retrieve story details', 'error')
        return redirect(url_for('index'))
    finally:
        if conn:
            conn.close()

    if not story:
        flash('Story not found or you do not have permission to edit', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        new_title = request.form.get('title')
        new_story_outline = request.form.get('story_outline')
        enhance = request.form.get('enhance') == 'on'  # checkbox
        try:
            consistency_level = float(request.form.get('consistency_level', 80)) / 100.0
        except ValueError:
            consistency_level = 0.8

        if not new_title or not new_story_outline:
            flash('Title and story outline are required', 'error')
            # Pass stories for the sidebar
            stories = get_sidebar_stories(user_id)
            return render_template('edit_story.html', story=story, stories=stories)

        # If "Enhance" is checked, run text_enhancer
        if enhance:
            updated_story = text_enhancer(new_story_outline)
        else:
            updated_story = new_story_outline

        # Use existing character_styles from DB (if any)
        # story is a dict from DB, so story['character_styles'] might be JSON or None
        character_styles = story.get('character_styles') or {}

        # 1) Recreate images, audio, PDF, PowerPoint
        assets = recreate_story_assets(
            story_id,
            updated_story,
            new_title,
            character_styles=character_styles,
            consistency_level=consistency_level
        )
        if not assets:
            flash('Failed to generate new assets for the updated story', 'error')
            # Repopulate the sidebar
            stories = get_sidebar_stories(user_id)
            return render_template('edit_story.html', story=story, stories=stories)

        # 2) Update the story in the DB
        success = update_story_data(
            story_id,
            new_title,
            new_story_outline,  # Original outline
            updated_story,  # Enhanced (or not) story
            assets['audio_files'],
            assets['image_files'],
            assets['pdf_path'],
            assets['pptx_path'],
            character_styles=character_styles,
            user_id=user_id
        )
        if success:
            flash('Story updated and assets regenerated successfully', 'success')
            return redirect(url_for('view_story', story_id=story_id))
        else:
            flash('Failed to update story', 'error')

    # On GET, pass stories to the template
    stories = get_sidebar_stories(user_id)
    return render_template('edit_story.html', story=story, stories=stories)


@app.route('/storyboards')
@login_required
def storyboards():
    """List all user storyboards."""
    user_id = session.get('user_id')
    stories = get_user_stories(user_id, limit=20)
    return render_template('storyboards.html', stories=stories)


@app.route('/profile')
@login_required
def profile():
    """User profile page."""
    user_id = session.get('user_id')
    user = get_user_by_id(user_id)
    stories = get_sidebar_stories(user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('index'))

    # Count the user's stories
    all_stories = get_user_stories(user_id, limit=1000)
    story_count = len(all_stories)

    return render_template('profile.html', user=user, stories=stories, story_count=story_count)


@app.route('/assets/<path:filename>')
@login_required
def serve_asset(filename):
    """Serve images, audio, and PDFs from local filesystem."""
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))


# --------------------------
# API Routes
# --------------------------
@app.route('/api/story/<int:story_id>')
@login_required
def get_story_api(story_id):
    """API route to get story details as JSON."""
    story = get_story_details(story_id)
    if not story:
        return jsonify({"error": "Story not found"}), 404

    if story.get('user_id') and story['user_id'] != session.get('user_id'):
        return jsonify({"error": "Unauthorized"}), 403

    # Convert datetime to string
    if story.get('created_at'):
        story['created_at'] = story['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(story)


@app.route('/api/generate-panel', methods=['POST'])
@login_required
def generate_panel():
    """Generate a single panel image for a given scene using OpenAI's gpt-image-1 model."""
    data = request.json
    scene_text = data.get('scene_text')
    panel_number = data.get('panel_number', 1)

    if not scene_text:
        return jsonify({"error": "Scene text is required"}), 400

    try:
        # Create a unique directory for this panel
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, f"panel_{panel_number}.png")

        # Check if OpenAI is properly configured
        if not openai_api_key:
            # Use fallback image generation if OpenAI not available
            if fallback_image_generation(f"Black and white comic style art of the following scene: {scene_text}",
                                         image_path):
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Used fallback image generation"
                })
            else:
                return jsonify({"error": "Failed to generate fallback image"}), 500

        # Use OpenAI API directly for image generation
        import openai
        client = openai.OpenAI(api_key=openai_api_key)

        response = client.images.generate(
            model="gpt-image-1",
            prompt=f"Black and white comic style art of the following scene: {scene_text}",
            size="1024x1024",
            quality="high",
            n=1,
        )

        # Log the response for debugging
        logging.info(f"OpenAI API response structure: {response}")

        # Get the image URL - handle different possible response formats
        image_url = None
        if hasattr(response, 'data') and response.data and len(response.data) > 0:
            data_item = response.data[0]

            # Try to get URL from different possible attributes
            if hasattr(data_item, 'url') and data_item.url:
                image_url = data_item.url
            elif hasattr(data_item, 'image_url') and data_item.image_url:
                image_url = data_item.image_url
            elif hasattr(data_item, 'b64_json') and data_item.b64_json:
                # Handle base64 encoded image if that's returned
                import base64
                img_data = base64.b64decode(data_item.b64_json)
                with open(image_path, 'wb') as img_file:
                    img_file.write(img_data)
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Generated from base64 data"
                })

        if not image_url:
            logging.error("No image URL found in the API response")
            # Use fallback if URL not available
            if fallback_image_generation(f"Black and white comic style art of the following scene: {scene_text}",
                                         image_path):
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Used fallback image generation (no URL in response)"
                })
            else:
                return jsonify({"error": "Failed to generate fallback image"}), 500

        # If we have a URL, download the image
        image_response = requests.get(image_url)

        if image_response.status_code == 200:
            with open(image_path, 'wb') as img_file:
                img_file.write(image_response.content)

            return jsonify({
                "success": True,
                "image_url": url_for('serve_asset', filename=image_path),
                "panel_number": panel_number
            })
        else:
            return jsonify({"error": f"Failed to download image: HTTP {image_response.status_code}"}), 500

    except Exception as e:
        logging.error(f"Error generating panel: {e}")
        # Try fallback if OpenAI fails
        try:
            if fallback_image_generation(f"Black and white comic style art of the following scene: {scene_text}",
                                         image_path):
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Used fallback image generation after error"
                })
        except Exception as fallback_error:
            logging.error(f"Fallback also failed: {fallback_error}")

        return jsonify({"error": str(e)}), 500


@app.route('/api/extract-characters', methods=['POST'])
@login_required
def extract_characters():
    """Extract characters from story text (OpenAI)."""
    data = request.json
    story_text = data.get('story_text')

    if not story_text:
        return jsonify({"error": "Story text is required"}), 400

    try:
        characters = extract_characters_from_story(story_text)
        return jsonify({"success": True, "characters": characters})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-consistent-panel', methods=['POST'])
@login_required
def generate_consistent_panel():
    """Generate a style-consistent panel image for a given scene using OpenAI's gpt-image-1 model."""
    data = request.json
    scene_text = data.get('scene_text')
    panel_number = data.get('panel_number', 1)
    character_styles = data.get('character_styles', {})
    consistency_level = float(data.get('consistency_level', 0.8))

    if not scene_text:
        return jsonify({"error": "Scene text is required"}), 400

    try:
        # Create a unique directory for this panel
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, f"panel_{panel_number}.png")

        # Generate a consistent prompt
        enhanced_prompt = generate_consistent_image_prompt(scene_text, character_styles, consistency_level)

        # Check if OpenAI is properly configured
        if not openai_api_key:
            # Use fallback image generation if OpenAI not available
            if fallback_image_generation(enhanced_prompt, image_path):
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Used fallback image generation"
                })
            else:
                return jsonify({"error": "Failed to generate fallback image"}), 500

        # Use OpenAI API directly for image generation
        import openai
        client = openai.OpenAI(api_key=openai_api_key)

        response = client.images.generate(
            model="gpt-image-1",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="high",
            n=1,
        )

        # Log the response for debugging
        logging.info(f"OpenAI API response structure: {response}")

        # Get the image URL - handle different possible response formats
        image_url = None
        if hasattr(response, 'data') and response.data and len(response.data) > 0:
            data_item = response.data[0]

            # Try to get URL from different possible attributes
            if hasattr(data_item, 'url') and data_item.url:
                image_url = data_item.url
            elif hasattr(data_item, 'image_url') and data_item.image_url:
                image_url = data_item.image_url
            elif hasattr(data_item, 'b64_json') and data_item.b64_json:
                # Handle base64 encoded image if that's returned
                import base64
                img_data = base64.b64decode(data_item.b64_json)
                with open(image_path, 'wb') as img_file:
                    img_file.write(img_data)
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Generated from base64 data"
                })

        if not image_url:
            logging.error("No image URL found in the API response")
            # Use fallback if URL not available
            if fallback_image_generation(enhanced_prompt, image_path):
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Used fallback image generation (no URL in response)"
                })
            else:
                return jsonify({"error": "Failed to generate fallback image"}), 500

        # If we have a URL, download the image
        image_response = requests.get(image_url)

        if image_response.status_code == 200:
            with open(image_path, 'wb') as img_file:
                img_file.write(image_response.content)

            return jsonify({
                "success": True,
                "image_url": url_for('serve_asset', filename=image_path),
                "panel_number": panel_number
            })
        else:
            return jsonify({"error": f"Failed to download image: HTTP {image_response.status_code}"}), 500

    except Exception as e:
        logging.error(f"Error generating consistent panel: {e}")
        # Try fallback if OpenAI fails
        try:
            if fallback_image_generation(enhanced_prompt, image_path):
                return jsonify({
                    "success": True,
                    "image_url": url_for('serve_asset', filename=image_path),
                    "panel_number": panel_number,
                    "note": "Used fallback image generation after error"
                })
        except Exception as fallback_error:
            logging.error(f"Fallback also failed: {fallback_error}")

        return jsonify({"error": str(e)}), 500


@app.route('/api/save-character-styles', methods=['POST'])
@login_required
def save_character_styles_api():
    """Save character styles for a story."""
    data = request.json
    story_id = data.get('story_id')
    character_styles = data.get('character_styles', {})

    if not story_id:
        return jsonify({"error": "Story ID is required"}), 400

    try:
        success = save_character_styles(story_id, character_styles)
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to save character styles"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/character-consistency-analysis', methods=['POST'])
@login_required
def analyze_character_consistency():
    """Analyze a story for character consistency issues."""
    data = request.json
    story_text = data.get('story_text')
    character_styles = data.get('character_styles')

    if not story_text:
        return jsonify({"error": "Story text is required"}), 400

    try:
        analysis = analyze_story_for_character_consistency(story_text, character_styles)
        return jsonify({"success": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/update-story', methods=['POST'])
@login_required
def update_story_route():
    """API route to update story fields (used by JavaScript/AJAX)."""
    user_id = session.get('user_id')
    data = request.json

    story_id = data.get('story_id')
    title = data.get('title')
    story_outline = data.get('story_outline')
    enhanced_story = data.get('enhanced_story')
    audio_files = data.get('audio_files', [])
    image_files = data.get('image_files', [])
    pdf_path = data.get('pdf_path', '')
    pptx_path = data.get('pptx_path', '')
    character_styles = data.get('character_styles', None)

    if not story_id or not title:
        return jsonify({"error": "Story ID and title are required"}), 400

    try:
        success = update_story_data(
            story_id,
            title,
            story_outline,
            enhanced_story,
            audio_files,
            image_files,
            pdf_path,
            pptx_path,
            character_styles=character_styles,
            user_id=user_id
        )
        if success:
            return jsonify({"success": True, "message": "Story updated successfully"})
        else:
            return jsonify({"error": "Failed to update story"}), 500
    except Exception as e:
        logging.error(f"Error in update_story_route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/regenerate-story-assets', methods=['POST'])
@login_required
def regenerate_story_assets_route():
    """API route to regenerate story assets (images, audio, PDF)."""
    user_id = session.get('user_id')
    data = request.json

    story_id = data.get('story_id')
    story_text = data.get('story_text')
    title = data.get('title')
    character_styles = data.get('character_styles', None)
    consistency_level = float(data.get('consistency_level', 0.8))

    if not story_id or not story_text or not title:
        return jsonify({"error": "Story ID, story text, and title are required"}), 400

    try:
        assets = recreate_story_assets(
            story_id,
            story_text,
            title,
            character_styles=character_styles,
            consistency_level=consistency_level
        )
        if assets:
            success = update_story_data(
                story_id,
                title,
                None,
                None,
                assets['audio_files'],
                assets['image_files'],
                assets['pdf_path'],
                assets['pptx_path'],
                character_styles=character_styles,
                user_id=user_id
            )
            if success:
                return jsonify({
                    "success": True,
                    "message": "Story assets regenerated successfully",
                    "assets": {
                        "audio_files": assets['audio_files'],
                        "image_files": assets['image_files'],
                        "pdf_path": assets['pdf_path'],
                        "pptx_path": assets['pptx_path']
                    }
                })
            else:
                return jsonify({"error": "Failed to update story with new assets"}), 500
        return jsonify({"error": "Failed to regenerate assets"}), 500
    except Exception as e:
        logging.error(f"Error in regenerate_story_assets_route: {e}")
        return jsonify({"error": str(e)}), 500


# --------------------------
# Error Handlers
# --------------------------
@app.errorhandler(404)
def page_not_found(e):
    user_id = session.get('user_id')
    stories = get_sidebar_stories(user_id)
    return render_template('404.html', stories=stories), 404


@app.errorhandler(500)
def internal_server_error(e):
    user_id = session.get('user_id')
    stories = get_sidebar_stories(user_id)
    return render_template('500.html', stories=stories), 500


# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
    )
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('storyboard_assets', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('FLASK_DEBUG', 'False') == 'True')