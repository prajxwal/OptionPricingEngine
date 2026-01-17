"""
Flask application factory for the Option Pricing Engine.
"""

from flask import Flask
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLASK_CONFIG


def create_app() -> Flask:
    """
    Application factory pattern.
    
    Creates and configures the Flask application.
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config['SECRET_KEY'] = FLASK_CONFIG.SECRET_KEY
    app.config['DEBUG'] = FLASK_CONFIG.DEBUG
    app.config['MAX_CONTENT_LENGTH'] = FLASK_CONFIG.MAX_CONTENT_LENGTH
    
    # Register blueprints
    from web.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app


# Create app instance for running directly
app = create_app()


if __name__ == '__main__':
    app.run(
        host=FLASK_CONFIG.HOST,
        port=FLASK_CONFIG.PORT,
        debug=FLASK_CONFIG.DEBUG,
    )
