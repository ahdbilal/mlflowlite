"""
Gateway UI Server
Serves the Gateway management UI alongside MLflow
"""
import os
from flask import Flask, send_from_directory, redirect

app = Flask(__name__)

# Optional CORS support
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass  # CORS not needed for local development

# Get the directory where this script is located
GATEWAY_UI_DIR = os.path.join(os.path.dirname(__file__), 'gateway')


@app.route('/gateway')
@app.route('/gateway/')
def gateway_home():
    """Serve the Gateway UI"""
    return send_from_directory(GATEWAY_UI_DIR, 'index_enhanced.html')


@app.route('/gateway/<path:filename>')
def gateway_static(filename):
    """Serve Gateway UI static files"""
    return send_from_directory(GATEWAY_UI_DIR, filename)


@app.route('/')
def root():
    """Redirect to Gateway UI"""
    return redirect('/gateway')


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 MLflow Gateway UI Server")
    print("="*70)
    print(f"\n📂 Serving from: {GATEWAY_UI_DIR}")
    print(f"\n🌐 Gateway UI: http://localhost:5001/gateway")
    print(f"   (Open this in your browser)")
    print("\n💡 Tip: Run MLflow on port 5000 and Gateway UI on 5001")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)

