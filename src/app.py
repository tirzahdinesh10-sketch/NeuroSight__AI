from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import io
import logging
import numpy as np
from PIL import Image
import joblib

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parents[0] / 'static'))


@app.route('/')
def index():
	return send_from_directory(app.static_folder, 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
	return send_from_directory(app.static_folder, filename)

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'


def latest_file(prefix: str):
	files = sorted(MODELS_DIR.glob(f"{prefix}_*.joblib"))
	return files[-1] if files else None


def load_artifacts():
	clf_path = latest_file('classifier')
	pca_path = latest_file('ipca')
	if clf_path is None or pca_path is None:
		logging.error('Model or PCA not found in models/')
		return None, None
	clf = joblib.load(clf_path)
	pca = joblib.load(pca_path)
	logging.info(f'Loaded classifier: {clf_path.name}, pca: {pca_path.name}')
	return clf, pca


clf, pca = load_artifacts()


def preprocess_image_file(file_stream, size=150):
	img = Image.open(io.BytesIO(file_stream)).convert('RGB')
	img = img.resize((size, size), Image.LANCZOS)
	arr = np.asarray(img).astype(np.float32)
	if arr.max() > 1.0:
		arr = arr / 255.0
	return arr.ravel()


@app.route('/models/confusion')
def models_confusion():
	# locate confusion matrix image and latest eval metrics
	conf_img = MODELS_DIR / 'confusion_matrix_v1.png'
	# find latest eval_metrics file
	evals = sorted(MODELS_DIR.glob('eval_metrics_*.txt'))
	accuracy = None
	if evals:
		text = evals[-1].read_text()
		for line in text.splitlines():
			if line.lower().startswith('test accuracy:'):
				try:
					accuracy = float(line.split(':',1)[1].strip())
				except Exception:
					pass
				break

	return jsonify({'confusion_image': f'/models/confusion_matrix_v1.png', 'test_accuracy': accuracy})


@app.route('/models/<path:filename>')
def serve_models(filename):
	return send_from_directory(str(MODELS_DIR), filename)


@app.route('/health')
def health():
	return jsonify({'status': 'ok', 'model_loaded': clf is not None and pca is not None})


@app.route('/predict', methods=['POST'])
def predict():
	global clf, pca
	if clf is None or pca is None:
		clf, pca = load_artifacts()
		if clf is None:
			return jsonify({'error': 'Model not available'}), 500

	if 'file' not in request.files:
		return jsonify({'error': 'no file provided'}), 400

	f = request.files['file']
	data = f.read()
	try:
		x = preprocess_image_file(data)
	except Exception as e:
		return jsonify({'error': f'failed to process image: {e}'}), 400

	x_reduced = pca.transform([x])
	preds = clf.predict(x_reduced)
	probs = clf.predict_proba(x_reduced)[0].tolist() if hasattr(clf, 'predict_proba') else []
	classes = clf.classes_.tolist() if hasattr(clf, 'classes_') else []

	# build response
	response = {'prediction': preds[0], 'classes': classes, 'probabilities': probs}
	# also return tumor presence
	response['tumor_present'] = preds[0] != 'notumor'

	return jsonify(response)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=False)

