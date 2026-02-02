import cv2
import numpy as np
import time
import os
import traceback
import urllib.request

# Optional: pygame for audio playback
PYGAME_AVAILABLE = False
try:
	import pygame
	pygame.mixer.init()
	PYGAME_AVAILABLE = True
except Exception:
	PYGAME_AVAILABLE = False

# Optional: Pillow for drawing Unicode text (Traditional Chinese checklist labels)
PIL_AVAILABLE = False
try:
	from PIL import Image, ImageDraw, ImageFont
	PIL_AVAILABLE = True
except Exception:
	PIL_AVAILABLE = False

# Try to import MediaPipe for more precise facial landmark-based smile metric
MP_AVAILABLE = False
MP_HANDS_AVAILABLE = False
mp_face_mesh = None
try:
	import mediapipe as mp
	MP_AVAILABLE = True
	mp_face_mesh = mp.solutions.face_mesh
except Exception as e:
	print('MediaPipe face_mesh import failed:')
	traceback.print_exc()

# Try to import MediaPipe Tasks API for hand detection (newer API)
try:
	from mediapipe.tasks import python as mp_tasks
	from mediapipe.tasks.python import vision as mp_vision
	MP_HANDS_AVAILABLE = True
except Exception as e:
	print('MediaPipe Tasks API import failed:')
	traceback.print_exc()

# Facemark (LBF) availability check (requires opencv-contrib)
FACEMARK_AVAILABLE = False
try:
	FACEMARK_AVAILABLE = hasattr(cv2, 'face') and hasattr(cv2.face, 'createFacemarkLBF')
except Exception:
	FACEMARK_AVAILABLE = False


def ensure_thanks_image(path, size=(640, 480)):
	if os.path.exists(path):
		return
	w, h = size
	img = np.zeros((h, w, 3), dtype=np.uint8)
	img[:] = (30, 120, 200)
	text = 'Thanks!'
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 3.0
	thickness = 6
	(text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
	x = (w - text_w) // 2
	y = (h + text_h) // 2
	cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
	cv2.imwrite(path, img)


def overlay_image_alpha(img, overlay, x, y):
	"""Overlay `overlay` onto `img` at position (x, y).
	`overlay` may have 3 or 4 channels (BGRA)."""
	h, w = overlay.shape[:2]
	if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
		return
	# Clip overlay if it goes outside the background image
	if x + w > img.shape[1]:
		w = img.shape[1] - x
		overlay = overlay[:, :w]
	if y + h > img.shape[0]:
		h = img.shape[0] - y
		overlay = overlay[:h, :]

	if overlay.shape[2] == 4:
		alpha = overlay[:, :, 3] / 255.0
		for c in range(3):
			img[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] + (1 - alpha) * img[y:y+h, x:x+w, c])
	else:
		img[y:y+h, x:x+w] = overlay


def main():
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

	if face_cascade.empty() or smile_cascade.empty():
		print('Error loading Haar cascades. Make sure OpenCV is installed correctly.')
		return

	project_dir = os.path.dirname(__file__)
	thanks_path = os.path.join(project_dir, 'thanks.jpg')
	ensure_thanks_image(thanks_path)

	lbf_model_path = os.path.join(project_dir, 'lbfmodel.yaml')

	def ensure_lbf_model(path):
		if os.path.exists(path):
			return True
		# Try to download authoritative LBF model (kurnianggoro/GSOC2017)
		url = 'https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml'
		try:
			print('Downloading LBF model...')
			urllib.request.urlretrieve(url, path)
			print('LBF model saved to', path)
			return True
		except Exception as e:
			print('Failed to download LBF model:', e)
			return False

	# Hand landmarker model path
	hand_model_path = os.path.join(project_dir, 'hand_landmarker.task')
	def ensure_hand_model(path):
		if os.path.exists(path):
			return True
		# Download MediaPipe hand landmarker model
		url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
		try:
			print('Downloading hand landmarker model...')
			urllib.request.urlretrieve(url, path)
			print('Hand landmarker model saved to', path)
			return True
		except Exception as e:
			print('Failed to download hand landmarker model:', e)
			return False

	# Initialize facemark if available
	facemark = None
	facemark_available = False
	if FACEMARK_AVAILABLE:
		if ensure_lbf_model(lbf_model_path):
			try:
				facemark = cv2.face.createFacemarkLBF()
				facemark.loadModel(lbf_model_path)
				facemark_available = True
			except Exception as e:
				print('Failed to initialize Facemark LBF:', e)
				facemark_available = False
	else:
		facemark_available = False

	surprised_path = os.path.join(project_dir, 'surprised.jpg')
	def ensure_surprised_image(path, size=(640, 480)):
		if os.path.exists(path):
			return
		w, h = size
		img = np.zeros((h, w, 3), dtype=np.uint8)
		img[:] = (80, 40, 200)
		text = 'Surprised!'
		font = cv2.FONT_HERSHEY_SIMPLEX
		scale = 2.5
		thickness = 6
		(text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
		x = (w - text_w) // 2
		y = (h + text_h) // 2
		cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
		cv2.imwrite(path, img)

	ensure_surprised_image(surprised_path)

	# Stunned overlay (wide eyes) - user-provided file
	stunned_path = os.path.join(project_dir, 'stunned.jpg')

	# Flat overlay (neutral) - user-provided file
	flat_path = os.path.join(project_dir, 'flat.jpg')

	# Cute overlay (smile + hand on mouth)
	cute_path = os.path.join(project_dir, 'cute.jpg')

	# 67 meme gesture overlay and audio
	pose67_path = os.path.join(project_dir, '67.webp')
	pose67_audio_path = os.path.join(project_dir, '67.mp3')
	pose67_audio_playing = False  # Track if audio is currently playing
	pose67_was_oscillating = False  # Track previous oscillation state to detect rising edge
	def ensure_cute_image(path, size=(640, 480)):
		if os.path.exists(path):
			return
		w, h = size
		img = np.zeros((h, w, 3), dtype=np.uint8)
		img[:] = (255, 180, 200)  # pink background
		text = 'Cute!'
		font = cv2.FONT_HERSHEY_SIMPLEX
		scale = 3.0
		thickness = 6
		(text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
		x = (w - text_w) // 2
		y = (h + text_h) // 2
		cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
		cv2.imwrite(path, img)

	ensure_cute_image(cute_path)

	# Try to load with alpha if available
	thanks_img = cv2.imread(thanks_path, cv2.IMREAD_UNCHANGED)
	if thanks_img is None:
		print('Failed to load thanks image; overlay will be disabled.')

	show_debug = False  # set True to render landmarks/text

	# Capture and display preferences
	desired_frame_width = 1280
	desired_frame_height = 720
	desired_fps = 60  # Set to 0 to keep driver default
	display_window_width = 1280
	display_window_height = 720
	overlay_width_ratio = 0.35  # Fraction of frame width used for overlays
	overlay_min_width = 80      # Minimum overlay width in pixels

	# Checklist tracking for each expression overlay
	checklist_enabled = True
	checklist_items = {
		'stunned': False,
		'cute': False,
		'surprised': False,
		'smile': False,
		'flat': False,
	}
	checklist_font_path = None  # Optional: set to a specific TTF/OTF for Chinese (e.g., C:\\Windows\\Fonts\\msjh.ttc)
	checklist_font_size = 22
	checklist_font = None
	checklist_icon_size = (64, 48)
	checklist_icon_pad = 10
	checklist_header_zh = '你能模仿這些表情嗎？'
	checklist_header_en = 'Can you imitate these emotions?'
	checklist_icons = {}

	def load_checklist_font():
		nonlocal checklist_font
		if not PIL_AVAILABLE:
			return
		candidates = []
		if checklist_font_path:
			candidates.append(checklist_font_path)
		# Common Traditional Chinese fonts on Windows
		win_dir = os.environ.get('WINDIR', 'C:\\Windows')
		candidates.extend([
			os.path.join(win_dir, 'Fonts', 'msjh.ttc'),      # Microsoft JhengHei
			os.path.join(win_dir, 'Fonts', 'msjhbd.ttc'),    # Microsoft JhengHei Bold
			os.path.join(win_dir, 'Fonts', 'mingliu.ttc'),   # MingLiU
			os.path.join(win_dir, 'Fonts', 'mingliub.ttc'),  # MingLiU Bold
		])
		for fp in candidates:
			if fp and os.path.exists(fp):
				try:
					checklist_font = ImageFont.truetype(fp, checklist_font_size)
					return
				except Exception:
					continue

	def load_checklist_icons():
		nonlocal checklist_icons
		if not PIL_AVAILABLE:
			return
		paths = {
			'stunned': stunned_path,
			'cute': cute_path,
			'surprised': surprised_path,
			'smile': thanks_path,
			'flat': flat_path,
		}
		for key, path in paths.items():
			if key in checklist_icons:
				continue
			if path and os.path.exists(path):
				img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
				if img_cv is None:
					continue
				if img_cv.shape[2] == 3:
					a = np.full((img_cv.shape[0], img_cv.shape[1], 1), 255, dtype=img_cv.dtype)
					img_cv = np.concatenate([img_cv, a], axis=2)
				img_rgba = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
				img_pil = Image.fromarray(img_rgba)
				img_pil = img_pil.resize(checklist_icon_size, Image.LANCZOS)
				checklist_icons[key] = img_pil

	def draw_checklist(frame, status_dict):
		if not checklist_enabled:
			return
		labels_zh = [
			('驚呆', status_dict['stunned'], 'stunned'),
			('可愛', status_dict['cute'], 'cute'),
			('驚訝', status_dict['surprised'], 'surprised'),
			('微笑', status_dict['smile'], 'smile'),
			('平淡', status_dict['flat'], 'flat'),
		]
		labels_fallback = [
			('Stunned', status_dict['stunned']),
			('Cute', status_dict['cute']),
			('Surprised', status_dict['surprised']),
			('Smile', status_dict['smile']),
			('Flat', status_dict['flat']),
		]

		# Prefer Pillow for proper Unicode rendering and icons
		if PIL_AVAILABLE:
			if checklist_font is None:
				load_checklist_font()
			if checklist_font is not None:
				load_checklist_icons()
				texts = [f"[{'X' if done else ' '}] {label}" for (label, done, _) in labels_zh]
				header = checklist_header_zh
				dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
				draw = ImageDraw.Draw(dummy_img)
				line_metrics = []
				max_text_w = 0
				max_line_h = 0
				for idx, txt in enumerate(texts):
					bbox = draw.textbbox((0, 0), txt, font=checklist_font)
					w = bbox[2] - bbox[0]
					h = bbox[3] - bbox[1]
					max_text_w = max(max_text_w, w)
					icon_key = labels_zh[idx][2]
					icon = checklist_icons.get(icon_key)
					icon_w, icon_h = (checklist_icon_size if icon else (0, 0))
					line_h = max(h, icon_h) + 6
					line_metrics.append((w, h, icon_w, icon_h, line_h))
				header_bbox = draw.textbbox((0, 0), header, font=checklist_font)
				header_w = header_bbox[2] - header_bbox[0]
				header_h = header_bbox[3] - header_bbox[1]
				height_total = header_h + 10 + sum(m[4] for m in line_metrics) + 14
				width_lines = max_text_w + (checklist_icon_size[0] + checklist_icon_pad if any(checklist_icons.values()) else 0)
				width_total = max(header_w, width_lines) + 24
				img = Image.new('RGBA', (max(width_total + 12, 180), max(height_total, 120)), (0, 0, 0, 0))
				draw = ImageDraw.Draw(img)
				draw.rectangle((0, 0, img.width, img.height), fill=(0, 0, 0, 200), outline=(80, 80, 80, 230), width=1)
				y = 10
				draw.text((10, y), header, font=checklist_font, fill=(255, 220, 120, 255))
				y += header_h + 6
				for i, txt in enumerate(texts):
					icon_key = labels_zh[i][2]
					icon = checklist_icons.get(icon_key)
					icon_w, icon_h, line_h = line_metrics[i][2], line_metrics[i][3], line_metrics[i][4]
					x_cursor = 10
					if icon is not None:
						img.paste(icon, (x_cursor, y + (line_h - icon_h) // 2), icon)
						x_cursor += checklist_icon_size[0] + checklist_icon_pad
					color = (0, 255, 0, 255) if labels_zh[i][1] else (160, 160, 160, 255)
					draw.text((x_cursor, y + (line_h - line_metrics[i][1]) // 2), txt, font=checklist_font, fill=color)
					y += line_h
				overlay_rgba = np.array(img)
				overlay_bgra = cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA)
				overlay_image_alpha(frame, overlay_bgra, 10, 10)
				return

		# Fallback to ASCII labels with OpenCV if no Unicode font
		items = labels_fallback
		x0, y0 = 10, 10
		line_h = 24
		font = cv2.FONT_HERSHEY_SIMPLEX
		scale = 0.65
		thickness = 2
		header = checklist_header_en
		header_size = cv2.getTextSize(header, font, scale, thickness)[0]
		texts = [f"[{'X' if done else ' '}] {label}" for (label, done) in items]
		max_w = header_size[0]
		for txt in texts:
			size = cv2.getTextSize(txt, font, scale, thickness)[0]
			max_w = max(max_w, size[0])
		height = header_size[1] + 8 + line_h * len(texts) + 14
		width = max_w + 20
		cv2.rectangle(frame, (x0 - 8, y0 - 8), (x0 + width, y0 + height), (0, 0, 0), -1)
		cv2.rectangle(frame, (x0 - 8, y0 - 8), (x0 + width, y0 + height), (80, 80, 80), 1)
		cv2.putText(frame, header, (x0, y0 + header_size[1]), font, scale, (255, 220, 120), thickness, cv2.LINE_AA)
		y_cursor = y0 + header_size[1] + 10
		for i, txt in enumerate(texts):
			color = (0, 255, 0) if items[i][1] else (160, 160, 160)
			cv2.putText(frame, txt, (x0, y_cursor + (i + 1) * line_h - 6), font, scale, color, thickness, cv2.LINE_AA)

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print('Cannot open webcam')
		return

	# Apply capture preferences (drivers may clamp these)
	if desired_frame_width > 0:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_frame_width)
	if desired_frame_height > 0:
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_frame_height)
	if desired_fps > 0:
		cap.set(cv2.CAP_PROP_FPS, desired_fps)

	actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	actual_fps = cap.get(cv2.CAP_PROP_FPS)

	# Allow resizing the display window without affecting capture resolution
	cv2.namedWindow('MyGO Smiler', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('MyGO Smiler', int(display_window_width), int(display_window_height))

	# Parameters for smoothing and hysteresis (smile uses mouth metric)
	smoothed_score = 0.0
	alpha = 0.35  # EMA smoothing factor
	on_threshold = 0.40
	off_threshold = 0.35
	on_frames_required = 2
	off_frames_required = 2
	on_count = 0
	off_count = 0
	overlay_visible = False

	# Mouth-open (surprised) detection params
	mouth_smoothed = 0.0
	mouth_alpha = 0.35
	mouth_on_threshold = 0.50
	mouth_off_threshold = 0.40
	mouth_on_frames_required = 2
	mouth_off_frames_required = 2
	mouth_on_count = 0
	mouth_off_count = 0
	mouth_overlay_visible = False

	# Eye-open (stunned) detection params
	eye_smoothed = 0.0
	eye_alpha = 0.35
	eye_on_threshold = 0.30
	eye_off_threshold = 0.29
	eye_on_frames_required = 2
	eye_off_frames_required = 2
	eye_on_count = 0
	eye_off_count = 0
	eye_overlay_visible = False

	# Flat (neutral) detection params
	flat_on_frames_required = 4
	flat_off_frames_required = 2
	flat_on_count = 0
	flat_off_count = 0
	flat_overlay_visible = False

	# Cute (hand on mouth while smiling) detection params
	cute_on_frames_required = 2
	cute_off_frames_required = 2
	cute_on_count = 0
	cute_off_count = 0
	cute_overlay_visible = False
	hand_near_mouth = False
	pose67_detected = False
	pose67_motion_detected = False
	pose67_top_hist = []
	pose67_bottom_hist = []
	pose67_hist_max = 20
	# Track pose67_motion_detected history to detect oscillation (continuous up/down motion)
	pose67_motion_hist = []
	pose67_motion_hist_max = 15  # how many frames to track
	pose67_oscillating = False
	pose67_min_transitions = 3  # minimum transitions (0->1 or 1->0) to consider as oscillating
	pose67_enabled = True  # can be toggled via key

	# Landmark smoothing (to stabilize mouth dots)
	prev_mouth_pts = None
	landmark_smooth_alpha = 0.55

	# Initialize MediaPipe face mesh if available
	mp_face = None
	hand_landmarker = None
	if MP_AVAILABLE and mp_face_mesh is not None:
		mp_face = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
										min_detection_confidence=0.5, min_tracking_confidence=0.5)
	else:
		print('MediaPipe face mesh not available: mouth-open (surprised) detection will be disabled.')

	# Initialize MediaPipe Hand Landmarker (new Tasks API)
	if MP_HANDS_AVAILABLE and ensure_hand_model(hand_model_path):
		try:
			base_options = mp_tasks.BaseOptions(model_asset_path=hand_model_path)
			options = mp_vision.HandLandmarkerOptions(
				base_options=base_options,
				running_mode=mp_vision.RunningMode.VIDEO,
				num_hands=2,
				min_hand_detection_confidence=0.5,
				min_hand_presence_confidence=0.5,
				min_tracking_confidence=0.5
			)
			hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
			print('Hand landmarker initialized successfully!')
		except Exception as e:
			print('Failed to initialize hand landmarker:', e)
			traceback.print_exc()
			hand_landmarker = None
	else:
		print('MediaPipe Hand Landmarker not available: cute detection will be disabled.')

	# Startup status
	print(f'Facemark available (opencv.face present): {FACEMARK_AVAILABLE}')
	print(f'Facemark initialized: {facemark_available}')
	print(f'MediaPipe face available: {MP_AVAILABLE}')
	print(f'MediaPipe hands available: {MP_HANDS_AVAILABLE}')
	print(f'Hand landmarker initialized: {hand_landmarker is not None}')
	print(f'Camera requested: {desired_frame_width}x{desired_frame_height}@{desired_fps}fps')
	print(f'Camera actual: {actual_width:.0f}x{actual_height:.0f}@{actual_fps:.1f}fps')

	# Frame timestamp for hand landmarker (VIDEO mode requires timestamps)
	frame_timestamp_ms = 0
	frame_interval_ms = int(1000 / actual_fps) if actual_fps and actual_fps > 1e-3 else 33

	print('Press q to quit.')
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_timestamp_ms += frame_interval_ms

		fh, fw = frame.shape[:2]
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
		raw_score = 0.0
		face_found = False
		mouth_raw = 0.0
		backend = 'none'

		# Detection backends: prefer Facemark -> MediaPipe -> heuristic
		if facemark_available:
			try:
				res = facemark.fit(gray, faces)
				landmarks = None
				if isinstance(res, tuple) and len(res) == 2:
					_, landmarks = res
				else:
					landmarks = res
			except Exception:
				landmarks = None

			if landmarks is not None and len(landmarks) > 0:
				face_found = True
				pts = np.array(landmarks[0]).reshape(-1, 2)
				mouth_pts = pts[48:68]
				eye_l_pts = pts[36:42]
				eye_r_pts = pts[42:48]
				mouth_w = np.max(mouth_pts[:, 0]) - np.min(mouth_pts[:, 0])
				mouth_h = np.max(mouth_pts[:, 1]) - np.min(mouth_pts[:, 1])
				eye_l_c = np.mean(eye_l_pts, axis=0)
				eye_r_c = np.mean(eye_r_pts, axis=0)
				eye_w = np.linalg.norm(eye_r_c - eye_l_c)
				if eye_w > 1e-6:
					# Smile metric: mouth aspect ratio (width/height) - smiles are wide and flat
					mouth_aspect = mouth_w / (mouth_h + 1e-6)
					# Normalize: aspect > 3 is a strong smile, ~2 is neutral
					# Also check if mouth corners (pts 48, 54) are higher than center bottom (pt 57)
					left_corner_y = mouth_pts[0][1]
					right_corner_y = mouth_pts[6][1]
					bottom_center_y = mouth_pts[9][1]
					# Corner elevation: positive means corners are above bottom center (smile)
					corner_elevation = ((bottom_center_y - left_corner_y) + (bottom_center_y - right_corner_y)) / (2.0 * eye_w)
					# Smile score: high aspect (wide/flat mouth) + elevated corners
					raw_score = max(0.0, (mouth_aspect - 2.0) * 0.3 + corner_elevation * 2.0)
					raw_score = min(1.0, raw_score)
					# Mouth-open metric
					mouth_raw = mouth_h / eye_w
					mouth_smoothed = mouth_alpha * mouth_raw + (1 - mouth_alpha) * mouth_smoothed
					# Eye-open metric (average of both eyes height/width)
					eye_l_w = np.max(eye_l_pts[:, 0]) - np.min(eye_l_pts[:, 0])
					eye_r_w = np.max(eye_r_pts[:, 0]) - np.min(eye_r_pts[:, 0])
					eye_l_h = np.max(eye_l_pts[:, 1]) - np.min(eye_l_pts[:, 1])
					eye_r_h = np.max(eye_r_pts[:, 1]) - np.min(eye_r_pts[:, 1])
					eye_raw = ((eye_l_h / (eye_l_w + 1e-6)) + (eye_r_h / (eye_r_w + 1e-6))) / 2.0
					eye_smoothed = eye_alpha * eye_raw + (1 - eye_alpha) * eye_smoothed
					backend = 'facemark'
					# Smooth mouth landmark positions (EMA) to reduce jitter
					cur_pts = mouth_pts.astype(np.float32)
				
					if prev_mouth_pts is None or prev_mouth_pts.shape != cur_pts.shape:
						prev_mouth_pts = cur_pts.copy()
					else:
						prev_mouth_pts = landmark_smooth_alpha * cur_pts + (1 - landmark_smooth_alpha) * prev_mouth_pts
					# draw smoothed mouth landmarks for debugging
					if show_debug:
						for (px, py) in prev_mouth_pts.astype(int):
							cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)
		elif MP_AVAILABLE and mp_face is not None:
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = mp_face.process(rgb)
			if results.multi_face_landmarks:
				face_found = True
				lm = results.multi_face_landmarks[0].landmark
				def to_xy(idx):
					p = lm[idx]
					return np.array([p.x * fw, p.y * fh])
				try:
					mouth_l = to_xy(61)
					mouth_r = to_xy(291)
					mouth_top = to_xy(13)
					mouth_bottom = to_xy(14)
					eye_l = to_xy(33)
					eye_r = to_xy(263)
					mouth_w = np.linalg.norm(mouth_r - mouth_l)
					mouth_h = np.linalg.norm(mouth_bottom - mouth_top)
					eye_w = np.linalg.norm(eye_r - eye_l)
					if eye_w > 1e-6:
						# Smile metric: mouth aspect ratio (width/height) - smiles are wide and flat
						mouth_aspect = mouth_w / (mouth_h + 1e-6)
						# Corner elevation using MediaPipe points
						left_corner_y = mouth_l[1]
						right_corner_y = mouth_r[1]
						bottom_center_y = mouth_bottom[1]
						corner_elevation = ((bottom_center_y - left_corner_y) + (bottom_center_y - right_corner_y)) / (2.0 * eye_w)
						# Smile score
						raw_score = max(0.0, (mouth_aspect - 2.0) * 0.3 + corner_elevation * 2.0)
						raw_score = min(1.0, raw_score)
						# Mouth-open metric (height normalized by inter-eye distance)
						mouth_raw = mouth_h / eye_w
						# Smooth mouth metric
						mouth_smoothed = mouth_alpha * mouth_raw + (1 - mouth_alpha) * mouth_smoothed
						# Eye-open metric using MediaPipe eye landmarks
						left_top = to_xy(159)
						left_bottom = to_xy(145)
						right_top = to_xy(386)
						right_bottom = to_xy(374)
						left_width = eye_w  # reuse inter-eye distance as scale
						right_width = eye_w
						left_h = np.linalg.norm(left_bottom - left_top)
						right_h = np.linalg.norm(right_bottom - right_top)
						eye_raw = ((left_h / (left_width + 1e-6)) + (right_h / (right_width + 1e-6))) / 2.0
						eye_smoothed = eye_alpha * eye_raw + (1 - eye_alpha) * eye_smoothed
						backend = 'mediapipe'
				except Exception:
					raw_score = 0.0
		else:
			# Heuristic fallback: analyze lower-face ROI and Haar-smile
			for (x, y, w, h) in faces:
				face_found = True
				# Haar-smile (keep as raw_score fallback)
				roi_gray = gray[y:y+h, x:x+w]
				smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
				if len(smiles) > 0:
					raw_score = min(1.0, 0.4 + 0.2 * len(smiles))
				# ROI heuristic for mouth opening
				ly = y + int(h * 0.45)
				ry2 = y + h
				if ly < y + h and w > 10 and h > 20:
					roi_mouth = gray[ly:y+h, x:x+w]
					if roi_mouth.size > 0:
						blur = cv2.GaussianBlur(roi_mouth, (5, 5), 0)
						_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
						th_inv = cv2.bitwise_not(th)
						contours, _ = cv2.findContours(th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
						if contours:
							cnt = max(contours, key=cv2.contourArea)
							rx, ry, rw, rh = cv2.boundingRect(cnt)
							if rw > 6 and rh > 4 and rw > rh:
								mouth_raw = float(rh) / float(h)
								mouth_raw = min(1.0, mouth_raw * 1.6)
								mouth_smoothed = mouth_alpha * mouth_raw + (1 - mouth_alpha) * mouth_smoothed
								# draw box
								cv2.rectangle(frame, (x+rx, ly+ry), (x+rx+rw, ly+ry+rh), (255, 0, 255), 1)
								backend = 'heuristic'


		# Smooth the score with EMA
		smoothed_score = alpha * raw_score + (1 - alpha) * smoothed_score

		# If MediaPipe is available, mouth_smoothed has been updated above; otherwise keep it decaying
		if not MP_AVAILABLE:
			mouth_smoothed = mouth_alpha * 0.0 + (1 - mouth_alpha) * mouth_smoothed
			eye_smoothed = eye_alpha * 0.0 + (1 - eye_alpha) * eye_smoothed

		# Hand detection for cute pose (hand near mouth) and 67 gestures
		hand_near_mouth = False
		pose67_detected = False
		pose67_motion_detected = False
		palm_facing_count = 0  # Track how many hands have palms facing camera
		# NOTE: Do NOT clear pose67_top_hist / pose67_bottom_hist here - they need to persist across frames
		hand_detected = False
		if hand_landmarker is not None:
			try:
				rgb_hand = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_hand)
				hand_results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
				if hand_results.hand_landmarks and len(hand_results.hand_landmarks) > 0:
					hand_detected = True
					# Get mouth center from face detection
					mouth_center_x, mouth_center_y = fw // 2, fh // 2  # default
					if facemark_available and prev_mouth_pts is not None:
						mouth_center_x = int(np.mean(prev_mouth_pts[:, 0]))
						mouth_center_y = int(np.mean(prev_mouth_pts[:, 1]))
					elif len(faces) > 0:
						fx, fy, fwid, fhei = faces[0]
						mouth_center_x = fx + fwid // 2
						mouth_center_y = fy + int(fhei * 0.75)
					# Draw mouth center for debugging
					if show_debug:
						cv2.circle(frame, (mouth_center_x, mouth_center_y), 10, (0, 255, 0), 2)
					# Fingertip landmarks
					fingertip_indices = [4, 8, 12, 16, 20]
					mouth_radius = fw * 0.20
					hand_centers = []
					for hand_lms in hand_results.hand_landmarks:
						if show_debug:
							for i, lm in enumerate(hand_lms):
								px, py = int(lm.x * fw), int(lm.y * fh)
								color = (255, 200, 0) if i in fingertip_indices else (100, 100, 255)
								cv2.circle(frame, (px, py), 3, color, -1)
						hand_center = np.mean(np.array([[lm.x * fw, lm.y * fh] for lm in hand_lms], dtype=np.float32), axis=0)
						hand_centers.append(hand_center)
						for tip_idx in fingertip_indices:
							tip = hand_lms[tip_idx]
							tip_x = int(tip.x * fw)
							tip_y = int(tip.y * fh)
							dist = np.sqrt((tip_x - mouth_center_x)**2 + (tip_y - mouth_center_y)**2)
							if dist < mouth_radius:
								hand_near_mouth = True
								if show_debug:
									cv2.circle(frame, (tip_x, tip_y), 12, (255, 0, 255), 3)
							if show_debug:
								cv2.circle(frame, (mouth_center_x, mouth_center_y), int(mouth_radius), (255, 255, 0), 1)
						# 67 static pose
						try:
							pts = np.array([[lm.x * fw, lm.y * fh] for lm in hand_lms], dtype=np.float32)
							wrist = pts[0]
							index_mcp, middle_mcp, ring_mcp, pinky_mcp = pts[5], pts[9], pts[13], pts[17]
							thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = pts[4], pts[8], pts[12], pts[16], pts[20]
							palm_span = max(1.0, np.linalg.norm(index_mcp - pinky_mcp))
							scale = max(palm_span, np.linalg.norm(wrist - middle_mcp))
							def angle(a, b, c):
								ba = a - b
								bc = c - b
								den = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
								cosang = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
								return np.degrees(np.arccos(cosang))
							idx_angle = angle(pts[6], pts[7], index_tip)
							mid_angle = angle(pts[10], pts[11], middle_tip)
							ring_angle = angle(pts[14], pts[15], ring_tip)
							d_thumb_ring = np.linalg.norm(thumb_tip - ring_tip)
							d_idx_mid = np.linalg.norm(index_tip - middle_tip)
							pinky_low = (pinky_tip[1] - ring_tip[1]) > 0.08 * scale
							index_high = index_tip[1] + 0.05 * scale < thumb_tip[1]
							middle_high = middle_tip[1] + 0.05 * scale < thumb_tip[1]
							spread_ok = 0.12 * scale < d_idx_mid < 0.55 * scale
							thumb_ring_close = d_thumb_ring < 0.18 * scale
							ring_bent = 50 <= ring_angle <= 120
							index_straight = idx_angle >= 155
							middle_straight = mid_angle >= 155
							thumb_not_on_index = np.linalg.norm(thumb_tip - index_tip) > 0.22 * scale
							if index_straight and middle_straight and ring_bent and thumb_ring_close and spread_ok and pinky_low and index_high and middle_high and thumb_not_on_index:
								pose67_detected = True
						except Exception:
							pass
					# Two-hand motion 67 gesture (stacked hands moving up/down, palms facing camera)
					# Check if both hands have palms facing the camera (flat in Z)
					palm_facing_hands = []
					for hand_lms in hand_results.hand_landmarks:
						# Get Z values for key landmarks to check palm orientation
						# Wrist(0), Index MCP(5), Middle MCP(9), Ring MCP(13), Pinky MCP(17)
						# If palm faces camera, these Z values should be similar
						wrist_z = hand_lms[0].z
						index_mcp_z = hand_lms[5].z
						middle_mcp_z = hand_lms[9].z
						ring_mcp_z = hand_lms[13].z
						pinky_mcp_z = hand_lms[17].z
						palm_z_values = [wrist_z, index_mcp_z, middle_mcp_z, ring_mcp_z, pinky_mcp_z]
						z_range = max(palm_z_values) - min(palm_z_values)
						# Also check fingertips have similar Z to palm (hand is flat, not fist)
						index_tip_z = hand_lms[8].z
						middle_tip_z = hand_lms[12].z
						# Palm facing camera: small Z variation across palm, fingers extended forward
						# Threshold: Z range should be small (< 0.15 for facing camera)
						palm_is_flat = z_range < 0.15
						# Also: the hand should be spread horizontally (X span > Y span suggests palm facing)
						pts_2d = np.array([[lm.x, lm.y] for lm in hand_lms], dtype=np.float32)
						x_span = pts_2d[:, 0].max() - pts_2d[:, 0].min()
						y_span = pts_2d[:, 1].max() - pts_2d[:, 1].min()
						# For palm facing camera with fingers up/down, Y span is larger than X span
						# For palm facing camera horizontally, X span is larger
						# We want hands that are relatively "flat" toward camera - not edge-on
						palm_facing = palm_is_flat
						if palm_facing:
							hc = np.mean(np.array([[lm.x * fw, lm.y * fh] for lm in hand_lms], dtype=np.float32), axis=0)
							palm_facing_hands.append(hc)
							palm_facing_count += 1
					
					if len(palm_facing_hands) >= 2:
						c1, c2 = palm_facing_hands[0], palm_facing_hands[1]
						top_c, bot_c = (c1, c2) if c1[1] < c2[1] else (c2, c1)
						x_gap = abs(top_c[0] - bot_c[0])
						y_gap = bot_c[1] - top_c[1]
						# Relaxed: hands stacked with some vertical gap, x loosely aligned
						sufficient_stack = y_gap > fh * 0.05 and x_gap < fw * 0.35
						if sufficient_stack:
							pose67_top_hist.append(top_c[1])
							pose67_bottom_hist.append(bot_c[1])
							if len(pose67_top_hist) > pose67_hist_max:
								pose67_top_hist.pop(0)
							if len(pose67_bottom_hist) > pose67_hist_max:
								pose67_bottom_hist.pop(0)
							# Require fewer frames and less motion range
							if len(pose67_top_hist) >= 5 and len(pose67_bottom_hist) >= 5:
								range_top = max(pose67_top_hist) - min(pose67_top_hist)
								range_bot = max(pose67_bottom_hist) - min(pose67_bottom_hist)
								# Either hand moving up/down triggers detection
								if range_top > fh * 0.04 or range_bot > fh * 0.04:
									pose67_motion_detected = True
						else:
							# Only clear if consistently not stacked
							if len(pose67_top_hist) > 0:
								pose67_top_hist.pop(0)
							if len(pose67_bottom_hist) > 0:
								pose67_bottom_hist.pop(0)
					else:
						# Only one hand or none - gradually decay history
						if len(pose67_top_hist) > 0:
							pose67_top_hist.pop(0)
						if len(pose67_bottom_hist) > 0:
							pose67_bottom_hist.pop(0)
			except Exception:
				pass

		# Track palm_facing_count oscillation only if enabled
		if pose67_enabled:
			# When hands move up/down, palm detection flickers between values
			pose67_motion_hist.append(palm_facing_count)
			if len(pose67_motion_hist) > pose67_motion_hist_max:
				pose67_motion_hist.pop(0)
			# Count transitions (any change in palm_facing_count)
			transitions = 0
			for i in range(1, len(pose67_motion_hist)):
				if pose67_motion_hist[i] != pose67_motion_hist[i - 1]:
					transitions += 1
			# If we have enough transitions in recent history, it's oscillating (real up/down motion)
			pose67_oscillating = transitions >= pose67_min_transitions

			# Play 67.mp3 whenever the 67 overlay is active and audio is not already playing
			if pose67_oscillating and PYGAME_AVAILABLE and os.path.exists(pose67_audio_path):
				try:
					if not pygame.mixer.music.get_busy():
						pygame.mixer.music.load(pose67_audio_path)
						pygame.mixer.music.play()
						pose67_audio_playing = True
				except Exception:
					pass
		else:
			# Disabled: clear oscillation state and history; stop audio if playing
			pose67_oscillating = False
			pose67_motion_hist.clear()
			pose67_top_hist.clear()
			pose67_bottom_hist.clear()
			if PYGAME_AVAILABLE:
				try:
					if pygame.mixer.music.get_busy():
						pygame.mixer.music.stop()
						pose67_audio_playing = False
				except Exception:
					pass

		pose67_was_oscillating = pose67_oscillating

		# Hysteresis counters for smile (uses mouth_smoothed)
		if face_found and mouth_smoothed > on_threshold and mouth_smoothed <= mouth_on_threshold:
			on_count += 1
			off_count = 0
		elif mouth_smoothed < off_threshold:
			off_count += 1
			on_count = 0
		else:
			# In-between: slowly decay counters
			on_count = max(0, on_count - 1)
			off_count = max(0, off_count - 1)

		# Mouth-open hysteresis
		if face_found and mouth_smoothed > mouth_on_threshold:
			mouth_on_count += 1
			mouth_off_count = 0
		elif mouth_smoothed < mouth_off_threshold:
			mouth_off_count += 1
			mouth_on_count = 0
		else:
			mouth_on_count = max(0, mouth_on_count - 1)
			mouth_off_count = max(0, mouth_off_count - 1)

		# Eye-open hysteresis (stunned)
		if face_found and eye_smoothed > eye_on_threshold:
			eye_on_count += 1
			eye_off_count = 0
		elif eye_smoothed < eye_off_threshold:
			eye_off_count += 1
			eye_on_count = 0
		else:
			eye_on_count = max(0, eye_on_count - 1)
			eye_off_count = max(0, eye_off_count - 1)

		if on_count >= on_frames_required:
			overlay_visible = True
		if off_count >= off_frames_required or not face_found:
			overlay_visible = False

		if mouth_on_count >= mouth_on_frames_required:
			mouth_overlay_visible = True
		if mouth_off_count >= mouth_off_frames_required or not face_found:
			mouth_overlay_visible = False

		if eye_on_count >= eye_on_frames_required:
			eye_overlay_visible = True
		if eye_off_count >= eye_off_frames_required or not face_found:
			eye_overlay_visible = False

		# Cute hysteresis (hand near mouth - no smile requirement, just hand near mouth)
		if face_found and hand_near_mouth:
			cute_on_count += 1
			cute_off_count = 0
		elif not hand_near_mouth:
			cute_off_count += 1
			cute_on_count = 0
		else:
			cute_on_count = max(0, cute_on_count - 1)
			cute_off_count = max(0, cute_off_count - 1)

		if cute_on_count >= cute_on_frames_required:
			cute_overlay_visible = True
		if cute_off_count >= cute_off_frames_required or not face_found:
			cute_overlay_visible = False

		# Flat hysteresis (neutral when no other strong cues)
		flat_condition = face_found and not mouth_overlay_visible and not eye_overlay_visible and not cute_overlay_visible \
			and mouth_smoothed <= 0.35 #and raw_score < off_threshold and eye_smoothed < eye_off_threshold
		if flat_condition:
			flat_on_count += 1
			flat_off_count = 0
		else:
			flat_off_count += 1
			flat_on_count = 0

		if flat_on_count >= flat_on_frames_required:
			flat_overlay_visible = True
		if flat_off_count >= flat_off_frames_required or not face_found:
			flat_overlay_visible = False

		# Update checklist when each overlay becomes visible
		if eye_overlay_visible:
			checklist_items['stunned'] = True
		if cute_overlay_visible:
			checklist_items['cute'] = True
		if mouth_overlay_visible:
			checklist_items['surprised'] = True
		if overlay_visible:
			checklist_items['smile'] = True
		if flat_overlay_visible:
			checklist_items['flat'] = True

		# Draw face rectangles (optional)
		if show_debug:
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# Draw checklist overlay
		draw_checklist(frame, checklist_items)

		# Overlay priority: stunned > cute > surprised > smile > flat
		if eye_overlay_visible:
			if os.path.exists(stunned_path):
				stunned_img = cv2.imread(stunned_path, cv2.IMREAD_UNCHANGED)
				if stunned_img is not None:
					overlay_w = max(overlay_min_width, int(fw * overlay_width_ratio))
					ow = stunned_img.shape[1]
					oh = stunned_img.shape[0]
					overlay_h = int(oh * (overlay_w / float(ow)))
					overlay_resized = cv2.resize(stunned_img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
					x = fw - overlay_w - 10
					y = 10
					overlay_image_alpha(frame, overlay_resized, x, y)
		elif cute_overlay_visible:
			if os.path.exists(cute_path):
				cute_img = cv2.imread(cute_path, cv2.IMREAD_UNCHANGED)
				if cute_img is not None:
					overlay_w = max(overlay_min_width, int(fw * overlay_width_ratio))
					ow = cute_img.shape[1]
					oh = cute_img.shape[0]
					overlay_h = int(oh * (overlay_w / float(ow)))
					overlay_resized = cv2.resize(cute_img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
					x = fw - overlay_w - 10
					y = 10
					overlay_image_alpha(frame, overlay_resized, x, y)
		elif mouth_overlay_visible:
			if os.path.exists(surprised_path):
				surprised_img = cv2.imread(surprised_path, cv2.IMREAD_UNCHANGED)
				if surprised_img is not None:
					overlay_w = max(overlay_min_width, int(fw * overlay_width_ratio))
					ow = surprised_img.shape[1]
					oh = surprised_img.shape[0]
					overlay_h = int(oh * (overlay_w / float(ow)))
					overlay_resized = cv2.resize(surprised_img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
					x = fw - overlay_w - 10
					y = 10
					overlay_image_alpha(frame, overlay_resized, x, y)
		elif flat_overlay_visible:
			if os.path.exists(flat_path):
				flat_img = cv2.imread(flat_path, cv2.IMREAD_UNCHANGED)
				if flat_img is not None:
					overlay_w = max(overlay_min_width, int(fw * overlay_width_ratio))
					ow = flat_img.shape[1]
					oh = flat_img.shape[0]
					overlay_h = int(oh * (overlay_w / float(ow)))
					overlay_resized = cv2.resize(flat_img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
					x = fw - overlay_w - 10
					y = 10
					overlay_image_alpha(frame, overlay_resized, x, y)
		elif overlay_visible and (thanks_img is not None):
			# overlay width = fraction of frame width
			overlay_w = max(overlay_min_width, int(fw * overlay_width_ratio))
			ow = thanks_img.shape[1]
			oh = thanks_img.shape[0]
			overlay_h = int(oh * (overlay_w / float(ow)))
			overlay_resized = cv2.resize(thanks_img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
			x = fw - overlay_w - 10
			y = 10
			overlay_image_alpha(frame, overlay_resized, x, y)

		# 67 meme gesture overlay - shown when oscillating (continuous up/down motion detected)
		if pose67_enabled and pose67_oscillating:
			if os.path.exists(pose67_path):
				pose67_img = cv2.imread(pose67_path, cv2.IMREAD_UNCHANGED)
				if pose67_img is not None:
					overlay_w = max(overlay_min_width, int(fw * overlay_width_ratio))
					ow = pose67_img.shape[1]
					oh = pose67_img.shape[0]
					overlay_h = int(oh * (overlay_w / float(ow)))
					overlay_resized = cv2.resize(pose67_img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
					# Place on the left side so it doesn't overlap expression overlays
					x = 10
					y = 10
					overlay_image_alpha(frame, overlay_resized, x, y)

		# Draw debug info: raw and smoothed score, overlay state, and whether MediaPipe is used
		if show_debug:
			debug_color = (0, 255, 255)
			cv2.putText(frame, f'raw:{raw_score:.2f}', (10, fh - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
			cv2.putText(frame, f'smoothed:{smoothed_score:.2f}', (10, fh - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
			cv2.putText(frame, f'mouth:{mouth_smoothed:.2f} eye:{eye_smoothed:.2f}', (10, fh - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
			cv2.putText(frame, f'overlay:{int(overlay_visible)} mouth:{int(mouth_overlay_visible)} cute:{int(cute_overlay_visible)} stunned:{int(eye_overlay_visible)} flat:{int(flat_overlay_visible)}', (10, fh - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
			cv2.putText(frame, f'hand_detected:{int(hand_detected)} hand_near_mouth:{int(hand_near_mouth)} palm_facing:{palm_facing_count} pose67:{int(pose67_detected)} enabled:{int(pose67_enabled)}', (10, fh - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
			# Show oscillation info for 67 gesture
			transitions = sum(1 for i in range(1, len(pose67_motion_hist)) if pose67_motion_hist[i] != pose67_motion_hist[i - 1])
			cv2.putText(frame, f'pose67_oscillating:{int(pose67_oscillating)} transitions:{transitions}/{pose67_min_transitions}', (10, fh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
			cv2.putText(frame, f'backend:{backend}', (fw - 160, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, debug_color, 2)

		cv2.imshow('MyGO Smiler', frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		if key == ord('d'):
			show_debug = not show_debug
		if key == ord('r'):
			for k in checklist_items:
				checklist_items[k] = False
		if key == ord('g'):
			pose67_enabled = not pose67_enabled
			if not pose67_enabled and PYGAME_AVAILABLE:
				try:
					if pygame.mixer.music.get_busy():
						pygame.mixer.music.stop()
				except Exception:
					pass

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()

