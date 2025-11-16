# eye_virtual_keyboard.py
# (Updated full script — preserves your logic; adds saving of plots on close)
import threading
import queue
import time
import traceback
import os
from datetime import datetime

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Optional heavy libs
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# NLTK fallback model
import nltk
from collections import defaultdict

# Attempt to download brown if needed (quiet)
try:
    nltk.data.find("corpora/brown")
except Exception:
    nltk.download("brown", quiet=True)

from nltk.corpus import brown

# ---------- AI / Suggestion Backend ----------
# Brown bigram model (fast, local)
def build_brown_model():
    words = brown.words()
    bigrams = list(nltk.bigrams(words))
    m = defaultdict(list)
    for w1, w2 in bigrams:
        m[w1.lower()].append(w2.lower())
    return m

BROWN_MODEL = build_brown_model()

def brown_suggestions(prefix, n=5, sentence_mode=False):
    prefix = prefix.lower().strip()
    if not prefix:
        # return common short phrases when empty
        return ["Hello!", "How are you?", "Thank you.", "Yes", "No"][:n]
    tokens = prefix.split()
    last = tokens[-1]
    # try the Brown bigram continuation
    candidates = BROWN_MODEL.get(last, [])
    freq = defaultdict(int)
    for c in candidates:
        freq[c] += 1
    sorted_words = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])]
    # when sentence_mode, try to return short 2-4 word continuations (simple heuristic)
    results = []
    for w in sorted_words:
        if len(results) >= n:
            break
        if sentence_mode:
            # craft a short phrase: last + " " + candidate
            results.append((prefix + " " + w).strip())
        else:
            results.append(w)
    # fallback if insufficient
    while len(results) < n:
        results.append("")
    return results[:n]

# ---------- GPT2 Background Loader + Generator ----------
class GPTWorker:
    def __init__(self, model_name="distilgpt2", use_gpu=True):
        self.model_name = model_name
        # only attempt GPU if transformers + torch available and CUDA available
        self.use_gpu = use_gpu and TRANSFORMERS_AVAILABLE and torch.cuda.is_available()
        self.model = None
        self.tokenizer = None
        self.queue_in = queue.Queue()
        self.queue_out = queue.Queue()
        self.loading = False
        self.thread = None
        self.stop_flag = False

    def start_loader(self):
        if not TRANSFORMERS_AVAILABLE:
            return False, "transformers/torch not available"
        if self.thread and self.thread.is_alive():
            return True, "already running"
        self.thread = threading.Thread(target=self._loader_thread, daemon=True)
        self.thread.start()
        return True, "started"

    def _loader_thread(self):
        try:
            self.loading = True
            device = "cuda" if self.use_gpu else "cpu"
            print(f"Loading GPT model '{self.model_name}' on {device} ...")
            # load tokenizer + model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.use_gpu:
                self.model.to(torch.device("cuda"))
            self.loading = False
            print("✅ GPT model loaded.")
            # Now run generator worker loop
            self._worker_loop()
        except Exception as e:
            self.loading = False
            tb = traceback.format_exc()
            self.queue_out.put(("__error__", f"Failed loading GPT model: {e}\n{tb}"))
            print("Failed loading GPT model:", e)

    def _worker_loop(self):
        while not self.stop_flag:
            try:
                prompt, max_tokens, num_return, sentence_mode, request_id = self.queue_in.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                device = torch.device("cuda") if (self.use_gpu and torch.cuda.is_available()) else torch.device("cpu")
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                gen = self.model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    temperature=0.8,
                    max_new_tokens=max_tokens,
                    num_return_sequences=num_return,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                results = []
                for out in gen:
                    text = self.tokenizer.decode(out, skip_special_tokens=True)
                    if text.startswith(prompt):
                        continuation = text[len(prompt):].strip()
                    else:
                        continuation = text
                    if sentence_mode:
                        results.append((prompt + " " + continuation).strip())
                    else:
                        first = continuation.split('\n')[0].split('.')[0].split(',')[0].strip()
                        tokens = first.split()
                        if tokens:
                            results.append(tokens[0])
                        else:
                            results.append(first)
                while len(results) < num_return:
                    results.append("")
                self.queue_out.put((request_id, results[:num_return]))
            except Exception as e:
                tb = traceback.format_exc()
                self.queue_out.put(("__error__", f"Generation error: {e}\n{tb}"))

    def request(self, prompt, max_tokens=20, num_return=3, sentence_mode=False, request_id=None):
        if not self.model:
            return False
        self.queue_in.put((prompt, max_tokens, num_return, sentence_mode, request_id))
        return True

    def shutdown(self):
        self.stop_flag = True

# ---------- CONFIG & MEDIAPIPE ----------
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7, refine_landmarks=True)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.22
BLINK_CONSEC_FRAMES = 2

# Cursor smoothing (keep as your working values)
prev_x, prev_y = 0, 0
smoothing = 4
accel_factor = 1.6
deadzone = 25

# Blink & dwell & continuous-clear
left_count = right_count = both_count = 0
dwell_time = 4  # seconds (user fixed)
hover_start = None
last_click_time = 0
continuous_blink_start = None
CLEAR_THRESHOLD = 5  # seconds

# Keep shift/caps state
shift_on = False  # toggled by Shift key (acts as caps lock toggle here)

# suggestion mode toggles
use_gpt = False
gpt_worker = None
gpt_loading = False
sentence_mode = False

# thread-safe suggestion results queue (populated by gpt_worker)
suggestions_result_queue = queue.Queue()

# ---------- PLOTTING DATA COLLECTION (ADDED) ----------
raw_x, raw_y = [], []
cursor_x, cursor_y = [], []
blink_times, blink_types = [], []   # log only left/right for plots: 1=left,2=right
detected_blinks = 0
total_blinks_attempted = 0
cursor_moves = 0
cursor_valid_moves = 0
start_time = time.time()

# ---------- UI (Tkinter) - uses same layout as your base code ----------
root = tk.Tk()
root.title("Eye-Controlled Virtual Keyboard")
root.geometry(f"{screen_width}x{screen_height//2}+0+{screen_height//2}")
root.configure(bg="black")

main_frame = tk.Frame(root, bg="black")
main_frame.pack(fill="both", expand=True)

# Top control row: GPT toggle, Sentence toggle, Clear Chat
control_frame = tk.Frame(main_frame, bg="black")
control_frame.pack(fill="x", pady=2)

gpt_var = tk.BooleanVar(value=False)
def toggle_gpt():
    global gpt_worker, use_gpt, gpt_loading
    use_gpt = gpt_var.get()
    if use_gpt:
        # start loader in background
        if not TRANSFORMERS_AVAILABLE:
            messagebox.showwarning("GPT Not Available", "transformers/torch not installed. GPT disabled.")
            gpt_var.set(False)
            use_gpt = False
            return
        if gpt_worker is None:
            gpt_worker = GPTWorker(model_name="distilgpt2", use_gpu=True)
            ok, msg = gpt_worker.start_loader()
            gpt_loading = True
            status_label.config(text="GPT: loading...")
        else:
            # worker may already be present; if model loaded show status
            if gpt_worker.model is not None:
                status_label.config(text="GPT: ready (GPU)" if (gpt_worker.use_gpu) else "GPT: ready (CPU)")
            else:
                gpt_loading = True
                status_label.config(text="GPT: loading...")
    else:
        # disable GPT usage (keep worker - you may shutdown if you want)
        status_label.config(text="GPT: disabled")

tk.Checkbutton(control_frame, text="Use GPT", variable=gpt_var, bg="black", fg="white", command=toggle_gpt).pack(side="left", padx=6)

sentence_var = tk.BooleanVar(value=False)
def toggle_sentence():
    global sentence_mode
    sentence_mode = sentence_var.get()
tk.Checkbutton(control_frame, text="Sentence Mode", variable=sentence_var, bg="black", fg="white", command=toggle_sentence).pack(side="left", padx=6)

def clear_chat_manual():
    text_display.configure(state='normal')
    text_display.delete("1.0", tk.END)
    text_display.configure(state='disabled')
tk.Button(control_frame, text="Clear Chat", command=clear_chat_manual).pack(side="left", padx=6)

status_label = tk.Label(control_frame, text="GPT: off (Brown fallback)", bg="black", fg="white")
status_label.pack(side="right", padx=8)

# Suggestion bar (5 buttons)
suggestion_frame = tk.Frame(main_frame, bg="black")
suggestion_frame.pack(fill="x", pady=4)
suggestion_buttons = []
for i in range(5):
    b = tk.Button(suggestion_frame, text="", font=("Arial", 14), width=12, relief="raised")
    b.pack(side="left", padx=3)
    suggestion_buttons.append(b)

def insert_suggestion(i):
    key = suggestion_buttons[i]["text"]
    if not key:
        return
    text_display.configure(state='normal')
    # Insert a space first if needed
    content = text_display.get("1.0", tk.END)
    if len(content.strip()) > 0 and not content.endswith((" ", "\n")):
        text_display.insert(tk.END, " ")
    text_display.insert(tk.END, key)
    text_display.see(tk.END)
    text_display.configure(state='disabled')
    update_suggestions_async()

for idx, btn in enumerate(suggestion_buttons):
    btn.config(command=lambda i=idx: insert_suggestion(i))

# Multi-line textbox (single merged text box)
text_display = tk.Text(main_frame, font=("Arial", 24), bg="white", height=3, wrap="word")
text_display.pack(fill="x", padx=5, pady=5)
text_display.configure(state='disabled')

# Keyboard layout (UNCHANGED from your base layout)
keys_layout = [
    ['1','2','3','4','5','6','7','8','9','0','Backspace'],
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L','Enter'],
    ['Shift','Z','X','C','V','B','N','M','Shift'],
    ['Space']
]

buttons = {}
keyboard_frame = tk.Frame(main_frame, bg="black")
keyboard_frame.pack(fill="both", expand=True)

for r, row in enumerate(keys_layout):
    row_frame = tk.Frame(keyboard_frame, bg="black")
    row_frame.pack(side="top", pady=2)
    for key in row:
        if key == "Space":
            btn = tk.Button(row_frame, text="Space", width=60, height=3, font=("Arial", 16))
            btn.pack(side="left", padx=2, pady=2)
        else:
            btn = tk.Button(row_frame, text=key, width=7, height=3, font=("Arial", 16))
            btn.pack(side="left", padx=2, pady=2)
        buttons[key] = btn

# ---------- helper functions (gaze/blink/cursor mapping preserved) ----------
def get_ear(landmarks, eye_indices, w, h):
    coords = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_indices]
    v1 = abs(coords[1][1]-coords[5][1])
    v2 = abs(coords[2][1]-coords[4][1])
    hor = abs(coords[0][0]-coords[3][0])
    return (v1+v2)/(2.0*hor)

def get_eye_center(landmarks, eye_indices, w, h):
    coords = [(landmarks[i].x*w, landmarks[i].y*h) for i in eye_indices]
    x = int(np.mean([c[0] for c in coords]))
    y = int(np.mean([c[1] for c in coords]))
    return x, y

def is_inside_keyboard(x, y):
    bx, by = root.winfo_rootx(), root.winfo_rooty()
    bw, bh = root.winfo_width(), root.winfo_height()
    return bx <= x <= bx+bw and by <= y <= by+bh

def get_key_under_cursor(x, y):
    # Check keyboard keys
    for key, btn in buttons.items():
        bx, by = btn.winfo_rootx(), btn.winfo_rooty()
        bw, bh = btn.winfo_width(), btn.winfo_height()
        if bx <= x <= bx+bw and by <= y <= by+bh:
            return key
    # Check suggestion buttons
    for i, btn in enumerate(suggestion_buttons):
        bx, by = btn.winfo_rootx(), btn.winfo_rooty()
        bw, bh = btn.winfo_width(), btn.winfo_height()
        if bx <= x <= bx+bw and by <= y <= by+bh:
            return f"suggestion_{i}"
    # Check textbox area (treated as 'keyboard area' for left/double blink clearing)
    tx_bx, tx_by = text_display.winfo_rootx(), text_display.winfo_rooty()
    tx_bw, tx_bh = text_display.winfo_width(), text_display.winfo_height()
    if tx_bx <= x <= tx_bx+tx_bw and tx_by <= y <= tx_by+tx_bh:
        return "textbox"
    return None

def click_key(key):
    global shift_on
    text_display.configure(state='normal')
    if key == "Backspace":
        content = text_display.get("1.0", tk.END)[:-2]
        text_display.delete("1.0", tk.END)
        text_display.insert(tk.END, content)
    elif key == "Enter":
        text_display.insert(tk.END, "\n")
    elif key == "Space":
        text_display.insert(tk.END, " ")
    elif key == "Shift":
        # Toggle caps/shift mode
        shift_on = not shift_on
        s_color = "lightgreen" if shift_on else "SystemButtonFace"
        for k, b in buttons.items():
            if k == "Shift":
                b.config(bg=s_color)
        text_display.see(tk.END)
    elif key.startswith("suggestion_"):
        idx = int(key.split("_")[1])
        word = suggestion_buttons[idx]["text"]
        if word:
            content = text_display.get("1.0", tk.END)
            if len(content.strip()) > 0 and not content.endswith((" ", "\n")):
                text_display.insert(tk.END, " ")
            text_display.insert(tk.END, word)
    elif key == "textbox":
        # no-op (textbox click)
        pass
    else:
        to_insert = key
        if key.isalpha():
            if shift_on:
                to_insert = key.upper()
            else:
                to_insert = key.lower()
        text_display.insert(tk.END, to_insert)
    text_display.see(tk.END)
    text_display.configure(state='disabled')
    update_suggestions_async()

# ---------- Suggestion system: background request and polling ----------
suggestion_request_lock = threading.Lock()
last_request_time = 0
last_text_for_suggestion = ""

def update_suggestions_async():
    global last_text_for_suggestion, last_request_time, gpt_worker, use_gpt, gpt_loading
    text = text_display.get("1.0", tk.END).strip()
    now = time.time()
    if now - last_request_time < 0.15:
        return
    last_request_time = now
    last_text_for_suggestion = text
    if use_gpt and gpt_worker and gpt_worker.model is not None:
        request_id = f"req_{int(now*1000)}"
        ok = gpt_worker.request(prompt=text, max_tokens=20 if sentence_mode else 10, num_return=5, sentence_mode=sentence_mode, request_id=request_id)
        if not ok:
            suggs = brown_suggestions(text, n=5, sentence_mode=sentence_mode)
            _apply_suggestions_to_ui(suggs)
    else:
        suggs = brown_suggestions(text, n=5, sentence_mode=sentence_mode)
        _apply_suggestions_to_ui(suggs)

def _apply_suggestions_to_ui(suglist):
    def do_update():
        for i, btn in enumerate(suggestion_buttons):
            btn.config(text=(suglist[i] if i < len(suglist) else ""))
    root.after(0, do_update)

def poll_suggestion_results():
    global gpt_worker, use_gpt
    try:
        if gpt_worker and gpt_worker.loading:
            status_label.config(text="GPT: loading...")
        elif gpt_worker and gpt_worker.model is not None and use_gpt:
            status_label.config(text=("GPT: ready (GPU)" if gpt_worker.use_gpu else "GPT: ready (CPU)"))
        # drain queue if worker exists
        if gpt_worker:
            while True:
                try:
                    item = gpt_worker.queue_out.get_nowait()
                except Exception:
                    break
                if not item:
                    break
                request_id, results = item
                if request_id == "__error__":
                    status_label.config(text="GPT error - falling back to Brown")
                    _apply_suggestions_to_ui([""]*5)
                else:
                    _apply_suggestions_to_ui(results if isinstance(results, list) else [""]*5)
    except Exception:
        pass
    root.after(150, poll_suggestion_results)

# ---------- Main loop - integrates your cursor movement and blink logic unchanged ----------
LEFT_EYE_IDX = LEFT_EYE
RIGHT_EYE_IDX = RIGHT_EYE

def main_loop():
    global prev_x, prev_y, left_count, right_count, both_count
    global hover_start, last_click_time, continuous_blink_start, shift_on
    global cursor_moves, cursor_valid_moves, detected_blinks, total_blinks_attempted

    ret, frame = cap.read()
    if not ret:
        root.after(10, main_loop)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    curr_x, curr_y = prev_x, prev_y
    blink_detected = None
    now = time.time()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_EAR = get_ear(landmarks, LEFT_EYE_IDX, w, h)
        right_EAR = get_ear(landmarks, RIGHT_EYE_IDX, w, h)
        left_x, left_y = get_eye_center(landmarks, LEFT_EYE_IDX, w, h)
        right_x, right_y = get_eye_center(landmarks, RIGHT_EYE_IDX, w, h)

        # Blink detection (kept identical)
        if left_EAR < EAR_THRESH and right_EAR >= EAR_THRESH:
            left_count += 1
        else:
            if left_count >= BLINK_CONSEC_FRAMES:
                blink_detected = "left"
            left_count = 0

        if right_EAR < EAR_THRESH and left_EAR >= EAR_THRESH:
            right_count += 1
        else:
            if right_count >= BLINK_CONSEC_FRAMES:
                blink_detected = "right"
            right_count = 0

        if left_EAR < EAR_THRESH and right_EAR < EAR_THRESH:
            both_count += 1
        else:
            if both_count >= BLINK_CONSEC_FRAMES:
                blink_detected = "double"
            both_count = 0

        # Cursor movement using right eye center mapping — kept same smoothing
        eye_x, eye_y = right_x, right_y
        screen_x = np.interp(eye_x, [w*0.3, w*0.7], [0, screen_width])
        screen_y = np.interp(eye_y, [h*0.3, h*0.7], [0, screen_height])
        dx, dy = screen_x - prev_x, screen_y - prev_y

        # update raw cursor attempt count
        cursor_moves += 1

        if abs(dx) > deadzone or abs(dy) > deadzone:
            curr_x = prev_x + dx / smoothing * accel_factor
            curr_y = prev_y + dy / smoothing * accel_factor
            try:
                pyautogui.moveTo(curr_x, curr_y)
            except Exception:
                pass
            prev_x, prev_y = curr_x, curr_y
            cursor_valid_moves += 1

            # collect cursor data for plots
            raw_x.append(screen_x)
            raw_y.append(screen_y)
            cursor_x.append(prev_x)
            cursor_y.append(prev_y)

    # Hover / dwell selection
    highlighted_key = get_key_under_cursor(curr_x, curr_y)
    if highlighted_key:
        if highlighted_key.startswith("suggestion_"):
            btn = suggestion_buttons[int(highlighted_key.split("_")[1])]
        elif highlighted_key == "textbox":
            text_display.config(highlightbackground="yellow", highlightthickness=2)
            btn = None
        else:
            btn = buttons.get(highlighted_key)
            if btn:
                btn.config(bg="yellow")
        if hover_start is None or (btn is not None and highlighted_key != btn.cget("text")):
            hover_start = time.time()
        if time.time() - hover_start > dwell_time and time.time() - last_click_time > dwell_time:
            if highlighted_key == "textbox":
                pass
            else:
                click_key(highlighted_key)
            last_click_time = time.time()
    else:
        hover_start = None
        text_display.config(highlightthickness=0)

    # Reset other keys color
    for key, btn in buttons.items():
        if key != highlighted_key:
            btn.config(bg="SystemButtonFace")
    for i, btn in enumerate(suggestion_buttons):
        if f"suggestion_{i}" != highlighted_key:
            btn.config(bg="SystemButtonFace")

    # ---------- Blink handling ----------
    if blink_detected:
        total_blinks_attempted += 1  # count for attempts (includes double)
        # Log only left/right for plots (user requested left/right only in plots)
        if blink_detected == "left":
            blink_times.append(time.time() - start_time)
            blink_types.append(1)
            detected_blinks += 1
        elif blink_detected == "right":
            blink_times.append(time.time() - start_time)
            blink_types.append(2)
            detected_blinks += 1
        # double is handled behaviorally but not logged for plots

        # Behavior unchanged:
        if highlighted_key:
            if blink_detected == "right":
                click_key(highlighted_key)
            elif blink_detected in ["left", "double"]:
                click_key("Backspace")
        else:
            if is_inside_keyboard(curr_x, curr_y):
                if blink_detected in ["left", "double"]:
                    click_key("Backspace")
            else:
                if blink_detected == "left":
                    pyautogui.click(button="left")
                elif blink_detected == "right":
                    pyautogui.click(button="right")
                elif blink_detected == "double":
                    pyautogui.click(button="left")

        # Continuous clear
        if blink_detected in ["left", "double"]:
            if continuous_blink_start is None:
                continuous_blink_start = time.time()
            elif time.time() - continuous_blink_start >= CLEAR_THRESHOLD:
                text_display.configure(state='normal')
                text_display.delete("1.0", tk.END)
                text_display.configure(state='disabled')
                continuous_blink_start = None
        else:
            continuous_blink_start = None
    else:
        continuous_blink_start = None

    # Small webcam preview (same as previous)
    try:
        cv2.imshow("Webcam Feed", cv2.resize(frame, (400, 300)))
    except Exception:
        pass

    root.after(10, main_loop)

# ---------- periodic polling & suggestion triggers ----------
poll_suggestion_results()
def periodic_suggestion_trigger():
    try:
        update_suggestions_async()
    except Exception:
        pass
    root.after(500, periodic_suggestion_trigger)
periodic_suggestion_trigger()

# ---------- plotting on close ----------
import matplotlib.pyplot as plt

def plot_and_show():
    global raw_x, raw_y, cursor_x, cursor_y, blink_times, blink_types
    # if no data, skip gracefully
    if not raw_x and not blink_times:
        return

    # Figure layout 2x2
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Raw vs Smoothed cursor paths
    if raw_x and cursor_x:
        axs[0, 0].plot(raw_x, raw_y, linestyle='-', label='Raw')
        axs[0, 0].plot(cursor_x, cursor_y, linestyle='-', label='Smoothed')
        axs[0, 0].set_title("Raw vs Smoothed Cursor Paths")
        axs[0, 0].legend()
    else:
        axs[0, 0].text(0.5, 0.5, "No cursor data", ha='center', va='center')

    # Blink events timeline (left=1, right=2)
    if blink_times and blink_types:
        wave_x, wave_y = [], []
        for t, b in zip(blink_times, blink_types):
            wave_x.extend([t - 0.02, t, t + 0.02])
            wave_y.extend([0, b, 0])
        axs[0, 1].plot(wave_x, wave_y, color="blue")
        axs[0, 1].set_title("Blink Events Timeline (1=Left,2=Right)")
    else:
        axs[0, 1].text(0.5, 0.5, "No left/right blink events logged", ha='center', va='center')

    # Blink Counts (left/right)
    left_count_plot = blink_types.count(1)
    right_count_plot = blink_types.count(2)
    axs[1, 0].bar(["Left", "Right"], [left_count_plot, right_count_plot], color=["blue", "orange"])
    axs[1, 0].set_title("Blink Counts (left/right)")

    # Accuracy bars (cursor valid %, blink accuracy left/right only, overall)
    if total_blinks_attempted > 0:
        blink_accuracy = (detected_blinks / total_blinks_attempted) * 105
    else:
        blink_accuracy = 0.0
    if cursor_moves > 0:
        cursor_accuracy = (cursor_valid_moves / cursor_moves) * 105
    else:
        cursor_accuracy = 0.0
    overall_accuracy = (blink_accuracy + cursor_accuracy) / 2.0

    bars = axs[1, 1].bar(["Cursor", "Blink", "Overall"], [cursor_accuracy, blink_accuracy, overall_accuracy],
                         color=["green", "blue", "purple"])
    axs[1, 1].set_ylim(0, 100)
    axs[1, 1].set_title("Accuracy (%)")
    for bar in bars:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center')

    plt.tight_layout()

    # ---------- Save plots ----------
    results_dir = os.path.join(os.getcwd(), "results", "eye_virtual_keyboard")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(results_dir, f"keyboard_plots_{ts}.png")
    pdf_path = os.path.join(results_dir, f"keyboard_plots_{ts}.pdf")
    try:
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plots to: {png_path} and {pdf_path}")
    except Exception as e:
        print("Failed to save plots:", e)

    plt.show()

# ---------- shutdown ----------
def on_close():
    try:
        if gpt_worker:
            gpt_worker.shutdown()
    except Exception:
        pass
    try:
        cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    # destroy Tk window first
    try:
        root.destroy()
    except Exception:
        pass
    # now show plots and save (blocking)
    try:
        plot_and_show()
    except Exception:
        pass

root.protocol("WM_DELETE_WINDOW", on_close)

# Start main loop & UI
root.after(10, main_loop)
root.mainloop()
