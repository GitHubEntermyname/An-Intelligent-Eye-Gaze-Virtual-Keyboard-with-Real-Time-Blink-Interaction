# Eye-Controlled Virtual Keyboard (AI-Enhanced)

A **hands-free virtual keyboard** controlled entirely by **eye gaze and blinks**.  
This project integrates **MediaPipe FaceMesh** for gaze tracking, **OpenCV** for keyboard rendering, and **GPT‑2** for intelligent word/sentence prediction. Designed for accessibility, assistive technology, and futuristic HCI applications.

---

## 🚀 Features
- Real-time eye gaze → cursor highlight
- Blink-based key selection:
  - Left blink → character press
  - Right blink → delete
  - Long blink → enter/sentence mode
- Full onscreen keyboard drawn with **OpenCV**
- AI-powered suggestion engine:
  - **Rule-based mode** (dictionary frequency)
  - **GPT‑2 mode** (GPU accelerated, CUDA)
- Shift, Caps Lock, Clear Chat, and Sentence Prediction support
- GPU optimization for smooth inference (float16, no lag)

---

## ⚙️ In case you are using zip folder, 
-> go to Windows PowerShell (recommended) -> paste this command/prompt, and it will automatically run the files 
- (all libraries are provided in the zip file, just need to run them)

cd C:\Users\HP\Downloads\GazeTracking\GazeTracking-master
.\blink_env\Scripts\activate

python eye_virtual_keyboard.py

--- 

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/eye-virtual-keyboard.git
cd eye-virtual-keyboard
pip install -r requirements.txt

## Activate Conda Environment
conda env create -f environment.yml
conda activate eye_keyboard_env

## Run the main script
python src/eye_virtual_keyboard.py

