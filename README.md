👁️‍🗨️ Improved Gaze-Typing Prediction Pipeline
This project implements a two-stage deep learning pipeline for predicting typed characters based on gaze and hand-tracking data. Using Bidirectional LSTMs, the model first infers the finger used for typing and then classifies the intended key press. It supports advanced feature engineering, 3D normalization, and robust leave-one-out cross-validation (LOOCV) across participants.

🧠 Project Structure
🔍 Stage 1: Finger Prediction
Predicts which finger was used for typing based on:

Hand tip positions (normalized to wrist center)

Velocity of wrist movement

Delta (temporal) movement features

🔤 Stage 2: Character Prediction
Once a finger is inferred, a second model is trained to classify the exact character pressed, using:

Gaze-to-key distances

Finger-to-key distances

Temporal movement deltas

Wrist velocity

📁 Input Data
Format
Compressed .rar files per participant

Each contains .csv logs with columns like:

PressedLetter, CurrentLetter

LeftGazeHit, LeftGazeHitPosition_X, ...

Left_Hand_ThumbTip_X, Right_Hand_WristRoot_Y, ...

Key_Q_X, Key_Q_Y, Key_Q_Z, etc.

⚙️ Installation & Setup
bash
Copy
Edit
# 1. Clone this repository
git clone https://github.com/yourusername/gaze-typing-pipeline.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install additional tools for RAR extraction
sudo apt-get install unrar -y
pip install rarfile
▶️ How to Run
Step 1: Extract Participant Data
python
Copy
Edit
# Automatically unpacks .rar files into /content/extracted_participants/
extract_rar_files(rar_files, base_extract_dir)
Step 2: Preprocess and Normalize Features
python
Copy
Edit
df = load_and_preprocess_data("/content/extracted_participants")
Step 3: Run Evaluation
python
Copy
Edit
# Finger prediction
run_loocv_evaluation(df, stage1_features, 'InferredPressedFinger', "Stage 1")

# Character prediction within each finger cluster
run_loocv_evaluation(df_clustered, stage2_features, 'PressedLetter', "Stage 2")
🧮 Features Engineered
Dynamic Normalization: Origin set to average wrist root per frame

Gaze Hit Calculation: Uses valid left/right gaze hit detection

Delta Features: Time-series change per feature

Kinematics: Wrist velocity

Distances:

Gaze-to-key

Finger-to-key

Finger-to-gaze

📊 Evaluation Method
Leave-One-Out Cross-Validation (LOOCV) is performed:

Across participants for generalizability

With dynamic class weights for imbalance

Using classification reports and accuracy

📈 Sample Output
plaintext
Copy
Edit
Stage 1: Finger Prediction
Average Accuracy: 0.8421

Stage 2: Character Prediction per Finger
Finger 1: Accuracy = 0.7153
Finger 2: Accuracy = 0.6529
...



🛠 Technologies Used
Python, NumPy, Pandas

TensorFlow / Keras (Bidirectional LSTM)

scikit-learn (LOOCV, metrics)

Google Colab for prototyping

RAR extraction with rarfile + unrar
