🎵 Vocals Remover (Vocal Separator Pro)

A professional desktop application for separating vocals and instrumental tracks from audio files using state-of-the-art deep learning models (Demucs). Built with PyQt6 for a modern GUI and PyTorch for high‑quality audio source separation.

📌 Project Overview

Vocals Remover is a Python-based desktop application that allows users to upload an audio file and automatically split it into:

Vocals

Individual instrumental stems (drums, bass, etc.)

A combined instrumental track

The application is designed for music producers, students, content creators, and developers who need an easy-to-use yet powerful vocal separation tool.

✨ Key Features

🎧 High-quality vocal and instrumental separation

🖥️ Modern dark-themed GUI (PyQt6)

⚡ Multi-threaded processing (non-blocking UI)

🎛️ Multiple Demucs models support

📂 Supports multiple audio formats (MP3, WAV, FLAC, M4A, OGG)

🧠 Automatic resampling and stereo handling

📜 Real-time processing logs

📁 Custom output directory support

🧰 Tech Stack

Frontend (GUI)

PyQt6 – Desktop user interface

Qt Threads (QThread) – Background processing

Backend (Audio Processing)

Python 3.9+

PyTorch – Deep learning framework

Demucs – Music source separation model

Torchaudio – Audio loading & saving

NumPy – Audio data manipulation

SoundFile / PyDub – Fallback audio loaders

⚙️ Setup and Installation

1️⃣ Clone the Repository

git clone https://github.com/s-saqlain/Vocals-Remover.git
cd Vocals-Remover

2️⃣ Create Virtual Environment (Recommended)

python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)

3️⃣ Install Dependencies

pip install -r requirements.txt

⚠️ Note: For GPU acceleration, ensure CUDA is installed and compatible with your PyTorch version.

▶️ Usage Instructions

Run the application:

python main.py

Click Select Audio File and choose an audio file

Select a Demucs model

Choose output directory (optional)

Click Separate Audio

Wait for processing to complete

Access separated files in the output folder

🔌 API Endpoints

❌ Not applicable

This project is a standalone desktop application and does not expose REST or HTTP APIs.

🎨 Frontend Overview

Built using PyQt6

Dark modern UI for professional look

Components include:

File selector

Model selector

Progress indicator

Real-time logs

Action buttons

The UI remains responsive due to background processing using QThread.

🧠 Backend Overview

Uses Demucs pretrained models for audio separation

Automatically detects CPU/GPU

Handles:

Audio loading

Resampling

Stereo normalization

Source separation

File export

Main backend logic is implemented inside:

SeperationThread class

🚀 Future Improvements

🎚️ Waveform visualization

🎛️ Model parameter tuning

📦 Export stems in ZIP format

🌐 Web version (Flask / FastAPI)

🧩 Plugin support (VST)

⚡ Progress percentage tracking

🎼 Batch audio processing

📄 License

This project is licensed under the MIT License.

You are free to:

Use

Modify

Distribute

With proper attribution.

👤 Author

SaqlainGitHub: https://github.com/s-saqlain

⭐ If you like this project, don’t forget to star the repository!

