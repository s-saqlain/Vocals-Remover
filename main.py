import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QComboBox, QHBoxLayout, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import numpy as np

torch.set_num_threads(4)


class SeperationThread(QThread):
    progress= pyqtSignal(str)
    finished= pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, audio_path, output_dir, model_name):
        super().__init__()
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.model_name =  model_name
    
    def run(self):
        try:
            self.progress.emit("Loading model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = get_model(self.model_name)
            model.to(device)
            model.eval()
            self.progress.emit("Model loaded. Starting separation...")


            self.progress.emit("Loading audio file...")
            try:
                wav, sr = torchaudio.load(self.audio_path)
            except Exception:
                self.progress.emit("Using alternative audio loader...")
                try:
                    import soundfile as sf
                    data, sr = sf.read(self.audio_path)
                    wav = torch.from_numpy(data.T).float()
                except ImportError:
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(self.audio_path)
                        samples = np.array(audio.get_array_of_samples())

                        if audio.channels == 2:
                            samples = samples.reshape((-1, 2))
                        else:
                            samples = samples.reshape((-1, 1))

                        wav = torch.from_numpy(samples.T).float() / 32768.0
                        sr = audio.frame_rate
                    except ImportError:
                        raise Exception("Install soundfile or pydub")

            # Resample
            if sr != model.samplerate:
                self.progress.emit("Resampling audio...")
                wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
                sr = model.samplerate

            # Ensure stereo
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]

            self.progress.emit("Seperating audio sources...")
            wav = wav.to(device)

            with torch.no_grad():
                sources = apply_model(model, wav.unsqueeze(0))[0].cpu()

            self.progress.emit("Saving separated tracks...")
            os.makedirs(self.output_dir, exist_ok=True)

            base_name = Path(self.audio_path).stem
            source_names = model.sources

            vocals_idx = None
            instrumental = torch.zeros_like(sources[0])

            for i, name in enumerate(source_names):
                out = os.path.join(self.output_dir, f"{base_name}_{name}.wav")
                torchaudio.save(out, sources[i], sr)
                self.progress.emit(f"Saved: {name}.wav")

                if name.lower() == "vocals":
                    vocals_idx = i
                else:
                    instrumental += sources[i]

            if vocals_idx is not None:
                inst_path = os.path.join(self.output_dir, f"{base_name}_instrumental.wav")
                torchaudio.save(inst_path, instrumental, sr)
                self.progress.emit("Saved: instrumental.wav")

            self.finished.emit("Audio seperated successfully!")

        except Exception as e:
            self.error.emit(str(e))
       


class VocalSeparatorApp (QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_path =  None
        self.output_dir = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Vocal Separator Pro")
        self.setGeometry(100,100,700,600)

        #For dark theme
        self.setStyleSheet("""
        QMainWindow{
                           background-color: #1e1e2e
        }
        QLabel{
                   color: #cdd6f4;
                   font-size:13px;        
                           }
        QPushButton{
                    background-color:#89b4fa;
                    color:#1e1e2e;
                    border:none;
                    padding:12px;
                    font-size:14px;
                    font-weight:bold;
                    border-radius:6px;       
                           }
        QPushButton:hover{
                    background-color:#b4befe;       
                           }
        QPushButton:disabled{
                    background-color:#45475a;
                           }
        QComboBox{
                    background-color:#313244;
                    color: #cdd6f4;
                    border:2px solid #45475a;
                    padding:8px;
                    border-radius:6px;
                    font-size:13px;
                           }
        QComboBox:hover{
                    border:2px solid #89b4fa;       
                           }                 
        QComboBox:drop-down{
                    border:none;       
                           }
        QProgressBar{
                    background-color: #313244;
                    border: 2px solid #45475a;
                    border-radius: 6px;
                    text-align: center;
                    color: #cdd6f4;       
                           }
        QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 4px;
            }
        QTextEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 6px;
                padding: 8px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
                
        central_widget =  QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30,30,30,30)

        #Title

        title=QLabel("🎵 Vocal Separator Pro")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font=QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle=QLabel("Seperate vocals and instruments from any audio file")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color:#9399b2; font-size:14px;")
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        #Model selection
        model_layout=QHBoxLayout()
        model_label = QLabel("Model: ")
        model_label.setStyleSheet("font-weight:bold;")
        self.model_combo=QComboBox()
        self.model_combo.addItems([
            "htdemucs (Recommended-  Best Quality)",
            "htdemucs_ft (Fine-tuned)",
            "mdx_extra (Fast)"
        ])

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo,1)
        layout.addLayout(model_layout)

        #File Selection

        self.file_label = QLabel("No audio file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("""
                    background-color:#313244;
                    padding:20px;
                    border-radius: 8px;
                    border: 2px dashed #45475a;                              
        """)

        layout.addWidget(self.file_label)


        self.select_btn = QPushButton("📂 Select Audio File")
        self.select_btn.clicked.connect(self.select_file)
        layout.addWidget(self.select_btn)

        #Output directory

        self.output_label=QLabel("Output: Same as input file")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("color: #9399b2;")
        layout.addWidget(self.output_label)

        self.output_btn =  QPushButton("📂 Choose Output Directory")
        self.output_btn.clicked.connect(self.select_output)
        layout.addWidget(self.output_btn)

        #Progress bar
        self.progress_bar =  QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0,0)
        layout.addWidget(self.progress_bar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(180)
        layout.addWidget(self.log_text)

        #Seperate button
        self.seperate_btn = QPushButton("🎶 Seperate Audio")
        self.seperate_btn.clicked.connect(self.seperate_audio)
        self.seperate_btn.setEnabled(False)
        self.seperate_btn.setStyleSheet("""
                    QPushButton{
                                    background-color: #a6e3a1;
                                    color: #1e1e2e;
                                    font-size: 16px;
                                    padding: 15px;    
                                }
                    
                    QPushButton:hover{
                                background-color:#b8f5b0;
                                        }
                    QPushButton:disabled{
                                background-color: #45475a;
                                color: #6c7086;        
                                        }
        """)
        layout.addWidget(self.seperate_btn)
        layout.addStretch()

    def log(self,message):
        self.log_text.append(message)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg);;All Files (*.*)"
        )
        if file_path:
            self.audio_path =  file_path
            self.file_label.setText(f"Selected: {Path(file_path).name}")
            self.file_label.setStyleSheet("""
                    background-color: #313244;
                    padding: 20px;
                    border-radius:8px;
                    border: 2px solid #89b4fa;
                    color: #89b4fa;
            """)
            self.seperate_btn.setEnabled(True)
            self.log(f"Selected file: {file_path}")

            #Set default output directory
            if not self.output_dir:
                self.output_dir =  str(Path(file_path).parent / "seperated")
                self.output_label.setText(f"Output: {self.output_dir}")

    def select_output(self):
        dir_path =  QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        if dir_path:
            self.output_dir =  dir_path
            self.output_label.setText(f"Output: {dir_path}")
            self.log(f"Output directory: {dir_path}")
    
    def seperate_audio(self):
        if not self.output_dir:
            self.output_dir = str(Path(self.audio_path).parent / "separated")
        
        #Get model name
        model_text =  self.model_combo.currentText()
        model_name =  model_text.split(" ")[0]


        self.log(f"\n{'='*50}")
        self.log(f"Starting seperation with {model_name} model...")
        self.progress_bar.setVisible(True)
        self.seperate_btn.setEnabled(False)
        self.select_btn.setEnabled(False)

        #Create and start seperation thread
        self.thread =  SeperationThread(self.audio_path,self.output_dir, model_name)
        self.thread.setParent(self)
        self.thread.progress.connect(self.log)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()
    
    def on_finished(self, message):
        self.log(f"\n ✅ {message}")
        self.log(f"Output location: {self.output_dir}")
        self.progress_bar.setVisible(False)
        self.seperate_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
    
    def on_error(self, message):
        self.log(f"\n ❌ {message}")
        self.progress_bar.setVisible(False)
        self.seperate_btn.setEnabled(True)
        self.select_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window= VocalSeparatorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
                       