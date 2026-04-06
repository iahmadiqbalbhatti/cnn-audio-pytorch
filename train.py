import sys

import modal

app = modal.App("audio-cnn-pytorch")

image = (modal.Image.debian_slim().pip_install_from_requirements("requirements.txt").apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"]).run_commands([
    "cd /temp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
    "cd /temp && unzip esc50.zip",
    "mkdir -p /opt/esc50-data",
    "cp -r /temp/ESC-50-master/* /opt/esc50-data/",
    "rm -rf /temp/esc50.zip /temp/ESC-50-master",
    ])).add_local_python_source("model")

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)

model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


@app.function()
def train(image=image, gpu="T4", volumes= {"/data":volume, "/models":model_volume}, timeout=3600):
    print("Training model...")


@app.local_entrypoint()
def main():
    train.remote()