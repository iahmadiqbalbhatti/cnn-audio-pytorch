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

modal_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i

@app.local_entrypoint()
def main():
    # run the function locally
    print(f.local(10))

    # run the function remotely on Modal
    print(f.remote(10))

    # run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(10)):
        total += ret

    print(total)

