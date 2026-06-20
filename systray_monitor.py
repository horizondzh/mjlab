#!/usr/bin/env python3
"""System tray resource monitor: GPU / CPU / RAM — updates every 2s."""
import subprocess, threading, time
import pystray, psutil
from PIL import Image, ImageDraw

def gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], timeout=5).decode().strip().split(",")
        return {
            "gpu": int(out[0].strip()),
            "vram_used": int(out[1].strip()),
            "vram_total": int(out[2].strip()),
            "temp": int(out[3].strip()),
        }
    except Exception:
        return {"gpu": -1}

def make_icon(text):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 63, 63], fill=(30, 30, 30), outline=(80, 80, 80))
    lines = text.split("\n")
    y = 4
    for line in lines:
        d.text((4, y), line, fill="white")
        y += 14
    return img

def update(icon):
    while True:
        gpu = gpu_info()
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        if gpu["gpu"] >= 0:
            lines = [
                f"G {gpu['gpu']}% {gpu['temp']}°",
                f"V {gpu['vram_used']}/{gpu['vram_total']}",
                f"C {cpu}%",
                f"M {mem}%",
            ]
        else:
            lines = [f"CPU {cpu}%", f"RAM {mem}%", "GPU: N/A"]

        icon.icon = make_icon("\n".join(lines))
        icon.title = f"GPU:{gpu.get('gpu','?')}% CPU:{cpu}% RAM:{mem}%"
        time.sleep(2)

def main():
    icon = pystray.Icon("monitor", make_icon("..."), "Resource Monitor")
    threading.Thread(target=update, args=(icon,), daemon=True).start()
    icon.run()

if __name__ == "__main__":
    main()
