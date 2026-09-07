"""Independent native desktop window process for live CFD visualization.

Runs strictly in CPU space with Tkinter/PIL without initializing CUDA or JAX,
ensuring 0 bytes of GPU memory overhead.
"""

import json
from multiprocessing.shared_memory import SharedMemory
import os
import queue
import sys
import threading
import time

MODE_NAMES = {
    0: "Numerical Schlieren log10(1 + 15*|∇ρ|/ρ_ref)",
    1: "Density ρ [kg/m³]",
    2: "Pressure p [Pa]",
    3: "Mach Number",
    4: "Temperature T [K]",
}


def run_viewer(
    shm_name: str,
    width: int,
    height: int,
    title: str = "JAX-Fluid-Sim: Live GPU Monitor",
):
    """Run native Wayland/X11 window reading frame buffers from POSIX shared memory."""
    # Ensure pure software CPU rendering with 0 bytes GPU VRAM
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
    os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"

    try:
        import tkinter as tk
        from PIL import Image, ImageTk
        import numpy as np
    except ImportError as exc:
        sys.stderr.write(f"[LiveGUI] Could not import GUI dependencies: {exc}\n")
        sys.stderr.flush()
        return

    try:
        shm = SharedMemory(name=shm_name)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
    except Exception as exc:
        sys.stderr.write(f"[LiveGUI] Failed to attach to shared memory {shm_name}: {exc}\n")
        sys.stderr.flush()
        return

    rgba_buffer = np.ndarray((height, width, 4), dtype=np.uint8, buffer=shm.buf)

    root = tk.Tk()
    root.title(title)
    root.configure(bg="#01051f")
    root.resizable(False, False)

    # Top telemetry header
    header_frame = tk.Frame(root, bg="#01051f", padx=12, pady=6)
    header_frame.pack(fill=tk.X)

    title_label = tk.Label(
        header_frame,
        text="JAX-Fluid-Sim | Hypersonic DG/FV Monitor",
        font=("Helvetica", 12, "bold"),
        fg="#ffffff",
        bg="#01051f",
    )
    title_label.pack(side=tk.LEFT)

    telemetry_label = tk.Label(
        header_frame,
        text="Initializing GPU pipeline...",
        font=("Monospace", 10),
        fg="#00a9cf",
        bg="#01051f",
    )
    telemetry_label.pack(side=tk.RIGHT)

    # Canvas for RGBA image
    canvas = tk.Canvas(
        root,
        width=width,
        height=height,
        bg="#01051f",
        highlightthickness=1,
        highlightbackground="#20d49a",
    )
    canvas.pack()

    # Initial placeholder image
    blank = Image.new("RGBA", (width, height), (1, 5, 31, 255))
    current_photo = ImageTk.PhotoImage(blank)
    img_item = canvas.create_image(0, 0, anchor="nw", image=current_photo)

    # Bottom hotkeys bar
    footer_frame = tk.Frame(root, bg="#01051f", padx=12, pady=4)
    footer_frame.pack(fill=tk.X)

    mode_label = tk.Label(
        footer_frame,
        text="Mode: [1] Schlieren",
        font=("Helvetica", 10, "bold"),
        fg="#20d49a",
        bg="#01051f",
    )
    mode_label.pack(side=tk.LEFT)

    help_label = tk.Label(
        footer_frame,
        text="[1] Schlieren  [2] Density  [3] Pressure  [4] Mach  [5] Temp  |  [Q] Close",
        font=("Helvetica", 9),
        fg="#888888",
        bg="#01051f",
    )
    help_label.pack(side=tk.RIGHT)

    # Keyboard hotkey bindings
    def set_mode(mode_id: int):
        try:
            sys.stdout.write(f"mode {mode_id}\n")
            sys.stdout.flush()
            mode_label.configure(text=f"Mode: [{mode_id+1}] {MODE_NAMES.get(mode_id, '')[:16]}")
        except Exception:
            pass

    def on_close():
        try:
            sys.stdout.write("closed\n")
            sys.stdout.flush()
        except Exception:
            pass
        root.destroy()

    root.bind("<Key-1>", lambda e: set_mode(0))
    root.bind("<Key-2>", lambda e: set_mode(1))
    root.bind("<Key-3>", lambda e: set_mode(2))
    root.bind("<Key-4>", lambda e: set_mode(3))
    root.bind("<Key-5>", lambda e: set_mode(4))
    root.bind("<Key-q>", lambda e: on_close())
    root.bind("<Escape>", lambda e: on_close())
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Reader thread for stdin commands from solver
    msg_queue = queue.Queue(maxsize=4)

    def stdin_reader():
        for line in sys.stdin:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                try:
                    msg_queue.put_nowait(data)
                except queue.Full:
                    pass
            except Exception:
                pass
        # Stdin closed -> terminate
        try:
            msg_queue.put_nowait({"cmd": "STOP"})
        except Exception:
            pass

    reader_thread = threading.Thread(target=stdin_reader, daemon=True)
    reader_thread.start()

    # Inform parent solver that GUI is ready
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # Poll frame queue smoothly at ~60 Hz
    nonlocal_holder = {"photo": current_photo, "last_frame_time": time.perf_counter()}

    def poll_queue():
        latest_msg = None
        while True:
            try:
                msg = msg_queue.get_nowait()
                if msg.get("cmd") == "STOP":
                    root.destroy()
                    return
                latest_msg = msg
            except queue.Empty:
                break

        if latest_msg is not None and latest_msg.get("cmd") == "FRAME":
            now = time.perf_counter()
            fps = 1.0 / max(now - nonlocal_holder["last_frame_time"], 1.0e-5)
            nonlocal_holder["last_frame_time"] = now

            img = Image.fromarray(rgba_buffer)
            nonlocal_holder["photo"] = ImageTk.PhotoImage(img)
            canvas.itemconfig(img_item, image=nonlocal_holder["photo"])

            # Update telemetry text
            t = latest_msg.get("time", 0.0)
            step = latest_msg.get("step", 0)
            mach = latest_msg.get("mach", 0.0)
            cfl_dt = latest_msg.get("dt", 0.0)
            troubled = latest_msg.get("troubled", 0)
            chem = latest_msg.get("chem", "")

            telem_text = (
                f"Mach {mach:.1f} | t = {t*1e3:.3f} ms | Step {step} | "
                f"CFL dt = {cfl_dt:.2e} | Trb: {troubled} | GUI: {fps:.0f} FPS {chem}"
            )
            telemetry_label.configure(text=telem_text)

        root.after(16, poll_queue)

    root.after(16, poll_queue)
    try:
        root.mainloop()
    finally:
        try:
            shm.close()
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        shm_name = sys.argv[1]
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        title = sys.argv[4] if len(sys.argv) > 4 else "JAX-Fluid-Sim: Live GPU Monitor"
        run_viewer(shm_name, width, height, title)
