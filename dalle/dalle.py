import json
import os
import re
import time
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue

ALLOWED_SIZES = ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
ERROR_SPINNER_FRAMES = ["| ⛑️", "/ ⛑️", "- ⛑️", "\\ ⛑️"]

def intelligent_filename(prompt, index):
    sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', prompt).strip('_')
    if len(sanitized) > 20:
        sanitized = sanitized[:20]
    timestamp = int(time.time())
    return f"{sanitized}_{timestamp}_{index}.png"

def generate_and_save_image(client, prompt_text, size_str, quality, output_dir, index, log_q, image_list, app):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt_text,
            size=size_str,
            quality=quality,
            n=1,
        )
    except Exception as e:
        err_str = str(e)
        log_q.put(f"Error generating image version {index}: {err_str}\n")
        if "content_policy_violation" in err_str:
            app.show_error_spinner(index)
        return
    if response and response.data and len(response.data) > 0:
        image_url = response.data[0].url
        log_q.put(f"Image URL for version {index}: {image_url}\n")
        try:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            filename = intelligent_filename(prompt_text, index)
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "wb") as out_file:
                out_file.write(r.content)
            log_q.put(f"Image downloaded and saved to {output_path}\n")
            image_list.append(output_path)
        except Exception as e:
            log_q.put(f"Failed to download image version {index}: {e}\n")
    else:
        log_q.put(f"No image URL returned for version {index}.\n")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DALL-E 3 Image Generator")
        self.geometry("800x800")
        self.log_queue = queue.Queue()
        self.error_spinners = {}
        tk.Label(self, text="Credentials File:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.credentials_entry = tk.Entry(self, width=40)
        self.credentials_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse", command=self.browse_credentials).grid(row=0, column=2, padx=5, pady=5)
        tk.Label(self, text="Prompt Text:").grid(row=1, column=0, sticky="ne", padx=5, pady=5)
        self.prompt_text = tk.Text(self, height=5, width=40)
        self.prompt_text.grid(row=1, column=1, padx=5, pady=5, columnspan=2)
        self.prompt_text.insert("1.0", "black and white replication of the microsoft corporate logo that looks exactly like the Windows 95 logo")
        self.clear_prompt_button = tk.Button(self, text="Clear Prompt", command=self.clear_prompt)
        self.clear_prompt_button.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.copy_prompt_button = tk.Button(self, text="Copy Prompt", command=self.copy_prompt)
        self.copy_prompt_button.grid(row=2, column=2, sticky="e", padx=5, pady=5)
        tk.Label(self, text="Size:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.size_var = tk.StringVar()
        self.size_var.set(ALLOWED_SIZES[2])
        self.size_menu = tk.OptionMenu(self, self.size_var, *ALLOWED_SIZES)
        self.size_menu.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        tk.Label(self, text="Quality:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.quality_entry = tk.Entry(self, width=20)
        self.quality_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        self.quality_entry.insert(0, "standard")
        tk.Label(self, text="Parallel:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.parallel_entry = tk.Entry(self, width=5)
        self.parallel_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.parallel_entry.insert(0, "1")
        self.generate_button = tk.Button(self, text="Generate", command=self.start_generation)
        self.generate_button.grid(row=6, column=1, pady=10)
        self.spinner_label = tk.Label(self, text="")
        self.spinner_label.grid(row=6, column=2, padx=5, pady=10)
        self.log_text = tk.Text(self, height=15, width=70)
        self.log_text.grid(row=7, column=0, columnspan=3, padx=5, pady=5)
        self.prev_button = tk.Button(self, text="<<", command=self.show_prev_image, state="disabled")
        self.prev_button.grid(row=8, column=0, padx=5, pady=5)
        self.image_label = tk.Label(self, text="No Image", bg="gray")
        self.image_label.grid(row=8, column=1, padx=5, pady=5)
        self.next_button = tk.Button(self, text=">>", command=self.show_next_image, state="disabled")
        self.next_button.grid(row=8, column=2, padx=5, pady=5)
        self.download_button = tk.Button(self, text="Download", command=self.download_image, state="disabled")
        self.download_button.grid(row=9, column=1, padx=5, pady=5)
        self.error_frame = tk.Frame(self)
        self.error_frame.grid(row=10, column=0, columnspan=3, padx=5, pady=5)
        self.current_image_index = 0
        self.generated_images = []
        self.after(100, self.poll_log_queue)

    def browse_credentials(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.credentials_entry.delete(0, tk.END)
            self.credentials_entry.insert(0, path)

    def poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.poll_log_queue)

    def clear_prompt(self):
        self.prompt_text.delete("1.0", tk.END)

    def copy_prompt(self):
        prompt = self.prompt_text.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(prompt)

    def download_image(self):
        if self.generated_images:
            current_image_path = self.generated_images[self.current_image_index]
            try:
                download_dir = os.path.expanduser("~/Downloads")
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir, exist_ok=True)
                base_name = os.path.basename(current_image_path)
                destination = os.path.join(download_dir, base_name)
                shutil.copy(current_image_path, destination)
                self.log_queue.put(f"Downloaded current image to {destination}\n")
            except Exception as e:
                self.log_queue.put(f"Failed to download image: {e}\n")

    def start_generation(self):
        t = threading.Thread(target=self.generate_images)
        t.daemon = True
        t.start()

    def start_spinner(self):
        self.spinning = True
        self.spinner_frames = ["| 🎾", "/ 🎾", "- 🎾", "\\ 🎾"]
        self.current_spinner_frame = 0
        self.animate_spinner()

    def animate_spinner(self):
        if self.spinning:
            self.spinner_label.config(text=self.spinner_frames[self.current_spinner_frame])
            self.current_spinner_frame = (self.current_spinner_frame + 1) % len(self.spinner_frames)
            self.after(100, self.animate_spinner)

    def stop_spinner(self):
        self.spinning = False
        self.spinner_label.config(text="")

    def generate_images(self):
        self.generate_button.config(state="disabled")
        self.start_spinner()
        credentials_path = self.credentials_entry.get().strip()
        try:
            with open(credentials_path, "r") as f:
                credentials = json.load(f)
                api_key = credentials.get("api_key", "").strip()
                org_id = credentials.get("org_id", "").strip()
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Error reading credentials file: {e}"))
            self.after(0, self.stop_spinner)
            self.after(0, lambda: self.generate_button.config(state="normal"))
            return
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            self.after(0, lambda: messagebox.showerror("Error", "Prompt text is empty."))
            self.after(0, self.stop_spinner)
            self.after(0, lambda: self.generate_button.config(state="normal"))
            return
        size_str = self.size_var.get()
        quality = self.quality_entry.get().strip()
        try:
            parallel = int(self.parallel_entry.get().strip())
        except Exception:
            parallel = 1
        if not api_key:
            self.after(0, lambda: messagebox.showerror("Error", "API key not provided in credentials file."))
            self.after(0, self.stop_spinner)
            self.after(0, lambda: self.generate_button.config(state="normal"))
            return
        client = OpenAI(api_key=api_key, organization=org_id if org_id else None)
        output_dir = os.getcwd()
        self.generated_images = []
        if parallel > 1:
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(1, parallel + 1):
                    futures.append(executor.submit(generate_and_save_image, client, prompt, size_str, quality, output_dir, i, self.log_queue, self.generated_images, self))
                for future in futures:
                    future.result()
        else:
            generate_and_save_image(client, prompt, size_str, quality, output_dir, 1, self.log_queue, self.generated_images, self)
        self.after(0, self.on_generation_complete)

    def on_generation_complete(self):
        self.stop_spinner()
        self.generate_button.config(state="normal")
        width, height = map(int, self.size_var.get().split("x"))
        self.geometry(f"{max(800, width+100)}x{height+300}")
        if self.generated_images:
            self.current_image_index = 0
            self.show_image(self.current_image_index)
            if len(self.generated_images) > 1:
                self.prev_button.config(state="normal")
                self.next_button.config(state="normal")
            else:
                self.prev_button.config(state="disabled")
                self.next_button.config(state="disabled")
            self.download_button.config(state="normal")
        else:
            self.image_label.config(text="No Image Generated", image="")

    def show_image(self, index):
        if self.generated_images:
            path = self.generated_images[index]
            try:
                img = tk.PhotoImage(file=path)
                self.image_label.config(image=img, text="")
                self.image_label.image = img
            except Exception as e:
                self.log_queue.put(f"Failed to load image: {e}\n")

    def show_prev_image(self):
        if self.generated_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.generated_images)
            self.show_image(self.current_image_index)

    def show_next_image(self):
        if self.generated_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.generated_images)
            self.show_image(self.current_image_index)

    def show_error_spinner(self, thread_index):
        label = tk.Label(self.error_frame, text=f"Thread {thread_index}: {ERROR_SPINNER_FRAMES[0]}")
        label.pack(side="top", anchor="w")
        self.error_spinners[thread_index] = {"label": label, "frame": 0, "iterations": 0}
        self.animate_error_spinner(thread_index)

    def animate_error_spinner(self, thread_index, max_iterations=30):
        spinner = self.error_spinners.get(thread_index)
        if spinner is None:
            return
        if spinner["iterations"] < max_iterations:
            spinner["frame"] = (spinner["frame"] + 1) % len(ERROR_SPINNER_FRAMES)
            spinner["iterations"] += 1
            spinner_text = f"Thread {thread_index}: {ERROR_SPINNER_FRAMES[spinner['frame']]}"
            spinner["label"].config(text=spinner_text)
            self.after(100, lambda: self.animate_error_spinner(thread_index, max_iterations))
        else:
            spinner["label"].destroy()
            del self.error_spinners[thread_index]

if __name__ == "__main__":
    App().mainloop()