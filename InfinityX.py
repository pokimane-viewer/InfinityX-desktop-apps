#!/usr/bin/env python3
import sys, os, datetime, warnings, numpy as np, imghdr, uuid, argparse, threading, math, subprocess, shutil, json, re, time, requests, queue, keyring
from PIL import Image, ImageEnhance, ImageFilter, ImageTk
from sklearn.utils import check_random_state
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
SERVICE_NAME="InfinityX"
CONFIG_PATH=os.path.expanduser("~/.infinityx_config.json")
API_KEY_RE=re.compile(r'^sk-[A-Za-z0-9_-]{20,}$')

def load_api_key():
    env=os.getenv("OPENAI_API_KEY","").strip()
    if env and API_KEY_RE.match(env):
        return env
    try:
        k=keyring.get_password(SERVICE_NAME,"api_key")
        if k:
            return k
    except Exception:
        pass
    try:
        with open(CONFIG_PATH,"r") as f:
            return json.load(f).get("api_key","")
    except Exception:
        return ""

def save_api_key(k):
    if not k or not API_KEY_RE.match(k):
        return
    try:
        keyring.set_password(SERVICE_NAME,"api_key",k)
        return
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH),exist_ok=True)
        with open(CONFIG_PATH,"w") as f:
            json.dump({"api_key":k},f)
    except Exception:
        pass

try:
    RESAMPLE=Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE=Image.ANTIALIAS

def _test_webp(h,f):
    return 'webp' if h[:4]==b'RIFF' and h[8:12]==b'WEBP' else None

def _test_avif(h,f):
    return 'avif' if len(h)>=12 and h[4:8]==b'ftyp' and h[8:12] in (b'avif',b'avis') else None

def _test_heic(h,f):
    return 'heic' if len(h)>=12 and h[4:8]==b'ftyp' and h[8:12] in (b'heic',b'heix',b'mif1',b'msf1') else None

imghdr.tests = [_test_heic, _test_avif] + imghdr.tests + [_test_webp]

def check_and_fix_extension(path, compress=False):
    with open(path,"rb") as f:
        header = f.read(32)
    fmt = imghdr.what(None, header)
    if fmt not in ("jpeg","png","gif","bmp","webp","avif","heic"):
        raise ValueError(f"Unsupported format {path}")
    if fmt in ("avif","heic"):
        path2 = os.path.splitext(path)[0] + ".png"
        Image.open(path).convert("RGB").save(path2,"PNG")
        path, fmt = path2, "png"
    if compress:
        if fmt != "jpeg":
            path2 = os.path.splitext(path)[0] + ".jpeg"
            Image.open(path).convert("RGB").save(path2,"JPEG",quality=85)
            path = path2
        else:
            Image.open(path).save(path,"JPEG",optimize=True,quality=65)
    return path

def remove_background(img):
    img = img.convert("RGBA")
    data = [(255,255,255,0) if all(c>240 for c in p[:3]) else p for p in img.getdata()]
    img.putdata(data)
    return img

def _parse_crop(crop_str):
    size, x, y = crop_str.lower().split('+')
    w, h = size.split('x')
    return int(x), int(y), int(w), int(h)

def _apply_edits(img, opts):
    if opts.get("rotate"):
        img = img.rotate(opts["rotate"], expand=True)
    if opts.get("flip") == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif opts.get("flip") == "vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if opts.get("grayscale"):
        img = img.convert("L").convert("RGBA")
    if opts.get("crop"):
        x, y, w, h = _parse_crop(opts["crop"])
        img = img.crop((x, y, x+w, y+h))
    if opts.get("remove_bg"):
        img = remove_background(img)
    return img

def style_transfer_main(argv, log_fn=None, spin_fn=None, lock_fn=None):
    try:
        import tensorflow as tf
    except Exception as e:
        if log_fn: log_fn(str(e))
        return
    STYLE_WEIGHT, CONTENT_WEIGHT, TV_WEIGHT = 1e-2, 1e4, 30
    def _load(p, full=False):
        d = tf.io.read_file(p)
        i = tf.image.decode_image(d, channels=3)
        i = tf.image.convert_image_dtype(i, tf.float32)
        if not full:
            mx = 512
            s = tf.cast(tf.shape(i)[:-1], tf.float32)
            scale = mx / tf.reduce_max(s)
            nshape = tf.cast(s * scale, tf.int32)
            i = tf.image.resize(i, nshape)
        return i[tf.newaxis,:]
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    layers_s = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    layer_c = ['block5_conv2']
    outputs = [vgg.get_layer(n).output for n in layers_s + layer_c]
    extractor = tf.keras.Model(vgg.input, outputs)
    def gram(x):
        r = tf.linalg.einsum('bijc,bijd->bcd', x, x)
        s = tf.shape(x)
        return r / tf.cast(s[1]*s[2], tf.float32)
    def encode(i):
        p = tf.keras.applications.vgg19.preprocess_input(i * 255.0)
        o = extractor(p)
        s, c = o[:len(layers_s)], o[len(layers_s):]
        return {'style':[gram(x) for x in s], 'content':c}
    def loss(out, st, ct):
        sl = tf.add_n([(out['style'][i] - st[i])**2 for i in range(len(layers_s))]) / len(layers_s)
        cl = tf.add_n([(out['content'][0] - ct[0])**2])
        return STYLE_WEIGHT*tf.reduce_mean(sl) + CONTENT_WEIGHT*tf.reduce_mean(cl)
    ap = argparse.ArgumentParser()
    ap.add_argument("content"); ap.add_argument("styles", nargs="+")
    ap.add_argument("--rotate", type=int, default=0); ap.add_argument("--flip", choices=["horizontal","vertical"])
    ap.add_argument("--grayscale", action="store_true"); ap.add_argument("--crop"); ap.add_argument("--remove_bg", action="store_true")
    ap.add_argument("--format", choices=["PNG","JPEG","WEBP"], default="PNG"); ap.add_argument("--fullres", action="store_true")
    a = ap.parse_args(argv)
    if lock_fn: lock_fn(True)
    curr = a.content
    for s_path in a.styles:
        c_fixed, s_fixed = check_and_fix_extension(curr), check_and_fix_extension(s_path)
        c_img, s_img = _load(c_fixed, a.fullres), _load(s_fixed)
        t_style, t_content = encode(s_img)['style'], encode(c_img)['content']
        img = tf.Variable(c_img)
        opt = tf.optimizers.Adam(0.02)
        @tf.function
        def train():
            with tf.GradientTape() as tape:
                out = encode(img)
                l = loss(out, t_style, t_content) + TV_WEIGHT*tf.image.total_variation(img)
            g = tape.gradient(l, img)
            opt.apply_gradients([(g, img)])
            img.assign(tf.clip_by_value(img, 0.0, 1.0))
        for i in range(500):
            train()
            if spin_fn: spin_fn()
        fin = tf.squeeze(img, 0)
        fin = tf.image.convert_image_dtype(fin, tf.uint8)
        fin = Image.fromarray(fin.numpy(), "RGB").convert("RGBA")
        fin = _apply_edits(fin, {"rotate":a.rotate, "flip":a.flip, "grayscale":a.grayscale, "crop":a.crop, "remove_bg":a.remove_bg})
        c_name, s_name = os.path.splitext(os.path.basename(c_fixed))[0], os.path.splitext(os.path.basename(s_fixed))[0]
        out = f"stylized_{c_name}_with_{s_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out += (".png" if a.format=="PNG" else ".jpg" if a.format=="JPEG" else ".webp")
        fin.save(out, a.format)
        curr = out
        if log_fn: log_fn(out)
    if lock_fn: lock_fn(False)
    return curr

def _avg(img):
    return np.array(img.convert("L")).mean()

def _auto(img):
    f = 1.2 if _avg(img) < 130 else 1.1
    img = ImageEnhance.Contrast(img).enhance(f)
    img = ImageEnhance.Color(img).enhance(f)
    return ImageEnhance.Sharpness(img).enhance(1.1)

def _merge_imgs(paths, resize, rng, crop, same=False, full=False):
    acc, cnt, bw, bh = None, 0, None, None
    for i, p in enumerate(paths):
        if not os.path.isfile(p): sys.exit(f"{p} missing")
        im = Image.open(p).convert("RGBA")
        if full:
            if i == 0: bw, bh = im.size
            else: im = im.resize((bw, bh), RESAMPLE)
        else:
            if same and i > 0: im = im.resize((bw, bh), RESAMPLE)
            else:
                if not resize: resize = (1024, 1024)
                im = im.resize(resize, RESAMPLE); bw, bh = im.size
        arr = np.array(im, dtype=np.uint8)
        acc = arr.astype(np.int32) if acc is None else acc + arr
        cnt += 1
    avg = (acc / cnt).astype(np.uint8)
    noise = rng.normal(0, 10, avg.shape)
    out = np.clip(avg.astype(float) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(out, "RGBA")
    img = _auto(img)
    if crop:
        x, y, w, h = crop; img = img.crop((x, y, x+w, y+h))
    return img

def merge_main(argv, log_fn=None, spin_fn=None, lock_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+"); ap.add_argument("--resize"); ap.add_argument("--crop")
    ap.add_argument("--rotate", type=int); ap.add_argument("--flip", choices=["horizontal","vertical"])
    ap.add_argument("--quality", type=int, default=90); ap.add_argument("--format", choices=["PNG","JPEG","WEBP"], default="PNG")
    ap.add_argument("--brightness", type=float); ap.add_argument("--contrast", type=float); ap.add_argument("--color", type=float)
    ap.add_argument("--sharpness", type=float); ap.add_argument("--grayscale", action="store_true"); ap.add_argument("--blur", type=float)
    ap.add_argument("--sepia", action="store_true"); ap.add_argument("--auto_enhance", action="store_true"); ap.add_argument("--ai_edit", action="store_true")
    ap.add_argument("--human", action="store_true"); ap.add_argument("--remove_bg", action="store_true"); ap.add_argument("--fullres", action="store_true")
    a = ap.parse_args(argv)
    if len(a.images) < 2: return
    if lock_fn: lock_fn(True)
    rng = check_random_state(None)
    res = _parse_crop(a.resize) if a.resize and not a.human else None
    crop = _parse_crop(a.crop) if a.crop else None
    cur = a.images[0]
    for p in a.images[1:]:
        args = [cur, p, "--format", a.format, "--quality", str(a.quality)]
        if a.rotate: args += ["--rotate", str(a.rotate)]
        if a.flip: args += ["--flip", a.flip]
        if a.grayscale: args.append("--grayscale")
        if a.crop: args += ["--crop", a.crop]
        if a.remove_bg: args.append("--remove_bg")
        if a.resize: args += ["--resize", a.resize]
        if a.sepia: args.append("--sepia")
        if a.auto_enhance: args.append("--auto_enhance")
        if a.human: args.append("--human")
        cur = _merge_imgs([cur, p], res, rng, crop, a.human, a.fullres)
        tmp = f"tmp_{uuid.uuid4().hex[:8]}.png"; cur.save(tmp,"PNG"); cur = tmp
    fin = Image.open(cur)
    if a.rotate: fin = fin.rotate(a.rotate, expand=True)
    if a.flip=="horizontal": fin = fin.transpose(Image.FLIP_LEFT_RIGHT)
    elif a.flip=="vertical": fin = fin.transpose(Image.FLIP_TOP_BOTTOM)
    if a.brightness: fin = ImageEnhance.Brightness(fin).enhance(a.brightness)
    if a.contrast: fin = ImageEnhance.Contrast(fin).enhance(a.contrast)
    if a.color: fin = ImageEnhance.Color(fin).enhance(a.color)
    if a.sharpness: fin = ImageEnhance.Sharpness(fin).enhance(a.sharpness)
    if a.grayscale: fin = fin.convert("L").convert("RGBA")
    if a.blur: fin = fin.filter(ImageFilter.GaussianBlur(a.blur))
    if a.auto_enhance or a.ai_edit: fin = _auto(fin)
    if a.remove_bg: fin = remove_background(fin)
    name = f"merged_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    ext = ".png" if a.format=="PNG" else ".jpg" if a.format=="JPEG" else ".webp"
    out = name + ext
    params = {"quality":a.quality} if a.format in ("JPEG","WEBP") else {}
    fin.save(out, a.format, **params)
    if lock_fn: lock_fn(False)
    if log_fn: log_fn(out)
    return out

ALLOWED_SIZES=['256x256','512x512','1024x1024','1024x1792','1792x1024']
ERROR_SPINNER_FRAMES=["| ⛑️","/ ⛑️","- ⛑️","\\ ⛑️"]

def intelligent_filename(prompt,index):
    sanitized = re.sub(r'[^a-zA-Z0-9]+','_',prompt).strip('_')
    if len(sanitized) > 20:
        sanitized = sanitized[:20]
    timestamp = int(time.time())
    return f"{sanitized}_{timestamp}_{index}.png"

def generate_and_save_image(client,prompt_text,size_str,quality,output_dir,index,log_q,image_list,app):
    try:
        response = client.images.generate(model="dall-e-3", prompt=prompt_text, size=size_str, quality=quality, n=1)
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

def _crop_dialog(root, img_path, entry_widget):
    pass

def run_gui():
    root = tk.Tk()
    root.title("InfinityX")
    try: root.iconphoto(False, tk.PhotoImage(file="microsoft.png"))
    except: pass
    spin_can = tk.Canvas(root, width=100, height=100, bg="white"); spin_can.pack()
    notebook = ttk.Notebook(root)
    fs, fm, fp, ff, fd = (ttk.Frame(notebook) for _ in range(5))
    for f, t in zip((fs, fm, fp, ff, fd), ("Style Transfer","Merge Images","Photo Edit","FFMPEG","DALL-E")):
        notebook.add(f, text=t)
    notebook.pack(fill="both", expand=True)
    log_txt = tk.Text(root, height=6, width=80); log_txt.pack(side=tk.BOTTOM, fill=tk.X)
    def log(m): log_txt.insert(tk.END, m+"\n"); log_txt.see(tk.END); root.update_idletasks()
    running = {"style":False,"merge":False,"photo":False,"force":False}
    ang = 0
    def spin():
        nonlocal ang
        if any(running.values()):
            ang = (ang+10)%360; spin_can.delete("all"); c, r = 50, 40
            spin_can.create_oval(c-r,c-r,c+r,c+r, fill="pink")
            spin_can.create_line(c,c, c+r*math.cos(math.radians(ang)), c+r*math.sin(math.radians(ang)), width=4)
        else:
            spin_can.delete("all")
        root.after(200, spin)
    spin()
    def lock(k, s): running[k] = s
    def spinner_on(): running["force"] = True
    def spinner_off(): running["force"] = False
    def lock_merge(s): lock("merge", s)
    def lock_photo(s): lock("photo", s)
    style_last_out = [None]; merge_last_out = [None]; photo_last_out = [None]
    def download_file(path):
        if not path: return
        dst_dir = os.path.expanduser('~/Downloads'); os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(path)); shutil.copy(path, dst); log(f"Saved {dst}")

    ttk.Label(fs, text="Content:").grid(row=0, column=0, sticky="e")
    ent_content = ttk.Entry(fs, width=50); ent_content.grid(row=0, column=1)
    prev_content = ttk.Label(fs); prev_content.grid(row=0, column=3)
    btn_dl_style = ttk.Button(fs, text="Download", state="disabled",
                              command=lambda: download_file(style_last_out[0]))
    btn_dl_style.grid(row=0, column=4, padx=4)
    ttk.Label(fs, text="Style:").grid(row=1, column=0, sticky="e")
    ent_style = ttk.Entry(fs, width=50); ent_style.grid(row=1, column=1)

    def _browse(entry, cb=None):
        p = filedialog.askopenfilename()
        if p:
            p = check_and_fix_extension(p)
            entry.delete(0, tk.END); entry.insert(0, p)
            if cb:
                img = Image.open(p); img.thumbnail((300,300)); photo = ImageTk.PhotoImage(img)
                cb(photo)

    ttk.Button(fs, text="Browse", command=lambda: _browse(ent_content,
        lambda ph:(prev_content.config(image=ph), setattr(prev_content,"image",ph)))).grid(row=0, column=2)
    ttk.Button(fs, text="Browse", command=lambda: _browse(ent_style)).grid(row=1, column=2)

    lfo = ttk.LabelFrame(fs, text="Options"); lfo.grid(row=2, column=0, columnspan=5, sticky="ew")
    ttk.Label(lfo, text="Rotate").grid(row=0, column=0); ent_rot = ttk.Entry(lfo, width=5); ent_rot.insert(0,"0"); ent_rot.grid(row=0, column=1)
    flip = tk.StringVar(value="none"); ttk.Label(lfo, text="Flip").grid(row=0, column=2)
    for i,v in enumerate(("none","horizontal","vertical")): ttk.Radiobutton(lfo, text=v, variable=flip, value=v).grid(row=0, column=3+i)
    gray = tk.BooleanVar(); ttk.Checkbutton(lfo, text="Grayscale", variable=gray).grid(row=1, column=0)
    ttk.Label(lfo, text="Crop").grid(row=1, column=1); ent_crop = ttk.Entry(lfo, width=15); ent_crop.grid(row=1, column=2)
    ttk.Button(lfo, text="Crop", command=lambda: _crop_dialog(root, ent_content.get(), ent_crop)).grid(row=1, column=3)
    rembg = tk.BooleanVar(); ttk.Checkbutton(lfo, text="Remove BG", variable=rembg).grid(row=1, column=4)
    fmt = tk.StringVar(value="PNG"); ttk.OptionMenu(lfo, fmt, "PNG", "PNG","JPEG","WEBP").grid(row=1, column=5)
    full = tk.BooleanVar(); ttk.Checkbutton(lfo, text="FullRes", variable=full).grid(row=0, column=6)

    def run_style():
        def task():
            lock("style", True)
            args=[ent_content.get(), ent_style.get(), "--format", fmt.get()]
            if ent_rot.get()!="0": args+=["--rotate", ent_rot.get()]
            if flip.get()!="none": args+=["--flip", flip.get()]
            if gray.get(): args.append("--grayscale")
            if ent_crop.get(): args+=["--crop", ent_crop.get()]
            if rembg.get(): args.append("--remove_bg")
            if full.get(): args.append("--fullres")
            out = style_transfer_main(args, log, lambda:None, lambda s:lock("style", s))
            if out:
                style_last_out[0]=out; btn_dl_style.config(state="normal")
                im = Image.open(out); im.thumbnail((300,300)); ph = ImageTk.PhotoImage(im)
                prev_content.config(image=ph); prev_content.image=ph
            lock("style", False)
        threading.Thread(target=task).start()

    ttk.Button(fs, text="Run", command=run_style).grid(row=3, column=0, columnspan=5)

    ttk.Label(fm, text="Primary").grid(row=0, column=0, sticky="e")
    ent_primary = ttk.Entry(fm, width=50); ent_primary.grid(row=0, column=1)
    lbl_primary = ttk.Label(fm); lbl_primary.grid(row=0, column=3)
    btn_dl_merge = ttk.Button(fm, text="Download", state="disabled",
                              command=lambda: download_file(merge_last_out[0]))
    btn_dl_merge.grid(row=0, column=4, padx=4)
    ttk.Button(fm, text="Browse", command=lambda: _browse(ent_primary,
        lambda ph:(lbl_primary.config(image=ph), setattr(lbl_primary,"image",ph)))).grid(row=0, column=2)
    ttk.Label(fm, text="Additional").grid(row=1, column=0, sticky="ne")
    txt_imgs = tk.Text(fm, width=38, height=4); txt_imgs.grid(row=1, column=1, rowspan=2)
    def add_imgs(): paths = filedialog.askopenfilenames(); txt_imgs.insert(tk.END, '\n'.join(paths)+"\n")
    ttk.Button(fm, text="Add", command=add_imgs).grid(row=1, column=2, sticky="n")

    optm = ttk.LabelFrame(fm, text="Options"); optm.grid(row=3, column=0, columnspan=5, sticky="ew")
    ent_mrot = ttk.Entry(optm, width=5); ent_mrot.insert(0,"0")
    ttk.Label(optm, text="Rotate").grid(row=0, column=0); ent_mrot.grid(row=0, column=1)
    mflip = tk.StringVar(value="none"); ttk.Label(optm, text="Flip").grid(row=0, column=2)
    for i,v in enumerate(("none","horizontal","vertical")): ttk.Radiobutton(optm, text=v, variable=mflip, value=v).grid(row=0, column=3+i)
    mgray = tk.BooleanVar(); ttk.Checkbutton(optm, text="Gray", variable=mgray).grid(row=1, column=0)
    ttk.Label(optm, text="Crop").grid(row=1, column=1); ent_mcrop = ttk.Entry(optm, width=15); ent_mcrop.grid(row=1, column=2)
    ttk.Label(optm, text="Resize").grid(row=1, column=3); ent_mresize = ttk.Entry(optm, width=10); ent_mresize.grid(row=1, column=4)
    ttk.Label(optm, text="Quality").grid(row=1, column=5); ent_mquality = ttk.Entry(optm, width=5); ent_mquality.insert(0,"90"); ent_mquality.grid(row=1, column=6)
    mrem = tk.BooleanVar(); ttk.Checkbutton(optm, text="Remove BG", variable=mrem).grid(row=2, column=0)
    msep = tk.BooleanVar(); ttk.Checkbutton(optm, text="Sepia", variable=msep).grid(row=2, column=1)
    mai = tk.BooleanVar(); ttk.Checkbutton(optm, text="AI Enhance", variable=mai).grid(row=2, column=2)
    mhum = tk.BooleanVar(); ttk.Checkbutton(optm, text="Human", variable=mhum).grid(row=2, column=3)
    mf = tk.StringVar(value="PNG"); ttk.OptionMenu(optm, mf, "PNG", "PNG","JPEG","WEBP").grid(row=2, column=4)

    def run_merge():
        def task():
            spinner_on(); lock_merge(True)
            primary = ent_primary.get(); extras = [p for p in txt_imgs.get("1.0", tk.END).splitlines() if p]
            cur = primary
            for p in extras:
                args=[cur, p, "--format", mf.get(), "--quality", ent_mquality.get()]
                if ent_mrot.get()!="0": args+=["--rotate", ent_mrot.get()]
                if mflip.get()!="none": args+=["--flip", mflip.get()]
                if mgray.get(): args.append("--grayscale")
                if ent_mcrop.get(): args+=["--crop", ent_mcrop.get()]
                if mrem.get(): args.append("--remove_bg")
                if ent_mresize.get(): args+=["--resize", ent_mresize.get()]
                if msep.get(): args.append("--sepia")
                if mai.get(): args.append("--auto_enhance")
                if mhum.get(): args.append("--human")
                cur = merge_main(args, log, lambda:None, lambda s:lock_merge(s))
            ent_primary.delete(0, tk.END); ent_primary.insert(0, cur)
            merge_last_out[0] = cur; btn_dl_merge.config(state="normal")
            im = Image.open(cur); im.thumbnail((300,300)); ph = ImageTk.PhotoImage(im)
            lbl_primary.config(image=ph); lbl_primary.image=ph
            spinner_off(); lock_merge(False)
        threading.Thread(target=task).start()

    ttk.Button(fm, text="Run Merge", command=run_merge).grid(row=4, column=0, columnspan=5)

    ttk.Label(fp, text="Photo").grid(row=0, column=0, sticky="e")
    ent_photo = ttk.Entry(fp, width=50); ent_photo.grid(row=0, column=1)
    lbl_photo = ttk.Label(fp); lbl_photo.grid(row=1, column=0, columnspan=3)
    btn_dl_photo = ttk.Button(fp, text="Download", state="disabled", command=lambda: download_file(photo_last_out[0]))
    btn_dl_photo.grid(row=0, column=3, padx=4)
    ttk.Button(fp, text="Browse", command=lambda: _browse(ent_photo,
        lambda ph:(lbl_photo.config(image=ph), setattr(lbl_photo,"image",ph)))).grid(row=0, column=2)

    optp = ttk.LabelFrame(fp, text="Edit"); optp.grid(row=2, column=0, columnspan=4, sticky="ew")
    ttk.Label(optp, text="Rotate").grid(row=0, column=0); ent_prot = ttk.Entry(optp, width=5); ent_prot.insert(0,"0"); ent_prot.grid(row=0, column=1)
    pflip = tk.StringVar(value="none"); ttk.Label(optp, text="Flip").grid(row=0, column=2)
    for i,v in enumerate(("none","horizontal","vertical")): ttk.Radiobutton(optp, text=v, variable=pflip, value=v).grid(row=0, column=3+i)
    pgray = tk.BooleanVar(); ttk.Checkbutton(optp, text="Gray", variable=pgray).grid(row=1, column=0)
    ttk.Label(optp, text="Crop").grid(row=1, column=1); ent_pcrop = ttk.Entry(optp, width=15); ent_pcrop.grid(row=1, column=2)
    ttk.Button(optp, text="Crop", command=lambda: _crop_dialog(root, ent_photo.get(), ent_pcrop)).grid(row=1, column=3)
    prem = tk.BooleanVar(); ttk.Checkbutton(optp, text="Remove BG", variable=prem).grid(row=1, column=4)
    ttk.Label(optp, text="Brightness").grid(row=2, column=0); ent_pbright = ttk.Entry(optp, width=5); ent_pbright.grid(row=2, column=1)
    ttk.Label(optp, text="Contrast").grid(row=2, column=2); ent_pcont = ttk.Entry(optp, width=5); ent_pcont.grid(row=2, column=3)
    ttk.Label(optp, text="Color").grid(row=3, column=0); ent_pcol = ttk.Entry(optp, width=5); ent_pcol.grid(row=3, column=1)
    ttk.Label(optp, text="Sharp").grid(row=3, column=2); ent_psharp = ttk.Entry(optp, width=5); ent_psharp.grid(row=3, column=3)
    ttk.Label(optp, text="Blur").grid(row=4, column=0); ent_pblur = ttk.Entry(optp, width=5); ent_pblur.grid(row=4, column=1)
    psep = tk.BooleanVar(); ttk.Checkbutton(optp, text="Sepia", variable=psep).grid(row=4, column=2)
    peny = tk.BooleanVar(); ttk.Checkbutton(optp, text="Auto", variable=peny).grid(row=4, column=3)
    pf = tk.StringVar(value="PNG"); ttk.OptionMenu(optp, pf, "PNG", "PNG","JPEG","WEBP").grid(row=5, column=0)
    ttk.Label(optp, text="Resize").grid(row=5, column=1); ent_pres = ttk.Entry(optp, width=8); ent_pres.grid(row=5, column=2)
    ttk.Label(optp, text="Quality").grid(row=5, column=3); ent_pq = ttk.Entry(optp, width=5); ent_pq.insert(0,"90"); ent_pq.grid(row=5, column=4)

    def run_photo():
        def task():
            spinner_on(); lock_photo(True)
            p = ent_photo.get(); img = Image.open(p)
            opts = {"rotate":int(ent_prot.get()), "flip":pflip.get(), "grayscale":pgray.get(),
                    "crop":ent_pcrop.get() or None, "remove_bg":prem.get()}
            img = _apply_edits(img, opts)
            if ent_pbright.get(): img = ImageEnhance.Brightness(img).enhance(float(ent_pbright.get()))
            if ent_pcont.get(): img = ImageEnhance.Contrast(img).enhance(float(ent_pcont.get()))
            if ent_pcol.get(): img = ImageEnhance.Color(img).enhance(float(ent_pcol.get()))
            if ent_psharp.get(): img = ImageEnhance.Sharpness(img).enhance(float(ent_psharp.get()))
            if ent_pblur.get(): img = img.filter(ImageFilter.GaussianBlur(float(ent_pblur.get())))
            if psep.get():
                arr = np.array(img.convert("RGB")); r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                tr = 0.393*r + 0.769*g + 0.189*b; tg = 0.349*r + 0.686*g + 0.168*b; tb = 0.272*r + 0.534*g + 0.131*b
                arr[:,:,0], arr[:,:,1], arr[:,:,2] = np.clip(tr,0,255), np.clip(tg,0,255), np.clip(tb,0,255)
                img = Image.fromarray(arr.astype(np.uint8), "RGB")
            if peny.get(): img = _auto(img)
            if ent_pres.get():
                w,h = map(int, ent_pres.get().split('x')); img = img.resize((w,h), RESAMPLE)
            ext = ".png" if pf.get()=="PNG" else ".jpg" if pf.get()=="JPEG" else ".webp"
            out = os.path.splitext(p)[0] + "_edited" + ext
            prm = {"quality":int(ent_pq.get())} if pf.get() in ("JPEG","WEBP") else {}
            img.save(out, pf.get(), **prm)
            photo_last_out[0] = out; btn_dl_photo.config(state="normal")
            im = Image.open(out); im.thumbnail((300,300)); ph = ImageTk.PhotoImage(im)
            lbl_photo.config(image=ph); lbl_photo.image=ph; log(out)
            spinner_off(); lock_photo(False)
        threading.Thread(target=task).start()

    ttk.Button(fp, text="Apply", command=run_photo).grid(row=3, column=0, columnspan=4)

    ttk.Label(ff, text="Input").grid(row=0, column=0, sticky="e"); ent_in = ttk.Entry(ff, width=50); ent_in.grid(row=0, column=1)
    ttk.Button(ff, text="Browse", command=lambda: _browse(ent_in)).grid(row=0, column=2)
    ttk.Label(ff, text="Output").grid(row=1, column=0, sticky="e"); ent_out = ttk.Entry(ff, width=50); ent_out.grid(row=1, column=1)
    cmd = tk.StringVar(value="Convert MP4"); ttk.OptionMenu(ff, cmd, "Convert MP4","Convert MP4","Extract Audio","Resize","Custom").grid(row=2, column=1, sticky="w")
    ttk.Label(ff, text="Resolution").grid(row=3, column=0, sticky="e"); ent_res = ttk.Entry(ff, width=10); ent_res.grid(row=3, column=1, sticky="w")
    ttk.Label(ff, text="Params").grid(row=4, column=0, sticky="e"); ent_params = ttk.Entry(ff, width=50); ent_params.grid(row=4, column=1, columnspan=2, sticky="w")

    def run_ff():
        inp, outp = ent_in.get(), ent_out.get()
        if not os.path.isfile(inp): messagebox.showerror("Err","Invalid input"); return
        if not outp:
            base = os.path.splitext(inp)[0]
            outp = base + ("_out.mp4" if cmd.get()=="Convert MP4" else "_audio.mp3" if cmd.get()=="Extract Audio" else "_resized.mp4")
            ent_out.insert(0, outp)
        if cmd.get()=="Convert MP4": c = ["ffmpeg","-i",inp,outp]
        elif cmd.get()=="Extract Audio": c = ["ffmpeg","-i",inp,"-q:a","0","-map","a",outp]
        elif cmd.get()=="Resize":
            if not ent_res.get(): messagebox.showerror("Err","Set resolution"); return
            c = ["ffmpeg","-i",inp,"-vf",f"scale={ent_res.get()}",outp]
        else:
            if not ent_params.get(): messagebox.showerror("Err","Set params"); return
            c = ["ffmpeg","-i",inp] + ent_params.get().split() + [outp]
        try:
            subprocess.run(c,check=True)
            messagebox.showinfo("Done"," ".join(c))
        except subprocess.CalledProcessError:
            messagebox.showerror("FFMPEG","Command failed")

    ttk.Button(ff, text="Run", command=run_ff).grid(row=5, column=0, columnspan=3)

    log_q = queue.Queue(); error_spinners = {}
    tk.Label(fd, text="API Key:").grid(row=0, column=0, sticky="e"); ent_api = tk.Entry(fd, width=50, show="*"); ent_api.grid(row=0, column=1)
    ent_api.insert(0, load_api_key())
    tk.Label(fd, text="Prompt:").grid(row=1, column=0, sticky="ne"); txt_prompt = tk.Text(fd, width=40, height=5); txt_prompt.grid(row=1, column=1, columnspan=2)
    tk.Label(fd, text="Size:").grid(row=2, column=0, sticky="e"); size_var = tk.StringVar(value=ALLOWED_SIZES[2]); tk.OptionMenu(fd, size_var, *ALLOWED_SIZES).grid(row=2, column=1, sticky="w")
    tk.Label(fd, text="Quality:").grid(row=3, column=0, sticky="e"); ent_quality = tk.Entry(fd, width=10); ent_quality.insert(0,"standard"); ent_quality.grid(row=3, column=1, sticky="w")
    tk.Label(fd, text="Versions:").grid(row=4, column=0, sticky="e"); ent_parallel = tk.Entry(fd, width=5); ent_parallel.insert(0,"1"); ent_parallel.grid(row=4, column=1, sticky="w")
    btn_gen = tk.Button(fd, text="Generate"); btn_gen.grid(row=5, column=1, pady=5)
    spin_lbl = tk.Label(fd, text=""); spin_lbl.grid(row=5, column=2)
    log_txt_d = tk.Text(fd, height=10, width=70); log_txt_d.grid(row=6, column=0, columnspan=3)
    btn_prev = tk.Button(fd, text="<<", state="disabled"); btn_prev.grid(row=7, column=0)
    img_lbl = tk.Label(fd, text="No Image", bg="gray"); img_lbl.grid(row=7, column=1)
    btn_next = tk.Button(fd, text=">>", state="disabled"); btn_next.grid(row=7, column=2)
    btn_dl_d = tk.Button(fd, text="Download", state="disabled"); btn_dl_d.grid(row=8, column=1)
    dalle_imgs = []; dalle_idx = [0]; spinning = [False]

    def start_spin():
        spinning[0] = True
        def _rot():
            if spinning[0]:
                spin_lbl.config(text=ERROR_SPINNER_FRAMES[int(time.time()*10)%4])
                root.after(100,_rot)
        _rot()

    def stop_spin():
        spinning[0] = False; spin_lbl.config(text="")

    def poll_log():
        try:
            while True:
                line = log_q.get_nowait(); log_txt_d.insert(tk.END, line); log_txt_d.see(tk.END)
        except queue.Empty:
            pass
        root.after(100, poll_log)

    poll_log()

    def show_img(idx):
        if dalle_imgs:
            path = dalle_imgs[idx]
            try:
                im = tk.PhotoImage(file=path)
                img_lbl.config(image=im, text="")
                img_lbl.image = im
            except Exception:
                img_lbl.config(text="Failed", image="")

    def prev_img():
        dalle_idx[0] = (dalle_idx[0] - 1) % len(dalle_imgs); show_img(dalle_idx[0])
    def next_img():
        dalle_idx[0] = (dalle_idx[0] + 1) % len(dalle_imgs); show_img(dalle_idx[0])

    btn_prev.config(command=prev_img); btn_next.config(command=next_img)

    def dl_current():
        if dalle_imgs:
            download_file(dalle_imgs[dalle_idx[0]])
    btn_dl_d.config(command=dl_current)

    def select_dalle_image(event):
        if dalle_imgs:
            path = dalle_imgs[dalle_idx[0]]
            ent_content.delete(0, tk.END); ent_content.insert(0,path)
            ent_photo.delete(0, tk.END); ent_photo.insert(0,path)
            ent_primary.delete(0, tk.END); ent_primary.insert(0,path)
            log(f"Selected {path} as main image")
    img_lbl.bind("<Button-1>", select_dalle_image)

    def gen():
        btn_gen.config(state="disabled"); start_spin()
        api = ent_api.get().strip(); prompt = txt_prompt.get("1.0", tk.END).strip()
        if not api or not prompt or not API_KEY_RE.match(api):
            messagebox.showerror("Err","Invalid API key or prompt missing"); stop_spin(); btn_gen.config(state="normal"); return
        size = size_var.get(); quality = ent_quality.get().strip() or "standard"
        try: par = int(ent_parallel.get().strip())
        except: par = 1
        client = OpenAI(api_key=api)
        save_api_key(api)
        dalle_imgs.clear(); dalle_idx[0] = 0
        def worker():
            if par>1:
                with ThreadPoolExecutor() as ex:
                    fut=[ex.submit(generate_and_save_image,client,prompt,size,quality,os.getcwd(),i,log_q,dalle_imgs,root) for i in range(1,par+1)]
                    [f.result() for f in fut]
            else:
                generate_and_save_image(client,prompt,size,quality,os.getcwd(),1,log_q,dalle_imgs,root)
            stop_spin(); btn_gen.config(state="normal")
            if dalle_imgs:
                show_img(0); btn_dl_d.config(state="normal")
                if len(dalle_imgs)>1: btn_prev.config(state="normal"); btn_next.config(state="normal")
        threading.Thread(target=worker).start()

    btn_gen.config(command=gen)
    root.mainloop()

if __name__=="__main__":
    if len(sys.argv)>1:
        cmd=sys.argv[1].lower()
        if cmd=="style": style_transfer_main(sys.argv[2:])
        elif cmd=="merge": merge_main(sys.argv[2:])
        else: run_gui()
    else:
        run_gui()