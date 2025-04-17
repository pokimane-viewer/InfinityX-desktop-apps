#!/usr/bin/env python3
import sys, os, datetime, warnings, numpy as np, tensorflow as tf, imghdr
from PIL import Image

def test_webp(h, f):
    if h[:4] == b'RIFF' and h[8:12] == b'WEBP':
        return 'webp'

def test_avif(h, f):
    if len(h) >= 12 and h[4:8] == b'ftyp' and h[8:12] in (b'avif', b'avis'):
        return 'avif'

def test_heic(h, f):
    if len(h) >= 12 and h[4:8] == b'ftyp' and h[8:12] in (b'heic', b'heix', b'mif1', b'msf1'):
        return 'heic'

imghdr.tests.insert(0, test_heic)
imghdr.tests.insert(0, test_avif)
imghdr.tests.append(test_webp)

if len(sys.argv) < 3:
    print("Usage: python style_transfer.py content_image style_image")
    sys.exit(1)

CONTENT_PATH = sys.argv[1]
STYLE_PATH = sys.argv[2]
STYLE_WEIGHT = 1e-2
CONTENT_WEIGHT = 1e4
TOTAL_VARIATION_WEIGHT = 30

def check_and_fix_extension(path):
    data = tf.io.read_file(path)
    fmt = imghdr.what(None, data.numpy())
    if not fmt or fmt not in ("jpeg", "png", "gif", "bmp", "webp", "avif", "heic"):
        raise ValueError(
            f"Unknown or unsupported image format for '{path}'. TensorFlow requires JPEG, PNG, GIF, BMP, WebP, AVIF or HEIC."
        )
    supported_formats = ("jpeg", "png", "gif", "bmp", "webp")
    if fmt in ("avif", "heic"):
        new_path = os.path.splitext(path)[0] + ".png"
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB").save(new_path, "PNG")
        print(f"Converted '{path}' from {fmt} to PNG as '{new_path}'.")
        return new_path
    correct_ext = f".{fmt}"
    current_ext = os.path.splitext(path)[1].lower()
    if current_ext != correct_ext:
        new_path = os.path.splitext(path)[0] + correct_ext
        os.rename(path, new_path)
        print(f"Renamed '{path}' -> '{new_path}' to match detected format '{fmt}'.")
        return new_path
    return path

def load_img(path):
    img_data = tf.io.read_file(path)
    img = tf.image.decode_image(img_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    max_dim = 512
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]

CONTENT_PATH = check_and_fix_extension(CONTENT_PATH)
STYLE_PATH = check_and_fix_extension(STYLE_PATH)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)

content_layers = ['block5_conv2']
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
num_style_layers = len(style_layers)
extractor = vgg_layers(style_layers + content_layers)

def style_content_loss(outputs, style_targets, content_targets):
    s = tf.add_n([tf.reduce_mean((outputs['style'][n] - style_targets[n])**2) for n in outputs['style'].keys()]) / num_style_layers
    c = tf.add_n([tf.reduce_mean((outputs['content'][n] - content_targets[n])**2) for n in outputs['content'].keys()])
    return STYLE_WEIGHT * s + CONTENT_WEIGHT * c

def gram_matrix(x):
    r = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    sh = tf.shape(x)
    return r / tf.cast(sh[1] * sh[2], tf.float32)

def call_model(img):
    p = tf.keras.applications.vgg19.preprocess_input(img * 255.0)
    o = extractor(p)
    style_out, content_out = (o[:num_style_layers], o[num_style_layers:])
    style_out = [gram_matrix(x) for x in style_out]
    return {
        'content': {content_layers[i]: v for i, v in enumerate(content_out)},
        'style': {style_layers[i]: v for i, v in enumerate(style_out)}
    }

content_image = load_img(CONTENT_PATH)
style_image = load_img(STYLE_PATH)

style_targets = call_model(style_image)['style']
content_targets = call_model(content_image)['content']

image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02)

@tf.function
def train_step(img):
    with tf.GradientTape() as t:
        out = call_model(img)
        sc_loss = style_content_loss(out, style_targets, content_targets)
        tv_loss = TOTAL_VARIATION_WEIGHT * tf.image.total_variation(img)
        loss = sc_loss + tv_loss
    grad = t.gradient(loss, img)
    opt.apply_gradients([(grad, img)])
    img.assign(tf.clip_by_value(img, 0.0, 1.0))

print(f"Content Image: {CONTENT_PATH}, Shape: {content_image.shape}")
print(f"Style Image: {STYLE_PATH}, Shape: {style_image.shape}")
print("Starting training...")

for i in range(1, 501):
    train_step(image)
    if i % 50 == 0:
        print(f"Step {i}/500 complete")

final_img = tf.squeeze(image, axis=0)
final_img = tf.image.convert_image_dtype(final_img, dtype=tf.uint8)
encoded_img = tf.io.encode_png(final_img)

content_name = os.path.splitext(os.path.basename(CONTENT_PATH))[0]
style_name = os.path.splitext(os.path.basename(STYLE_PATH))[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"stylized_{content_name}_with_{style_name}_{timestamp}.png"

tf.io.write_file(output_filename, encoded_img)
print(f"Output saved as '{output_filename}'")