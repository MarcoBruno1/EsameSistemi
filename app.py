"""
CDSS - Sistema di Supporto alla Decisione Clinica
Classificazione di immagini endoscopiche (Kvasir Dataset)
Modello: EfficientNetB0 con fine-tuning + Grad-CAM
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import gradio as gr

# ============================================================
# CONFIGURAZIONE
# ============================================================

IMG_SIZE = (224, 224)
NUM_CLASSES = 8

CLASS_NAMES = [
    "dyed-lifted-polyps", "dyed-resection-margins", "esophagitis",
    "normal-cecum", "normal-pylorus", "normal-z-line",
    "polyps", "ulcerative-colitis"
]

# Nome dell'ultimo layer convoluzionale di EfficientNetB0, usato per il Grad-CAM
LAST_CONV_LAYER = "top_conv"

# ============================================================
# CARICAMENTO DEL MODELLO
# ============================================================

# Il file .h5 deve trovarsi nella stessa cartella di questo script
model = load_model("best_kvasir_model.h5")

# ============================================================
# FUNZIONI GRAD-CAM
# ============================================================

def genera_gradcam(img_array, model, ultimo_layer_conv=LAST_CONV_LAYER):
    """
    Genera una mappa Grad-CAM che evidenzia le regioni dell'immagine
    piu rilevanti per la decisione del modello.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(ultimo_layer_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        classe_predetta = tf.argmax(predictions[0])
        output_classe = predictions[:, classe_predetta]

    grads = tape.gradient(output_classe, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap, int(classe_predetta.numpy())


def sovrapponi_gradcam(img_originale_pil, heatmap, alpha=0.4):
    """
    Sovrappone la heatmap Grad-CAM all'immagine originale.
    """
    img_array = np.array(img_originale_pil.resize(IMG_SIZE))

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(img_array * (1 - alpha) + heatmap_color * alpha)
    return Image.fromarray(overlay)

# ============================================================
# FUNZIONE PRINCIPALE DI INFERENZA
# ============================================================

def predict_e_gradcam(input_image):
    """
    Prende l'immagine caricata dall'utente, esegue la predizione
    e genera la visualizzazione Grad-CAM.
    """
    if input_image is None:
        return None, None

    img_resized = input_image.resize(IMG_SIZE).convert("RGB")

    # Pixel in range [0, 255], coerente con il preprocessing usato in training
    img_array = np.array(img_resized).astype("float32")
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Predizione
    predictions = model.predict(img_array_expanded, verbose=0)[0]
    risultati_percentuali = {
        CLASS_NAMES[i]: float(predictions[i]) for i in range(NUM_CLASSES)
    }

    # Grad-CAM
    heatmap, _ = genera_gradcam(img_array_expanded, model)
    immagine_gradcam = sovrapponi_gradcam(img_resized, heatmap)

    return risultati_percentuali, immagine_gradcam

# ============================================================
# INTERFACCIA GRADIO
# ============================================================

with gr.Blocks(title="CDSS - Classificazione Endoscopica") as demo:

    gr.Markdown("# Sistema di Supporto alla Decisione Clinica (CDSS)")
    gr.Markdown(
        "Carica un'immagine endoscopica per ottenere una classificazione automatica "
        "tra 8 categorie patologiche, basata su EfficientNetB0 con fine-tuning. "
        "Il sistema mostra anche una mappa Grad-CAM che evidenzia le regioni "
        "dell'immagine piu rilevanti per la decisione del modello."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Carica immagine endoscopica",
                type="pil"
            )
            analyze_btn = gr.Button("Scansiona", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="Diagnosi dell'Intelligenza Artificiale",
                num_top_classes=8
            )

    gr.Markdown("### Mappa di Attivazione (Grad-CAM)")
    gr.Markdown(
        "Le aree evidenziate in rosso e giallo indicano le regioni dell'immagine "
        "su cui il modello ha basato maggiormente la propria decisione."
    )
    gradcam_output = gr.Image(label="Grad-CAM", type="pil")

    gr.Markdown(
        "**Nota:** questo strumento e un prototipo dimostrativo a scopo didattico "
        "e non sostituisce una diagnosi medica professionale. Il modello e stato "
        "addestrato esclusivamente sul Kvasir Dataset."
    )

    analyze_btn.click(
        fn=predict_e_gradcam,
        inputs=image_input,
        outputs=[output_label, gradcam_output]
    )

# ============================================================
# AVVIO APPLICAZIONE
# ============================================================

if __name__ == "__main__":
    demo.launch()
