"""
Standalone Gradio app for the EdgeViT / TinyViT CIFAR-10 classifier.

Extracted from the notebook's Final Stage so the demo can run outside Colab.
Expects an ONNX model exported by the notebook (default: tinyvit_student.onnx
in the current directory, or set EDGEVIT_MODEL_PATH to point elsewhere).

Usage:
    pip install -r requirements.txt
    python app.py
"""
import os
import numpy as np
import onnxruntime as ort
import gradio as gr

MODEL_PATH = os.environ.get('EDGEVIT_MODEL_PATH', 'tinyvit_student.onnx')
ort_sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
inp_name = ort_sess.get_inputs()[0].name

CIFAR_MEAN_NP = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD_NP  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CLASSES_LIST  = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']
CLASS_EMOJI   = {
    'airplane':'✈️','automobile':'🚗','bird':'🐦','cat':'🐱','deer':'🦌',
    'dog':'🐶','frog':'🐸','horse':'🐴','ship':'🚢','truck':'🚛'
}

def predict(img):
    if img is None:
        return "", "", ""
    x = np.array(img.resize((32, 32))).astype(np.float32) / 255.0
    x = (x - CIFAR_MEAN_NP) / CIFAR_STD_NP
    x = x.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    logits = ort_sess.run(None, {inp_name: x})[0][0]
    probs  = np.exp(logits - logits.max())
    probs  = probs / probs.sum()
    top3   = np.argsort(probs)[::-1][:3]

    top_label = CLASSES_LIST[top3[0]]
    emoji     = CLASS_EMOJI[top_label]
    confidence = probs[top3[0]] * 100

    # Main prediction card
    pred_html = f"""
    <div style="
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        border: 1px solid #00d4ff33;
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 30px #00d4ff22, inset 0 0 30px #ffffff05;
    ">
        <div style="font-size: 56px; margin-bottom: 8px;">{emoji}</div>
        <div style="
            font-size: 32px;
            font-weight: 700;
            color: #00d4ff;
            letter-spacing: 3px;
            text-transform: uppercase;
            text-shadow: 0 0 20px #00d4ff88;
        ">{top_label}</div>
        <div style="
            font-size: 18px;
            color: #a0aec0;
            margin-top: 6px;
            letter-spacing: 1px;
        ">{confidence:.1f}% confidence</div>
        <div style="
            margin-top: 16px;
            height: 4px;
            background: #1a1a2e;
            border-radius: 4px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                width: {confidence:.1f}%;
                background: linear-gradient(90deg, #00d4ff, #7c3aed);
                border-radius: 4px;
                box-shadow: 0 0 10px #00d4ff66;
                transition: width 0.6s ease;
            "></div>
        </div>
    </div>
    """

    # Top-3 breakdown
    bars_html = '<div style="display: flex; flex-direction: column; gap: 10px; font-family: \'Courier New\', monospace;">'
    colors = ['#00d4ff', '#7c3aed', '#06b6d4']
    for rank, (i, color) in enumerate(zip(top3, colors)):
        name  = CLASSES_LIST[i]
        pct   = probs[i] * 100
        emj   = CLASS_EMOJI[name]
        bars_html += f"""
        <div style="
            background: #0f0f1a;
            border: 1px solid {color}33;
            border-radius: 10px;
            padding: 10px 14px;
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #e2e8f0; font-size: 14px; font-weight: 600;">
                    {emj} {name.upper()}
                </span>
                <span style="color: {color}; font-size: 14px; font-weight: 700;">{pct:.1f}%</span>
            </div>
            <div style="height: 6px; background: #1a1a2e; border-radius: 3px; overflow: hidden;">
                <div style="
                    height: 100%;
                    width: {pct:.1f}%;
                    background: {color};
                    border-radius: 3px;
                    box-shadow: 0 0 8px {color}88;
                "></div>
            </div>
        </div>
        """
    bars_html += '</div>'

    # Stats footer
    stats_html = f"""
    <div style="
        background: #0f0f1a;
        border: 1px solid #ffffff11;
        border-radius: 10px;
        padding: 12px 16px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #4a5568;
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
        text-align: center;
        margin-top: 4px;
    ">
        <div><div style="color:#00d4ff; font-size:13px; font-weight:700;">TinyViT</div><div>Architecture</div></div>
        <div><div style="color:#7c3aed; font-size:13px; font-weight:700;">ONNX</div><div>Runtime</div></div>
        <div><div style="color:#06b6d4; font-size:13px; font-weight:700;">0.24MB</div><div>Model size</div></div>
    </div>
    """

    return pred_html, bars_html, stats_html


css = """
* { box-sizing: border-box; }

body, .gradio-container {
    background: #080810 !important;
    font-family: 'Courier New', monospace !important;
}

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* Header */
#header {
    text-align: center;
    padding: 32px 20px 16px;
    background: linear-gradient(180deg, #0d0d1f 0%, #080810 100%);
    border-bottom: 1px solid #00d4ff22;
    margin-bottom: 8px;
}

/* Upload zone */
.upload-zone .wrap {
    background: #0f0f1a !important;
    border: 2px dashed #00d4ff44 !important;
    border-radius: 16px !important;
    transition: border-color 0.3s ease !important;
}
.upload-zone .wrap:hover {
    border-color: #00d4ffaa !important;
    box-shadow: 0 0 20px #00d4ff22 !important;
}

/* Button */
#predict-btn {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Courier New', monospace !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    height: 48px !important;
    box-shadow: 0 0 20px #00d4ff44 !important;
    transition: all 0.2s ease !important;
}
#predict-btn:hover {
    box-shadow: 0 0 30px #00d4ff88 !important;
    transform: translateY(-1px) !important;
}

/* Output panels */
.output-panel {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.output-panel > .wrap { border: none !important; background: transparent !important; }
.output-panel label { display: none !important; }

/* Remove default panel backgrounds */
.panel { background: transparent !important; border: none !important; }
.block { background: transparent !important; }
.gap { gap: 12px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0f0f1a; }
::-webkit-scrollbar-thumb { background: #00d4ff44; border-radius: 2px; }
"""

HEADER_HTML = """
<div id="header">
    <div style="
        font-size: 11px;
        letter-spacing: 4px;
        color: #00d4ff;
        text-transform: uppercase;
        margin-bottom: 10px;
        opacity: 0.8;
    ">24AI636 · Deep Learning Project</div>
    <div style="
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 2px;
        line-height: 1.2;
        font-family: 'Courier New', monospace;
    ">CIFAR<span style="color:#00d4ff">-10</span> CLASSIFIER</div>
    <div style="
        font-size: 12px;
        color: #4a5568;
        margin-top: 10px;
        letter-spacing: 1px;
    ">CNN → LSTM → AE/GAN → ViT → KD → <span style="color:#7c3aed">ONNX</span></div>
    <div style="
        display: inline-flex;
        gap: 8px;
        margin-top: 14px;
        flex-wrap: wrap;
        justify-content: center;
    ">
        """ + "".join([
            f'<span style="background:#0f0f1a; border:1px solid #00d4ff22; color:#718096; '
            f'font-size:10px; padding:3px 10px; border-radius:20px; letter-spacing:1px;">'
            f'{CLASS_EMOJI[c]} {c}</span>'
            for c in CLASSES_LIST
        ]) + """
    </div>
</div>
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type='pil',
                label='Upload Image',
                elem_classes=['upload-zone'],
                height=280
            )
            predict_btn = gr.Button(
                "⚡ CLASSIFY",
                elem_id="predict-btn",
                variant="primary"
            )

        with gr.Column(scale=1):
            pred_out  = gr.HTML(elem_classes=['output-panel'])
            bars_out  = gr.HTML(elem_classes=['output-panel'])
            stats_out = gr.HTML(elem_classes=['output-panel'])

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[pred_out, bars_out, stats_out]
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[pred_out, bars_out, stats_out]
    )

if __name__ == '__main__':
    demo.launch(share=False)
