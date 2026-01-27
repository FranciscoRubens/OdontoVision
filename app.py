import streamlit as st
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from rede2 import AttentionUNet
from transforms import get_inference_transform
import io
import base64
import time

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import os

# ==========================================
# 🦷 CONFIGURAÇÃO INICIAL
# ==========================================
st.set_page_config(
    page_title="Segmentação de Radiografias",
    page_icon="🦷",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_banner = os.path.join(BASE_DIR, "assets", "dente5.png")

# ==========================================
# ⚙️ CARREGAR MODELO
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionUNet().to(device)
model.load_state_dict(
    torch.load("AttUNet_final_trained.pt", map_location=device)
)
model.eval()


transform = get_inference_transform()

# ==========================================
# 📂 ESTADOS DE SESSÃO
# ==========================================
for key, default_value in {
    "radiografia": None,
    "mask": None,
    "mask_for_download": None,
    "processing": False,
    "show_results": False,
    "show_clear_success": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ==========================================
# 🧭 SIDEBAR
# ==========================================
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #CAE9F5;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.image(default_banner, width="stretch")
    st.divider()
    st.header("UPLOAD DA RADIOGRAFIA")

    uploaded_file = st.file_uploader(
        "SELECIONE UMA RADIOGRAFIA",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None and st.session_state.radiografia is not None:
        st.session_state.radiografia = None
        st.session_state.mask = None
        st.session_state.mask_for_download = None
        st.session_state.show_results = False

    if uploaded_file:
        try:
            st.session_state.radiografia = Image.open(uploaded_file).convert("L")
            st.session_state.show_results = False
            st.session_state.mask = None
            st.session_state.mask_for_download = None
        except UnidentifiedImageError:
            st.error("Arquivo inválido. Tente PNG ou JPG.")

    if st.session_state.radiografia:
        st.image(
            st.session_state.radiografia,
            caption="Pré-visualização",
            width="stretch"
        )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Segmentar", width="stretch"):
            if st.session_state.radiografia is not None:
                st.session_state.processing = True
                st.session_state.show_results = True
                st.session_state.mask = None
                st.session_state.mask_for_download = None

    with col2:
        if st.button("Limpar", width="stretch"):
            st.session_state.radiografia = None
            st.session_state.mask = None
            st.session_state.mask_for_download = None
            st.session_state.show_results = False
            st.session_state.show_clear_success = True

    st.divider()
    st.caption("Desenvolvido para fins acadêmicos — OdontoVision © 2025")

# ==========================================
# 🩻 TÍTULO PRINCIPAL
# ==========================================
st.markdown(
    """
    <h1 style="
        text-align: center;
        font-weight: 750;
        text-transform: uppercase;
        color: #000000;
    ">
        Segmentação de Radiografias Panorâmicas 🦷
    </h1>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 🚀 MENSAGEM DE SUCESSO
# ==========================================
if st.session_state.show_clear_success:
    st.success("Segmentação removida com sucesso ✅")
    st.session_state.show_clear_success = False

# ==========================================
# 🔍 PROCESSAMENTO + TEMPO COMPUTACIONAL
# ==========================================
if st.session_state.radiografia and st.session_state.show_results:

    img_np = np.array(st.session_state.radiografia)

    if st.session_state.mask is None:
        with st.spinner("Gerando segmentação..."):

            augmented = transform(image=img_np)
            input_tensor = augmented["image"].unsqueeze(0).to(device)

            # 🔹 Warm-up (não contabilizado)
            with torch.no_grad():
                _ = model(input_tensor)

            # 🔹 Medição do tempo computacional de inferência
            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                pred = model(input_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()

                tempo_computacional = end_time - start_time  # segundos

                print(
                    f"[INFO] Tempo computacional de inferência: "
                    f"{tempo_computacional * 1000:.2f} ms"
                )

                if isinstance(pred, list):
                    pred = pred[-1]

                mask = (
                    (pred.squeeze().cpu().numpy() > 0.5)
                    .astype(np.uint8) * 255
                )

            st.session_state.mask = mask
            st.session_state.mask_for_download = Image.fromarray(mask)

# ==========================================
# 🧾 FUNÇÃO PARA GERAR PDF
# ==========================================
def gerar_pdf(radiografia, mask, logo_path):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=landscape(A4))
    width, height = landscape(A4)

    try:
        logo = Image.open(logo_path)
        logo_io = io.BytesIO()
        logo.save(logo_io, format="PNG")
        logo_io.seek(0)
        c.drawImage(
            ImageReader(logo_io),
            (width - 130) / 2,
            height - 95,
            width=130,
            height=68,
            mask="auto"
        )
    except Exception:
        pass

    img_width, img_height = 380, 195
    spacing = 40

    radiografia = radiografia.resize((img_width, img_height))
    mask = mask.resize((img_width, img_height))

    rad_io, mask_io = io.BytesIO(), io.BytesIO()
    radiografia.save(rad_io, format="PNG")
    mask.save(mask_io, format="PNG")
    rad_io.seek(0)
    mask_io.seek(0)

    start_x = (width - (img_width * 2 + spacing)) / 2
    y_pos = (height - img_height) / 2 - 40

    c.drawImage(ImageReader(rad_io), start_x, y_pos, img_width, img_height)
    c.drawImage(
        ImageReader(mask_io),
        start_x + img_width + spacing,
        y_pos,
        img_width,
        img_height
    )

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 4, y_pos + img_height + 25, "Radiografia Original")
    c.drawCentredString(3 * width / 4, y_pos + img_height + 25, "Máscara Segmentada")

    c.setFont("Helvetica", 11)
    c.drawCentredString(
        width / 2,
        30,
        "Relatório gerado automaticamente — OdontoVision © 2025"
    )

    c.showPage()
    c.save()

    pdf_buffer.seek(0)
    return pdf_buffer

# ==========================================
# ✅ EXIBIÇÃO DO RESULTADO
# ==========================================
if st.session_state.mask is not None and st.session_state.radiografia is not None:
    st.subheader("Resultado da Segmentação")

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            st.session_state.radiografia,
            caption="Radiografia Original",
            width="stretch"
        )
    with col2:
        st.image(
            st.session_state.mask,
            caption="Máscara Segmentada",
            width="stretch"
        )

    pdf_buffer = gerar_pdf(
        st.session_state.radiografia,
        st.session_state.mask_for_download,
        default_banner
    )

    pdf_data = pdf_buffer.getvalue()
    b64_pdf = base64.b64encode(pdf_data).decode()

    st.markdown(
        f"""
        <a href="data:application/pdf;base64,{b64_pdf}"
           download="segmentacao_radiografia.pdf"
           style="
                display: inline-block;
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                padding: 10px 20px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
           ">
            📄 Baixar PDF
        </a>
        """,
        unsafe_allow_html=True
    )
