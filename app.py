import streamlit as st
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from rede2 import AttentionUNet
from transforms import get_inference_transform
import io
import base64
import time
import os

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


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

model_path = os.path.join(BASE_DIR, "AttUNet_final_trained.pt")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = get_inference_transform()


# ==========================================
# 📂 ESTADOS DE SESSÃO
# ==========================================
default_session = {
    "radiografia": None,
    "mask": None,
    "mask_for_download": None,
    "processing": False,
    "show_results": False,
    "show_clear_success": False
}

for k, v in default_session.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================
# 🧭 SIDEBAR
# ==========================================
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #CAE9F5;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image(default_banner, use_column_width=True)
    st.divider()
    st.header("UPLOAD DA RADIOGRAFIA")

    uploaded_file = st.file_uploader(
        "SELECIONE UMA RADIOGRAFIA",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None and st.session_state.radiografia is not None:
        st.session_state.update(default_session)

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
            use_column_width=True
        )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Segmentar", use_container_width=True):
            if st.session_state.radiografia is not None:
                st.session_state.processing = True
                st.session_state.show_results = True
                st.session_state.mask = None
                st.session_state.mask_for_download = None

    with col2:
        if st.button("Limpar", use_container_width=True):
            st.session_state.update(default_session)
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

            with torch.no_grad():
                _ = model(input_tensor)

            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                pred = model(input_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                tempo = (time.perf_counter() - start)

                print(f"[INFO] Inferência: {tempo*1000:.2f} ms")

                if isinstance(pred, list):
                    pred = pred[-1]

                mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

            st.session_state.mask = mask
            st.session_state.mask_for_download = Image.fromarray(mask)


# ==========================================
# 🧾 GERAR PDF
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
        c.drawImage(ImageReader(logo_io),
                    (width - 130) / 2,
                    height - 95,
                    width=130, height=68,
                    mask="auto")
    except:
        pass

    img_w, img_h = 380, 195
    spacing = 40

    radiografia = radiografia.resize((img_w, img_h))
    mask = mask.resize((img_w, img_h))

    rad_io, mask_io = io.BytesIO(), io.BytesIO()
    radiografia.save(rad_io, format="PNG")
    mask.save(mask_io, format="PNG")
    rad_io.seek(0)
    mask_io.seek(0)

    start_x = (width - (img_w * 2 + spacing)) / 2
    y_pos = (height - img_h) / 2 - 40

    c.drawImage(ImageReader(rad_io), start_x, y_pos, img_w, img_h)
    c.drawImage(ImageReader(mask_io), start_x + img_w + spacing, y_pos, img_w, img_h)

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 4, y_pos + img_h + 25, "Radiografia Original")
    c.drawCentredString(3 * width / 4, y_pos + img_h + 25, "Máscara Segmentada")

    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2, 30,
                        "Relatório gerado automaticamente — OdontoVision © 2025")

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
            use_column_width=True
        )

    with col2:
        st.image(
            st.session_state.mask,
            caption="Máscara Segmentada",
            use_column_width=True
        )

    pdf_buffer = gerar_pdf(
        st.session_state.radiografia,
        st.session_state.mask_for_download,
        default_banner
    )

    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()

    st.markdown(
        f"""
        <a href="data:application/pdf;base64,{b64}"
           download="segmentacao_radiografia.pdf"
           style="
                display:inline-block;
                background:white;
                color:black;
                border:1px solid #ccc;
                padding:10px 20px;
                border-radius:8px;
                text-decoration:none;
                font-weight:bold;
           ">
            📄 Baixar PDF
        </a>
        """,
        unsafe_allow_html=True
    )
