import http
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os
from dataclasses import dataclass, field
from typing import Any, List, Dict
import io
import base64
from datetime import datetime

# Configuration 
LOGO_PATH = "logo.png"
USER_NAME = "Guemmi Hamza"
REPORT_TITLE = "Hands On 1"
Link_github="https://github.com/guemmihamza/Hands-On"

# Fonctions Utilitaires 
def encode_logo_to_base64(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, "rb") as f: return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

def pil_to_fpdf(pdf, pil_image, x, y, w):
    if pil_image.mode not in ['RGB', 'L']: pil_image = pil_image.convert('RGB')
    image_bytes = io.BytesIO(); pil_image.save(image_bytes, format='PNG')
    image_bytes.seek(0); pdf.image(image_bytes, x=x, y=y, w=w)

# Extraits de code pour l'annexe du PDF 
CODE_SNIPPETS = {
    'Analyse Statistique': """def get_image_statistics(gray_image):
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_image)
    mean_val = np.mean(gray_image)
    std_dev = np.std(gray_image)
    return { "Min": f"{min_val:.0f}", "Max": f"{max_val:.0f}", "Moyenne": f"{mean_val:.2f}", "Écart-type": f"{std_dev:.2f}" }""",
    'Génération d\'Histogramme': """def generate_histogram(gray_image):
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # ... (code pour dessiner l'histogramme)""",
    'Application de Filtres': """# Exemples de filtres OpenCV
Image.fromarray(cv2.Canny(gray_image, 100, 200))
Image.fromarray(cv2.GaussianBlur(gray_image, (5, 5), 0))""",
    'Séparation des Canaux': "r, g, b, _ = pil_image.convert('RGBA').split()"
}

# Structures de Données 
@dataclass
class SingleImageAnalysis:
    id: str; original_image: Any; basic_info: Dict = field(default_factory=dict); stats: Dict = field(default_factory=dict); hist_stats: Dict = field(default_factory=dict); visualizations: Dict[str, Any] = field(default_factory=dict); action_log: List[Dict] = field(default_factory=list)

@dataclass
class GlobalReportState:
    analyst_name: str = USER_NAME; analyses: List[SingleImageAnalysis] = field(default_factory=list)

#Classe PDF
class PDF(FPDF):
    def header(self):
        if os.path.exists(LOGO_PATH):
            logo_width, margin = 20, 10
            x_pos = self.w - logo_width - margin; y_pos = margin
            self.image(LOGO_PATH, x=x_pos, y=y_pos, w=logo_width)
            self.set_xy(x_pos, y_pos + logo_width)
            self.set_font('Helvetica', 'B', 10); self.cell(logo_width, 10, f"By {USER_NAME}", align='C')
        self.set_font('Helvetica', 'B', 14); self.set_xy(0, margin)
        self.cell(0, 10, REPORT_TITLE, align='C'); self.ln(20)
    def footer(self):
        self.set_y(-15); self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

# Logique de Génération PDF 
def generate_pdf(list_of_analyses: List[SingleImageAnalysis], analyst_name: str) -> str:
    if not list_of_analyses: raise gr.Error("Le rapport est vide ! Impossible de générer un PDF.")
    pdf = PDF('P', 'mm', 'A4')
    def write_section(title, data_dict):
        pdf.set_font('Helvetica', 'B', 12); pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Helvetica', '', 11)
        for key, value in data_dict.items(): pdf.multi_cell(0, 6, f"{key}: {str(value)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


        pdf.ln(4)
    for i, analysis in enumerate(list_of_analyses):
        pdf.add_page(); pdf.set_font('Helvetica', 'B', 16); pdf.cell(0, 10, f"Chapitre {i + 1}: Analyse de l'Image (ID: {analysis.id})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        write_section("Métadonnées", {"Link to Project":Link_github, "Analyste": analyst_name})
        write_section("Caractéristiques de l'Image", analysis.basic_info); write_section("Analyse Statistique", analysis.stats)
        pdf.add_page(); pdf.set_font('Helvetica', 'B', 16); pdf.cell(0, 10, f"Image (ID: {analysis.id}) - Originale", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pil_to_fpdf(pdf, analysis.original_image, x=15, y=pdf.get_y(), w=180)
        for title, img in analysis.visualizations.items():
            pdf.add_page(); pdf.set_font('Helvetica', 'B', 16); pdf.cell(0, 10, f'Image (ID: {analysis.id}) - Visualisation: {title}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if title == 'Histogramme': write_section("Données de l'Histogramme", analysis.hist_stats)
            pil_to_fpdf(pdf, img, x=15, y=pdf.get_y(), w=180)
        for action in analysis.action_log:
            pdf.add_page(); pdf.set_font('Helvetica', 'B', 16); pdf.cell(0, 10, f"Image (ID: {analysis.id}) - Action: {action['type']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if action['type'] == "Lecture de Pixel": write_section("Détails", action['data'])
            elif action['type'] == "Recadrage":
                write_section("Détails du Recadrage", action['data']); write_section("Analyse de la Zone", action['analysis'])
                pdf.add_page(); pdf.set_font('Helvetica', 'B', 16); pdf.cell(0, 10, 'Image Recadrée', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                PIXELS_TO_MM = 25.4 / 72; cropped_img = action['image']; width_mm = cropped_img.width * PIXELS_TO_MM
                x_pos = (210 - width_mm) / 2; pil_to_fpdf(pdf, cropped_img, x=x_pos, y=pdf.get_y(), w=width_mm)
    pdf.add_page(); pdf.set_font('Helvetica', 'B', 20); pdf.cell(0, 15, "Annexe: Code Source", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for title, code in CODE_SNIPPETS.items():
        pdf.set_font('Helvetica', 'B', 14); pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Courier', '', 10); pdf.set_fill_color(240, 240, 240); pdf.multi_cell(0, 5, code, border=1, fill=True); pdf.ln(5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name); return tmp_file.name

#Fonctions d'analyse 
def get_image_statistics(gray_image): return { "Min": f"{cv2.minMaxLoc(gray_image)[0]:.0f}", "Max": f"{cv2.minMaxLoc(gray_image)[1]:.0f}", "Moyenne": f"{np.mean(gray_image):.2f}", "Écart-type": f"{np.std(gray_image):.2f}"}
def get_basic_info(pil_image): return {"Dimensions": f"{pil_image.width}x{pil_image.height}px", "Mode": pil_image.mode}
def generate_histogram(gray_image): hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]); hist_img = np.zeros((300, 256, 3), dtype=np.uint8); cv2.normalize(hist, hist, 0, 299, cv2.NORM_MINMAX); [cv2.line(hist_img, (i, 299), (i, 299 - int(h[0])), (255,255,255)) for i,h in enumerate(hist)]; return Image.fromarray(hist_img), {"Pic": f"{np.argmax(hist)}"}
def apply_filters(gray_image): return {"Bords (Canny)": Image.fromarray(cv2.Canny(gray_image, 100, 200)), "Gaussien": Image.fromarray(cv2.GaussianBlur(gray_image, (5,5), 0)), "Médian": Image.fromarray(cv2.medianBlur(gray_image, 5)), "Égalisé": Image.fromarray(cv2.equalizeHist(gray_image))}
def get_color_channels(pil_image): r,g,b,_ = pil_image.convert('RGBA').split(); return r,g,b
def crop_image_from_coords(image, x, y, w, h): return image.crop((int(x), int(y), int(x+w), int(y+h)))
def read_pixel_value(image, x, y): return f"Valeur à ({int(x)},{int(y)}): {image.getpixel((int(x), int(y)))}"
def perform_full_analysis(image):
    analysis_id = datetime.now().strftime("%H%M%S-%f")[:-3]
    gray_pil = image.convert('L'); gray_cv = np.array(gray_pil); hist_img, hist_stats = generate_histogram(gray_cv); filtered_imgs = apply_filters(gray_cv); r_ch, g_ch, b_ch = get_color_channels(image)
    visualizations = {"Histogramme": hist_img, **filtered_imgs, "Canal Rouge": r_ch, "Canal Vert": g_ch, "Canal Bleu": b_ch}
    return SingleImageAnalysis(id=analysis_id, original_image=image, basic_info=get_basic_info(image), stats=get_image_statistics(gray_cv), hist_stats=hist_stats, visualizations=visualizations)

#Gestionnaires d'événements 
def handle_analyze_image(image):
    if image is None: raise gr.Error("Veuillez charger une image.")
    current_analysis = perform_full_analysis(image)
    viz = current_analysis.visualizations
    analysis_text = (f"**Caractéristiques**\n" + "\n".join([f"- {k}: {v}" for k,v in current_analysis.basic_info.items()]) + f"\n\n**Statistiques**\n" + "\n".join([f"- {k}: {v}" for k,v in current_analysis.stats.items()]))
    return [current_analysis, analysis_text, viz.get("Canal Rouge"), viz.get("Canal Vert"), viz.get("Canal Bleu"), viz.get("Histogramme"), viz.get("Bords (Canny)"), viz.get("Égalisé"), viz.get("Gaussien"), viz.get("Médian")]
def update_dropdown_choices(global_state): return gr.Dropdown(choices=[f"ID: {a.id}" for a in global_state.analyses], label="Analyses dans le rapport")
def handle_add_to_report(global_state, current_analysis):
    if current_analysis is None: raise gr.Error("Veuillez d'abord analyser une image.")
    if any(a.id == current_analysis.id for a in global_state.analyses): raise gr.Error(f"L'analyse {current_analysis.id} est déjà dans le rapport.")
    global_state.analyses.append(current_analysis)
    return global_state, update_dropdown_choices(global_state)
def handle_delete_analysis(global_state, analysis_id_to_delete):
    if not analysis_id_to_delete: raise gr.Error("Veuillez sélectionner une analyse à supprimer.")
    clean_id = analysis_id_to_delete.replace("ID: ", "")
    initial_len = len(global_state.analyses)
    global_state.analyses = [a for a in global_state.analyses if a.id != clean_id]
    if len(global_state.analyses) == initial_len: raise gr.Error(f"Analyse {clean_id} non trouvée.")
    return global_state, update_dropdown_choices(global_state), f"✅ Analyse {clean_id} supprimée."
def handle_generate_global_report(global_state): return generate_pdf(global_state.analyses, global_state.analyst_name)
def handle_generate_single_report(global_state, analysis_id):
    if not analysis_id: raise gr.Error("Veuillez sélectionner une analyse.")
    clean_id = analysis_id.replace("ID: ", "")
    selected_analysis = next((a for a in global_state.analyses if a.id == clean_id), None)
    if not selected_analysis: raise gr.Error(f"Analyse {clean_id} non trouvée.")
    return generate_pdf([selected_analysis], global_state.analyst_name)
def handle_read_pixel(current_analysis, image, x, y):
    if current_analysis is None: raise gr.Error("Veuillez d'abord analyser une image.")
    value = read_pixel_value(image, x, y); current_analysis.action_log.append({'type': 'Lecture de Pixel', 'data': {'Coordonnées': f"({int(x)}, {int(y)})", 'Valeur': value.split(': ')[1]}})
    return current_analysis, value
def handle_crop(current_analysis, x, y, w, h):
    if current_analysis is None: raise gr.Error("Veuillez d'abord analyser une image.")
    cropped_image = crop_image_from_coords(current_analysis.original_image, x, y, w, h); analysis_of_crop = get_image_statistics(np.array(cropped_image.convert('L')))
    current_analysis.action_log.append({'type': 'Recadrage', 'data': {'Coordonnées': f"X:{int(x)},Y:{int(y)},L:{int(w)},H:{int(h)}"}, 'analysis': analysis_of_crop, 'image': cropped_image})
    return current_analysis, cropped_image

# INTERFACE UTILISATEUR
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
).set(
    body_background_fill="#F3F4F6", body_background_fill_dark="#E5E7EB",
    body_text_color="#9CA3AF", body_text_color_dark="#F9FAFB",
    background_fill_primary="#1F2937", background_fill_primary_dark="#1F2937",
    background_fill_secondary="#374151", background_fill_secondary_dark="#374151",
    border_color_accent="#374151", border_color_accent_dark="#374151",
    border_color_primary="#374151", border_color_primary_dark="#374151",
    color_accent_soft="transparent", color_accent_soft_dark="transparent",
    link_text_color="#9CA3AF", link_text_color_dark="#9CA3AF",
    block_background_fill="#1F2937", block_background_fill_dark="#1F2937",
    block_border_width="1px",
    block_title_text_color="#9CA3AF", block_title_text_color_dark="#9CA3AF",
    block_label_text_color="#9CA3AF", block_label_text_color_dark="#9CA3AF",
    button_primary_background_fill="#2563EB", button_primary_background_fill_dark="#2563EB",
    button_primary_text_color="#9CA3AF", button_primary_text_color_dark="#9CA3AF",
    button_secondary_background_fill="#374151", button_secondary_background_fill_dark="#374151",
    button_secondary_text_color="#9CA3AF", button_secondary_text_color_dark="#9CA3AF",
)

with gr.Blocks(theme=theme, title=REPORT_TITLE) as app:
    logo_b64 = encode_logo_to_base64(LOGO_PATH)
    global_report_state = gr.State(value=GlobalReportState())
    current_single_analysis = gr.State()

    with gr.Row():
        if logo_b64:
            with gr.Column(scale=1, min_width=150):
                gr.HTML(f"<div style='display:flex;height:100%;align-items:center;justify-content:center;'><img src='{logo_b64}' style='max-height:80px;max-width:100%;border-radius:20%;'/></div>")
        with gr.Column(scale=5):
            gr.Markdown(f"# {REPORT_TITLE}\n*By {USER_NAME}*")
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 🖼️ 1. Entrée & Actions")
                input_image = gr.Image(type="pil", label="Chargez votre image ici")
                image_display = gr.Image(type="pil", label="Prévisualisation pour actions")
                input_image.upload(lambda img: img, inputs=input_image, outputs=image_display)
                analyze_btn = gr.Button("🔬 Analyser l'image", variant="primary")
                with gr.Accordion("🛠️ Actions interactives", open=False):
                    gr.Markdown("**Lecture de Pixel**")
                    with gr.Row():
                        pixel_x = gr.Number(label="X", value=20)
                        pixel_y = gr.Number(label="Y", value=20)
                    read_pixel_btn = gr.Button("📍 Lire Pixel")
                    pixel_value_output = gr.Textbox(label="Valeur", interactive=False)
                    gr.Markdown("**Recadrage**")
                    with gr.Row():
                        crop_x = gr.Number(label="X", value=0)
                        crop_y = gr.Number(label="Y", value=0)
                    with gr.Row():
                        crop_w = gr.Number(label="L", value=100)
                        crop_h = gr.Number(label="H", value=100)
                    crop_btn = gr.Button("✂️ Recadrer")
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📚 2. Gestion du Rapport")
                add_to_report_btn = gr.Button("➕ Ajouter au rapport", variant="secondary")
                analyses_dropdown = gr.Dropdown(label="Analyses dans le rapport", choices=[])
                delete_btn = gr.Button("🗑️ Supprimer la sélection")
                status_text = gr.Markdown()
            with gr.Group():
                gr.Markdown("### 📄 3. Génération PDF")
                download_single_btn = gr.Button("Télécharger le rapport sélectionné")
                download_global_btn = gr.Button("🌍 Générer le Rapport Global", variant="primary")
                download_report_file = gr.File(label="📥 Fichier PDF", interactive=False)

    with gr.Accordion("🔍 Voir les résultats de l'analyse", open=True):
        with gr.Tabs():
            with gr.TabItem("📊 Infos & Stats"):
                analysis_output_text = gr.Markdown()
            with gr.TabItem("🌈 Canaux de Couleur"):
                with gr.Row():
                    channel_r = gr.Image(label="Rouge")
                    channel_g = gr.Image(label="Vert")
                    channel_b = gr.Image(label="Bleu")
            with gr.TabItem("📉 Histogramme"):
                histogram_output = gr.Image()
            with gr.TabItem("✨ Filtres Appliqués"):
                with gr.Row():
                    edges_output = gr.Image(label="Bords (Canny)")
                    equalized_output = gr.Image(label="Histogramme Égalisé")
                with gr.Row():
                    gaussian_output = gr.Image(label="Filtre Gaussien")
                    median_output = gr.Image(label="Filtre Médian")

   
    analysis_outputs = [current_single_analysis, analysis_output_text, channel_r, channel_g, channel_b, histogram_output, edges_output, equalized_output, gaussian_output, median_output]
    analyze_btn.click(fn=handle_analyze_image, inputs=[input_image], outputs=analysis_outputs)
    add_to_report_btn.click(fn=handle_add_to_report, inputs=[global_report_state, current_single_analysis], outputs=[global_report_state, analyses_dropdown])
    delete_btn.click(fn=handle_delete_analysis, inputs=[global_report_state, analyses_dropdown], outputs=[global_report_state, analyses_dropdown, status_text])
    read_pixel_btn.click(fn=handle_read_pixel, inputs=[current_single_analysis, image_display, pixel_x, pixel_y], outputs=[current_single_analysis, pixel_value_output])
    crop_btn.click(fn=handle_crop, inputs=[current_single_analysis, crop_x, crop_y, crop_w, crop_h], outputs=[current_single_analysis, image_display])
    download_global_btn.click(fn=handle_generate_global_report, inputs=[global_report_state], outputs=[download_report_file])
    download_single_btn.click(fn=handle_generate_single_report, inputs=[global_report_state, analyses_dropdown], outputs=[download_report_file])

if __name__ == "__main__":
    app.launch(debug=True, share=False)
    
