"""
SPROUT AI — AI-Powered Plant Care Assistant
Built for College Design Thinking and Methodology (DTM) Project.
Features genuine local PyTorch inference, webcam input, botanical recommendations,
evaluation suite, and editable 5-stage Design Thinking documentation.
"""

import os
import io
import time
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Custom project modules
from model_utils import CLASSES, CLASS_LABELS, get_device
from predict import SproutPredictor, assess_image_quality
from recommendations import get_recommendation, BOTANICAL_DISCLAIMER
from train_model import (
    train_sprout_model,
    check_dataset_status,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODELS_DIR,
    MODEL_FILENAME,
    METADATA_FILENAME
)
from evaluation import evaluate_test_dataset, generate_confusion_matrix_plot
from design_thinking_data import (
    DEFAULT_PROBLEM_STATEMENT,
    DEFAULT_DTM_STAGES,
    generate_markdown_report
)

# ---------------------------------------------------------
# Page Configuration & Botanical Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="SPROUT AI — Plant Care Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Botanical Color Palette */
    :root {
        --sprout-forest: #1A3E26;
        --sprout-leaf: #2E7D32;
        --sprout-accent: #4CAF50;
        --sprout-sage: #E8F5E9;
        --sprout-card-bg: #FFFFFF;
        --sprout-border: #E0E7E1;
        --sprout-text-dark: #1E293B;
        --sprout-text-muted: #64748B;
    }

    /* App Header Card */
    .sprout-header {
        background: linear-gradient(135deg, #1A3E26 0%, #2E5A36 60%, #3B7A46 100%);
        color: #FFFFFF;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(26, 62, 38, 0.2);
    }
    .sprout-header h1 {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .sprout-header p {
        font-size: 1.05rem;
        margin-top: 0.5rem;
        color: #D1E7D5;
        font-weight: 400;
    }
    .sprout-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
        backdrop-filter: blur(4px);
    }

    /* Clean Elevated Cards */
    .sprout-card {
        background: #FFFFFF;
        border: 1px solid var(--sprout-border);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }

    /* Result Badges */
    .badge-healthy {
        background: #E8F5E9;
        color: #1B5E20;
        border: 1px solid #A5D6A7;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .badge-stressed {
        background: #FFEBEE;
        color: #B71C1C;
        border: 1px solid #EF9A9A;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .badge-uncertain {
        background: #FFF3E0;
        color: #E65100;
        border: 1px solid #FFCC80;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }

    /* Warning Callout */
    .sprout-alert {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #92400E;
    }

    .sprout-disclaimer {
        background: #F1F5F9;
        border-left: 4px solid #64748B;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "dtm_stages" not in st.session_state:
    st.session_state.dtm_stages = {
        k: v["content"] for k, v in DEFAULT_DTM_STAGES.items()
    }

if "predictor" not in st.session_state:
    st.session_state.predictor = SproutPredictor()

predictor = st.session_state.predictor

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="sprout-header">
    <span class="sprout-badge">🌱 College Design Thinking & Methodology Project</span>
    <h1>SPROUT AI</h1>
    <p>AI-Powered Plant Care Assistant — Zero-Budget, Camera-Based Vision & Intelligent Botanical Care</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ System Settings")
    conf_threshold = st.slider(
        "Confidence Reliability Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Predictions below this confidence value are flagged as 'Uncertain' to prevent false diagnoses."
    )

    st.markdown("---")
    st.subheader("🤖 Model Status")
    is_ready = predictor.is_model_ready()
    if is_ready:
        st.success("✅ Model Loaded & Active")
        meta = predictor.metadata or {}
        st.caption(f"**Architecture:** {meta.get('architecture', 'PyTorch CNN')}")
        st.caption(f"**Trained On:** {meta.get('training_samples', 'N/A')} samples")
        st.caption(f"**Val Accuracy:** {meta.get('final_val_accuracy', 0)*100:.1f}%")
        st.caption(f"**Hardware Device:** {predictor.device.type.upper()}")
    else:
        st.warning("⚠️ No Trained Model Loaded")
        st.caption(predictor.load_error or "Train a model in the 'Model Training' tab.")

    if st.button("🔄 Reload Model Weights", use_container_width=True):
        predictor.reload_model()
        st.rerun()

    st.markdown("---")
    st.caption("SPROUT AI • Zero-Budget Human-Centered Engineering")

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_analysis, tab_history, tab_train, tab_eval, tab_dtm, tab_about = st.tabs([
    "🌱 Plant Analysis",
    "📜 Analysis History",
    "🏋️ Model Training",
    "🧪 Model Evaluation",
    "💡 Design Thinking Report",
    "ℹ️ About & Demo Guide"
])


# =========================================================
# TAB 1: PLANT ANALYSIS (HOME)
# =========================================================
with tab_analysis:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Plant Image Input")
        input_mode = st.radio(
            "Select Image Source:",
            ["📁 Upload Image File", "📷 Live Laptop Webcam"],
            horizontal=True
        )

        image_data = None
        source_name = "Upload"

        if input_mode == "📁 Upload Image File":
            uploaded_file = st.file_uploader(
                "Upload a leaf or plant photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="Accepts high-resolution JPG, PNG, or WEBP photos of leaves."
            )
            if uploaded_file is not None:
                try:
                    image_data = Image.open(uploaded_file)
                    source_name = f"File: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Failed to open image file: {str(e)}")

        else:
            st.caption("Allow camera permissions when prompted by your browser.")
            webcam_image = st.camera_input("Capture live leaf photograph")
            if webcam_image is not None:
                try:
                    image_data = Image.open(webcam_image)
                    source_name = "Live Webcam"
                except Exception as e:
                    st.error(f"Failed to process webcam feed: {str(e)}")

        if image_data is not None:
            st.image(image_data, caption=f"Selected Input ({source_name})", use_container_width=True)
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                run_analysis = st.button("🔍 Analyse Plant", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("🗑️ Reset", use_container_width=True):
                    st.rerun()
        else:
            st.info("👆 Please upload a plant image or take a photo with your webcam above to begin analysis.")
            run_analysis = False

    with col_result:
        st.subheader("2. Diagnostic Results & Care Plan")

        if run_analysis and image_data is not None:
            with st.spinner("Processing image and executing neural inference..."):
                time.sleep(0.15)  # smooth visual transition
                res = predictor.predict(image_data, confidence_threshold=conf_threshold)

            if not res.get("success", False) and res.get("model_missing", False):
                st.error("🚨 Trained Model Weights Missing")
                st.write(
                    "SPROUT AI requires a trained model to make genuine predictions. "
                    "A starter dataset has been created for you! Please switch to the **'Model Training'** tab "
                    "and click **'Start Training Model'** (takes ~15 seconds on standard CPU)."
                )
            elif not res.get("success", False):
                st.error(f"Prediction Error: {res.get('error', 'Unknown failure')}")
            else:
                pred_class = res["predicted_class"]
                conf = res["confidence"]
                probs = res.get("probabilities", {})
                rec = res["recommendation"]
                quality = res.get("quality_assessment", {})

                # Session history logging
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": source_name,
                    "condition": rec["status_title"],
                    "confidence": f"{conf*100:.1f}%",
                    "p_healthy": f"{probs.get('healthy', 0)*100:.1f}%",
                    "p_stressed": f"{probs.get('stressed', 0)*100:.1f}%",
                    "image": image_data.copy()
                })

                # Result Header Card
                badge_class = f"badge-{pred_class}"
                st.markdown(f"""
                <div class="sprout-card" style="border-left: 6px solid {rec['badge_color']};">
                    <span class="{badge_class}">{rec['status_title']}</span>
                    <h3 style="margin-top: 0.8rem; margin-bottom: 0.2rem; color: #1E293B;">Confidence: {conf*100:.1f}%</h3>
                    <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 0.8rem;">{res['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence Gauge
                st.write("**Model Probability Distribution:**")
                c1, c2 = st.columns(2)
                c1.metric("Healthy Probability", f"{probs.get('healthy', 0)*100:.1f}%")
                c2.metric("Stressed Probability", f"{probs.get('stressed', 0)*100:.1f}%")
                st.progress(float(probs.get('stressed', 0)), text="Stress Probability Scale")

                # Quality / Blur Alerts
                if quality.get("warnings"):
                    for w in quality["warnings"]:
                        st.markdown(f"<div class='sprout-alert'>⚠️ <b>Image Quality Alert:</b> {w}</div>", unsafe_allow_html=True)

                # Tailored Recommendations Section
                st.markdown("#### 📋 Recommended Care Actions")
                st.markdown(f"**Diagnostic Summary:** {rec['summary']}")

                st.write("**Action Checklist:**")
                for item in rec.get("primary_actions", []):
                    st.markdown(f"- ✅ {item}")

                if rec.get("what_not_to_do"):
                    st.markdown("#### 🛑 Critical 'What NOT To Do'")
                    for no_item in rec["what_not_to_do"]:
                        st.markdown(f"- ❌ **{no_item}**")

                # Botanical Integrity Disclaimer
                st.markdown(f"""
                <div class="sprout-disclaimer">
                    ℹ️ <b>Botanical Integrity Notice:</b> {rec['disclaimer']}
                </div>
                """, unsafe_allow_html=True)

        elif image_data is None:
            st.markdown("""
            <div class="sprout-card" style="text-align: center; padding: 3rem 2rem; color: #64748B;">
                <span style="font-size: 3rem;">🌿</span>
                <h3 style="margin-top: 1rem; color: #1E293B;">Awaiting Plant Image</h3>
                <p>Upload a photograph or capture a frame from your webcam on the left to receive a real-time health classification and tailored botanical care steps.</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# TAB 2: ANALYSIS HISTORY
# =========================================================
with tab_history:
    st.subheader("📜 Session Analysis History")
    st.write("Review all plant diagnoses performed during this active session.")

    if not st.session_state.history:
        st.info("No analyses performed in this session yet. Analyse a plant in Tab 1 to build your history log.")
    else:
        # Action Bar
        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            # Create CSV Export
            history_export_data = [
                {
                    "Timestamp": h["timestamp"],
                    "Source": h["source"],
                    "Predicted Condition": h["condition"],
                    "Confidence": h["confidence"],
                    "Healthy Prob": h["p_healthy"],
                    "Stressed Prob": h["p_stressed"]
                }
                for h in st.session_state.history
            ]
            df_export = pd.DataFrame(history_export_data)
            csv_bytes = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export History to CSV",
                data=csv_bytes,
                file_name=f"sprout_ai_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_h2:
            if st.button("🗑️ Clear History Log", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.markdown("---")

        # Table & Card display
        for idx, entry in enumerate(reversed(st.session_state.history)):
            with st.container():
                c_img, c_detail = st.columns([1, 3])
                with c_img:
                    st.image(entry["image"], width=160)
                with c_detail:
                    st.markdown(f"**Analysis #{len(st.session_state.history) - idx}** — `{entry['timestamp']}`")
                    st.markdown(f"**Predicted Condition:** `{entry['condition']}` (Confidence: **{entry['confidence']}**)")
                    st.caption(f"Source: {entry['source']} | Healthy: {entry['p_healthy']} | Stressed: {entry['p_stressed']}")
                st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)


# =========================================================
# TAB 3: MODEL TRAINING
# =========================================================
with tab_train:
    st.subheader("🏋️ Local Model Training Pipeline")
    st.write(
        "SPROUT AI implements a genuine machine learning training workflow using PyTorch. "
        "Images from `dataset/healthy/` and `dataset/stressed/` are split into training and validation sets, "
        "augmented, and trained locally on your laptop CPU."
    )

    dataset_info = check_dataset_status(DEFAULT_DATASET_DIR)

    col_ds1, col_ds2, col_ds3 = st.columns(3)
    col_ds1.metric("Healthy Images", dataset_info["healthy_count"])
    col_ds2.metric("Stressed Images", dataset_info["stressed_count"])
    col_ds3.metric("Total Dataset Size", dataset_info["total_count"])

    if not dataset_info["is_ready"]:
        st.warning(
            f"⚠️ Insufficient images to train. Found {dataset_info['healthy_count']} healthy and "
            f"{dataset_info['stressed_count']} stressed images. Add at least 2 images to both "
            f"`{DEFAULT_DATASET_DIR}/healthy` and `{DEFAULT_DATASET_DIR}/stressed`."
        )

    st.markdown("---")
    st.markdown("#### Configure Hyperparameters")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        arch_choice = st.selectbox(
            "Model Architecture",
            ["MobileNetV2 (Transfer Learning)", "SproutCustomCNN (From Scratch)"],
            help="MobileNetV2 uses lightweight pretrained feature layers. SproutCustomCNN trains completely from scratch offline."
        )
        arch_key = "mobilenet_v2" if "MobileNetV2" in arch_choice else "custom_cnn"

    with col_p2:
        epochs_input = st.slider("Training Epochs", min_value=3, max_value=25, value=8, step=1)

    with col_p3:
        batch_size_input = st.selectbox("Batch Size", [4, 8, 16], index=1)

    start_training = st.button("🚀 Start Training Model", type="primary", disabled=not dataset_info["is_ready"], use_container_width=True)

    if start_training:
        progress_bar = st.progress(0, text="Initializing training pipeline...")
        status_text = st.empty()
        epoch_log_container = st.empty()

        logs = []

        def training_callback(summary: dict):
            ep = summary["epoch"]
            tot = summary["total_epochs"]
            progress_bar.progress(ep / tot, text=f"Epoch {ep}/{tot}: Loss={summary['train_loss']:.4f}, Val Acc={summary['val_acc']*100:.1f}%")
            logs.append(summary)
            # Display real-time table of recent epochs
            df_live = pd.DataFrame(logs)[["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]]
            df_live.columns = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
            epoch_log_container.dataframe(df_live, use_container_width=True)

        try:
            with st.spinner("Training neural network on CPU..."):
                train_res = train_sprout_model(
                    dataset_dir=DEFAULT_DATASET_DIR,
                    output_dir=DEFAULT_MODELS_DIR,
                    arch=arch_key,
                    pretrained=(arch_key == "mobilenet_v2"),
                    epochs=epochs_input,
                    batch_size=batch_size_input,
                    learning_rate=0.001,
                    progress_callback=training_callback
                )

            st.success(f"🎉 Model training successfully completed! Saved to `{train_res['model_path']}`.")
            predictor.reload_model()

            # Plot Loss and Accuracy curves
            meta = train_res["metadata"]
            hist = meta.get("history", {})
            if hist:
                df_hist = pd.DataFrame(hist)
                df_hist.index = range(1, len(df_hist) + 1)
                st.write("**Training & Validation Accuracy Progression:**")
                st.line_chart(df_hist[["train_acc", "val_acc"]])

        except Exception as e:
            st.error(f"Training failed: {str(e)}")


# =========================================================
# TAB 4: MODEL EVALUATION
# =========================================================
with tab_eval:
    st.subheader("🧪 Model Evaluation & Testing Suite")
    st.write(
        "Evaluate the trained model on unseen test images not used during training. "
        "Calculates genuine scikit-learn metrics: Accuracy, Precision, Recall, F1-Score, "
        "and visualizes the Confusion Matrix."
    )

    if not predictor.is_model_ready():
        st.warning("⚠️ Please train or load a model first before running evaluation.")
    else:
        eval_mode = st.radio(
            "Evaluation Data Source:",
            ["📂 Use Pre-packaged Unseen Test Set (`sample_test/`)", "📤 Upload Custom Test Images"],
            horizontal=True
        )

        test_records = []

        if eval_mode.startswith("📂"):
            # Load from sample_test folder
            h_dir = os.path.join("sample_test", "healthy")
            s_dir = os.path.join("sample_test", "stressed")

            for fn in os.listdir(h_dir) if os.path.exists(h_dir) else []:
                if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                    test_records.append({
                        "image": os.path.join(h_dir, fn),
                        "true_label": "healthy",
                        "filename": fn
                    })

            for fn in os.listdir(s_dir) if os.path.exists(s_dir) else []:
                if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                    test_records.append({
                        "image": os.path.join(s_dir, fn),
                        "true_label": "stressed",
                        "filename": fn
                    })

            st.info(f"Loaded **{len(test_records)}** unseen test samples (Healthy: {len([r for r in test_records if r['true_label']=='healthy'])}, Stressed: {len([r for r in test_records if r['true_label']=='stressed'])}).")

        else:
            col_up1, col_up2 = st.columns(2)
            with col_up1:
                healthy_files = st.file_uploader("Upload Unseen HEALTHY Test Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
                if healthy_files:
                    for f in healthy_files:
                        test_records.append({"image": Image.open(f), "true_label": "healthy", "filename": f.name})
            with col_up2:
                stressed_files = st.file_uploader("Upload Unseen STRESSED Test Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
                if stressed_files:
                    for f in stressed_files:
                        test_records.append({"image": Image.open(f), "true_label": "stressed", "filename": f.name})

            st.info(f"Total uploaded test samples: **{len(test_records)}**.")

        if st.button("📊 Run Model Evaluation Benchmark", type="primary", disabled=len(test_records) == 0):
            with st.spinner("Evaluating unseen test set against trained model..."):
                eval_res = evaluate_test_dataset(test_records, predictor, confidence_threshold=conf_threshold)

            if eval_res.get("success"):
                metrics = eval_res["metrics"]
                df_res = eval_res["results_table"]

                # Metrics Overview Cards
                st.markdown("### 📈 Evaluation Summary on Unseen Test Data")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Overall Accuracy", f"{metrics['accuracy']*100:.1f}%")
                m2.metric("Weighted Precision", f"{metrics['precision']*100:.1f}%")
                m3.metric("Weighted Recall", f"{metrics['recall']*100:.1f}%")
                m4.metric("Weighted F1-Score", f"{metrics['f1_score']*100:.1f}%")
                m5.metric("Coverage Rate", f"{metrics['coverage_rate']*100:.1f}%")

                st.caption(f"*Evaluated on {metrics['total_test_samples']} unseen samples. {metrics['uncertain_predictions']} samples flagged as Uncertain (<{conf_threshold*100:.0f}% threshold).*")

                st.markdown("---")
                col_cm, col_tab = st.columns([1, 1], gap="large")

                with col_cm:
                    st.write("**Confusion Matrix (Confident Predictions):**")
                    cm_fig = generate_confusion_matrix_plot(metrics["confusion_matrix"], labels=["Healthy", "Stressed"])
                    st.pyplot(cm_fig)

                with col_tab:
                    st.write("**Evaluation Breakdown:**")
                    st.dataframe(
                        df_res[["filename", "true_label", "predicted_class", "confidence", "is_correct"]],
                        use_container_width=True
                    )

                    # Export Evaluation to CSV
                    csv_eval = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Export Evaluation Report to CSV",
                        data=csv_eval,
                        file_name=f"sprout_ai_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error(eval_res.get("error", "Evaluation failed."))


# =========================================================
# TAB 5: DESIGN THINKING REPORT
# =========================================================
with tab_dtm:
    st.subheader("💡 Design Thinking and Methodology (DTM) Documentation")
    st.write(
        "Academic documentation structured across the 5 core stages of Design Thinking. "
        "Review or customize each stage directly, then export the finalized report for college submission."
    )

    # Core Problem Statement Box
    st.markdown(f"""
    <div class="sprout-card" style="border-left: 6px solid #2E7D32;">
        <h4 style="margin: 0; color: #1E293B;">🎯 Central Design Thinking Problem Statement</h4>
        <p style="font-size: 1.1rem; color: #15803D; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0;">
            "{DEFAULT_PROBLEM_STATEMENT}"
        </p>
    </div>
    """, unsafe_allow_html=True)

    dtm_tabs = st.tabs([
        "1. Empathise",
        "2. Define",
        "3. Ideate",
        "4. Prototype",
        "5. Test"
    ])

    stage_keys = ["empathise", "define", "ideate", "prototype", "test"]

    for idx, key in enumerate(stage_keys):
        with dtm_tabs[idx]:
            info = DEFAULT_DTM_STAGES[key]
            st.markdown(f"### {info['title']}")
            st.caption(info["subtitle"])

            edited_text = st.text_area(
                f"Edit notes for {info['title']}:",
                value=st.session_state.dtm_stages[key],
                height=260,
                key=f"dtm_edit_{key}"
            )
            st.session_state.dtm_stages[key] = edited_text

    st.markdown("---")
    # Export Report Button
    full_md_report = generate_markdown_report(st.session_state.dtm_stages)
    col_rep1, col_rep2 = st.columns([1, 1])
    with col_rep1:
        st.download_button(
            "📥 Export Full Design Thinking Report (Markdown)",
            data=full_md_report.encode("utf-8"),
            file_name=f"SPROUT_AI_Design_Thinking_Report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_rep2:
        st.caption("This export includes problem statements, user personas, weighted concept matrices, system architecture, and test cases.")


# =========================================================
# TAB 6: ABOUT & DEMO GUIDE
# =========================================================
with tab_about:
    st.subheader("ℹ️ About SPROUT AI & College Evaluator Demo Guide")

    col_a1, col_a2 = st.columns([1, 1], gap="large")

    with col_a1:
        st.markdown("### 🎓 Presentation & Viva Checklist")
        st.markdown("""
        Follow this 4-step sequence when demonstrating SPROUT AI to your college professor or evaluator:

        1. **State the Design Thinking Challenge:**
           - Read the central problem statement: *"How might we help students, beginners, and plant owners recognise visible signs of plant stress..."*
           - Emphasize the **Zero-Budget** criterion: runs on standard student laptops with zero external hardware or paid API keys.

        2. **Demonstrate Genuine AI Inference:**
           - Go to **Tab 1 (Plant Analysis)**.
           - Upload a healthy leaf photo → Point out the **Confidence Gauge** and routine care steps.
           - Switch to the laptop **Webcam** → Hold up a leaf (or photo) and click *Capture & Analyse*.
           - Demonstrate **Uncertainty Rejection**: Show how setting a high threshold (e.g. 0.90) turns an ambiguous image into *"Uncertain / Insufficient Confidence"* instead of hallucinating.

        3. **Highlight Botanical Integrity:**
           - Explain why SPROUT AI instructs users to perform the **2-inch soil knuckle test** rather than falsely claiming computer vision can measure root moisture.

        4. **Showcase Model Evaluation:**
           - Switch to **Tab 4 (Model Evaluation)**.
           - Run the test benchmark → Show the **Confusion Matrix** and explain Accuracy vs Recall.
        """)

    with col_a2:
        st.markdown("### ❓ Likely Evaluator Questions & Model Answers")
        with st.expander("Q: Why didn't you use an Arduino or soil moisture sensor?"):
            st.write(
                "**Answer:** In the Ideation stage (Concept Evaluation Matrix), physical IoT sensors failed our "
                "accessibility and zero-budget criteria. Probes corrode, cost money, and require hardware maintenance. "
                "SPROUT AI leverages the student's existing webcam and guides them through physical validation steps."
            )

        with st.expander("Q: Can computer vision tell if a plant is overwatered or underwatered?"):
            st.write(
                "**Answer:** No, and claiming it can would be scientifically dishonest. Both overwatering (root asphyxiation) "
                "and underwatering cause leaf wilting and chlorosis. SPROUT AI detects visible stress optically, then instructs "
                "the user to perform physical checks (knuckle test, drainage hole check)."
            )

        with st.expander("Q: Why did you choose MobileNetV2?"):
            st.write(
                "**Answer:** MobileNetV2 uses depthwise separable convolutions, which reduces model parameters by over 85% "
                "compared to traditional CNNs (like VGG16), allowing sub-100ms inference on standard laptop CPUs without a GPU."
            )

        with st.expander("Q: What prevents the AI from making wild guesses on bad photos?"):
            st.write(
                "**Answer:** We implemented a calibrated confidence rejection threshold (default 70%) and OpenCV "
                "Laplacian variance blur detection. If the model is not confident, it outputs 'Uncertain' rather than guessing."
            )

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748B; font-size: 0.9rem;">
        SPROUT AI — AI-Powered Plant Care Assistant • College Design Thinking and Methodology (DTM) Submission
    </div>
    """, unsafe_allow_html=True)
