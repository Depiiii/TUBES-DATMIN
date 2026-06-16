import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Deposito Nasabah",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    logreg    = joblib.load(os.path.join(BASE_DIR, "logreg_model.pkl"))
    nb        = joblib.load(os.path.join(BASE_DIR, "nb_model.pkl"))
    scaler    = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    kmeans    = joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))
    sc_km     = joblib.load(os.path.join(BASE_DIR, "scaler_kmeans.pkl"))
    with open(os.path.join(BASE_DIR, "feature_list.json")) as f:
        features = json.load(f)
    return logreg, nb, scaler, kmeans, sc_km, features

logreg, nb, scaler, kmeans, sc_km, FEATURES = load_models()

# ─── HELPER: build feature vector ────────────────────────────────────────────
def build_feature_vector(inp: dict) -> np.ndarray:
    """
    Turn raw form inputs into the exact dummy-encoded feature vector
    the models were trained on.
    """
    row = {f: 0 for f in FEATURES}

    # Numeric pass-throughs
    for col in ["duration", "pdays", "previous", "campaign"]:
        if col in row:
            row[col] = inp[col]

    # One-hot / binary dummies
    if inp["poutcome"] == "success"  and "poutcome_success" in row:
        row["poutcome_success"] = 1
    if inp["poutcome"] == "unknown"  and "poutcome_unknown"  in row:
        row["poutcome_unknown"]  = 1
    if inp["contact"]  == "unknown"  and "contact_unknown"   in row:
        row["contact_unknown"]   = 1
    if inp["housing"]  == "yes"      and "housing_yes"       in row:
        row["housing_yes"]       = 1
    if inp["loan"]     == "yes"      and "loan_yes"          in row:
        row["loan_yes"]          = 1
    if inp["job"]      == "retired"  and "job_retired"       in row:
        row["job_retired"]       = 1
    if inp["job"]      == "blue-collar" and "job_blue-collar" in row:
        row["job_blue-collar"]   = 1

    month_map = {
        "may": "month_may", "mar": "month_mar",
        "oct": "month_oct", "sep": "month_sep",
    }
    m_key = month_map.get(inp["month"])
    if m_key and m_key in row:
        row[m_key] = 1

    return np.array([row[f] for f in FEATURES]).reshape(1, -1)


def cluster_label(c: int) -> str:
    labels = {0: "Cluster 0 – Nasabah Netral",
              1: "Cluster 1 – Nasabah Resistif",
              2: "Cluster 2 – Nasabah Potensial Tinggi"}
    return labels.get(c, f"Cluster {c}")


def cluster_desc(c: int) -> str:
    descs = {
        0: ("⚠️ Kelompok terbesar (54%). Durasi panggilan cukup panjang. "
            "Perlu strategi lebih spesifik untuk meningkatkan konversi."),
        1: ("❌ Tingkat deposito terendah (24%). Saldo kecil, sering dihubungi "
            "namun konversi rendah. Kurangi alokasi sumber daya untuk kelompok ini."),
        2: ("✅ Tingkat deposito tertinggi (64,8%). Saldo lebih tinggi, riwayat "
            "kampanye sebelumnya sukses. **Prioritaskan kelompok ini!**"),
    }
    return descs.get(c, "")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/"
        "Python-logo-notext.svg/50px-Python-logo-notext.svg.png",
        width=40,
    )
    st.title("🏦 Bank Deposit Predictor")
    st.markdown("**Kelompok 5 – Penambangan Data**")
    st.markdown("Universitas Telkom · 2026")
    st.divider()
    st.info(
        "Isi form di sebelah kanan untuk memprediksi apakah seorang nasabah "
        "akan **berlangganan deposito berjangka** atau tidak."
    )
    st.divider()
    st.caption("Model: Logistic Regression (79.76%) & Naive Bayes (75.46%)")

# ─── MAIN ────────────────────────────────────────────────────────────────────
st.title("🏦 Dashboard Prediksi Respons Nasabah Deposito")
st.markdown(
    "Dataset: **Bank Marketing** · Portugal · 11.162 nasabah · "
    "Metode: K-Means Clustering, Logistic Regression, Naive Bayes"
)
st.divider()

tab_pred, tab_info = st.tabs(["🔮 Prediksi Nasabah", "ℹ️ Info Model"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION FORM
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("📋 Data Nasabah")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Pribadi**")
        age = st.number_input(
            "Usia (tahun)", min_value=18, max_value=95, value=35,
            help="Rentang usia nasabah: 18–95 tahun"
        )
        job = st.selectbox(
            "Pekerjaan",
            ["admin.", "blue-collar", "entrepreneur", "housemaid",
             "management", "retired", "self-employed", "services",
             "student", "technician", "unemployed", "unknown"],
            index=4,
            help="Jenis pekerjaan nasabah"
        )
        marital = st.selectbox(
            "Status Pernikahan",
            ["divorced", "married", "single"],
            index=1,
        )
        education = st.selectbox(
            "Tingkat Pendidikan",
            ["primary", "secondary", "tertiary", "unknown"],
            index=1,
        )

    with col2:
        st.markdown("**Kondisi Keuangan**")
        balance = st.number_input(
            "Saldo Rata-rata Tahunan (€)",
            min_value=-10000, max_value=100000, value=1500, step=100,
            help="Saldo rekening rata-rata per tahun"
        )
        default = st.selectbox(
            "Kredit Macet?",
            ["no", "yes"],
            help="Apakah nasabah memiliki kredit macet?"
        )
        housing = st.selectbox(
            "Pinjaman Rumah (KPR)?",
            ["no", "yes"],
            index=1,
            help="Apakah nasabah memiliki pinjaman rumah?"
        )
        loan = st.selectbox(
            "Pinjaman Pribadi?",
            ["no", "yes"],
            help="Apakah nasabah memiliki pinjaman pribadi?"
        )

    with col3:
        st.markdown("**Data Kampanye**")
        contact = st.selectbox(
            "Jenis Kontak",
            ["cellular", "telephone", "unknown"],
            help="Metode kontak yang digunakan"
        )
        month = st.selectbox(
            "Bulan Kontak Terakhir",
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"],
            index=4,
        )
        day = st.slider("Hari Kontak Terakhir", 1, 31, 15)
        duration = st.number_input(
            "Durasi Panggilan Terakhir (detik)",
            min_value=0, max_value=4000, value=300, step=10,
            help="Durasi panggilan terakhir dalam detik. Rata-rata: 372 detik."
        )

    st.divider()

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**Riwayat Kampanye Sebelumnya**")
        campaign = st.number_input(
            "Jumlah Kontak Kampanye Ini",
            min_value=1, max_value=63, value=2,
            help="Jumlah kali nasabah dihubungi selama kampanye ini"
        )
        pdays = st.number_input(
            "Hari Sejak Kontak Terakhir (pdays)",
            min_value=-1, max_value=854, value=-1,
            help="-1 artinya nasabah belum pernah dihubungi sebelumnya"
        )
        previous = st.number_input(
            "Jumlah Kontak Sebelum Kampanye",
            min_value=0, max_value=58, value=0,
            help="Jumlah kontak sebelum kampanye ini dilakukan"
        )
        poutcome = st.selectbox(
            "Hasil Kampanye Sebelumnya",
            ["failure", "other", "success", "unknown"],
            index=3,
            help="Hasil kampanye pemasaran sebelumnya"
        )

    with col5:
        st.markdown("**Pilihan Model**")
        model_choice = st.radio(
            "Gunakan model:",
            ["Logistic Regression (Akurasi: 79.76%)",
             "Naive Bayes (Akurasi: 75.46%)",
             "Keduanya (Bandingkan)"],
            index=2,
        )

    st.divider()

    # ── PREDICT BUTTON ───────────────────────────────────────────────────────
    predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_btn:
        inp = {
            "age": age, "job": job, "marital": marital,
            "education": education, "balance": balance,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day": day,
            "duration": duration, "campaign": campaign,
            "pdays": pdays, "previous": previous, "poutcome": poutcome,
        }

        fv = build_feature_vector(inp)
        fv_scaled = scaler.transform(fv)
        fv_km_scaled = sc_km.transform(fv)

        # Predictions
        pred_lr  = logreg.predict(fv_scaled)[0]
        prob_lr  = logreg.predict_proba(fv_scaled)[0]
        pred_nb  = nb.predict(fv_scaled)[0]
        prob_nb  = nb.predict_proba(fv_scaled)[0]
        cluster  = kmeans.predict(fv_km_scaled)[0]

        st.divider()
        st.subheader("📊 Hasil Prediksi")

        # ── Cluster result
        st.markdown("#### 🔵 Segmentasi Nasabah (K-Means Clustering)")
        c_col1, c_col2 = st.columns([1, 2])
        with c_col1:
            cluster_color = ["#FFA500", "#FF4444", "#00CC66"][int(cluster)]
            st.markdown(
                f"<div style='background:{cluster_color};padding:16px;"
                f"border-radius:10px;text-align:center;color:white;"
                f"font-weight:bold;font-size:18px'>"
                f"{cluster_label(int(cluster))}</div>",
                unsafe_allow_html=True,
            )
        with c_col2:
            st.info(cluster_desc(int(cluster)))

        st.divider()

        # ── Classification results
        st.markdown("#### 🤖 Prediksi Klasifikasi")

        if "Keduanya" in model_choice:
            r1, r2 = st.columns(2)
            models_to_show = [
                ("Logistic Regression", pred_lr, prob_lr),
                ("Naive Bayes", pred_nb, prob_nb),
            ]
        elif "Logistic" in model_choice:
            r1, r2 = st.columns([1, 1])
            models_to_show = [("Logistic Regression", pred_lr, prob_lr)]
        else:
            r1, r2 = st.columns([1, 1])
            models_to_show = [("Naive Bayes", pred_nb, prob_nb)]

        cols_result = st.columns(len(models_to_show))
        for i, (mname, pred, prob) in enumerate(models_to_show):
            with cols_result[i]:
                label = "✅ YA – Berlangganan Deposito" if pred == 1 else "❌ TIDAK – Tidak Berlangganan"
                color = "#00CC66" if pred == 1 else "#FF4444"
                prob_yes = prob[1] * 100
                prob_no  = prob[0] * 100

                st.markdown(
                    f"<div style='border:2px solid {color};border-radius:12px;"
                    f"padding:20px;text-align:center'>"
                    f"<h4 style='color:#555'>{mname}</h4>"
                    f"<div style='background:{color};color:white;padding:12px;"
                    f"border-radius:8px;font-size:16px;font-weight:bold'>"
                    f"{label}</div>"
                    f"<br>"
                    f"<b>Probabilitas Deposito (YA):</b> {prob_yes:.1f}%<br>"
                    f"<b>Probabilitas Tidak (TIDAK):</b> {prob_no:.1f}%"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.progress(int(prob_yes), text=f"Peluang YA: {prob_yes:.1f}%")

        # ── Recommendation
        st.divider()
        st.markdown("#### 💡 Rekomendasi Bisnis")
        final_pred = pred_lr if "Logistic" in model_choice or "Keduanya" in model_choice else pred_nb
        final_prob = prob_lr[1] if "Logistic" in model_choice or "Keduanya" in model_choice else prob_nb[1]

        if final_pred == 1 and final_prob >= 0.7:
            st.success(
                "🎯 **Nasabah ini sangat potensial!** Prioritaskan dalam kampanye telemarketing. "
                "Hubungi segera dengan penawaran deposito yang relevan."
            )
        elif final_pred == 1 and final_prob >= 0.5:
            st.warning(
                "📞 **Nasabah ini cukup berpotensi.** Pertimbangkan untuk menghubungi "
                "dengan pendekatan personal yang menarik."
            )
        else:
            st.error(
                "⏭️ **Nasabah ini kemungkinan tidak tertarik.** Alokasikan sumber daya "
                "ke nasabah lain yang lebih potensial untuk efisiensi biaya kampanye."
            )

        # ── Summary table
        st.divider()
        st.markdown("#### 📝 Ringkasan Input")
        summary = {
            "Usia": age, "Pekerjaan": job, "Status Nikah": marital,
            "Pendidikan": education, "Saldo (€)": balance,
            "KPR": housing, "Pinjaman": loan, "Kontak": contact,
            "Bulan": month, "Durasi (detik)": duration,
            "Jumlah Kontak": campaign, "Pdays": pdays,
            "Kontak Sebelumnya": previous, "Hasil Kampanye Lalu": poutcome,
        }
        st.dataframe(
            pd.DataFrame(summary.items(), columns=["Variabel", "Nilai"]),
            use_container_width=True, hide_index=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("ℹ️ Informasi Model & Dataset")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📊 Performa Model")
        perf_df = pd.DataFrame({
            "Model":     ["Logistic Regression", "Naive Bayes"],
            "Akurasi":   ["79.76%", "75.46%"],
            "Precision": ["79.59%", "71.50%"],
            "Recall":    ["77.03%", "80.15%"],
            "F1-Score":  ["78.29%", "75.58%"],
            "AUC-ROC":   ["0.87",   "0.81"],
        })
        st.dataframe(perf_df, hide_index=True, use_container_width=True)

        st.markdown("### 🔵 Hasil K-Means Clustering (k=3)")
        cluster_df = pd.DataFrame({
            "Cluster": ["Cluster 0 – Netral", "Cluster 1 – Resistif", "Cluster 2 – Potensial Tinggi"],
            "Jumlah":  ["6.045 (54.2%)", "2.558 (22.9%)", "2.559 (22.9%)"],
            "Rate Deposito": ["~50%", "~24%", "~64.8%"],
            "Karakteristik": [
                "Durasi panggilan tertinggi (386 det.)",
                "Saldo kecil, paling sering dihubungi",
                "Saldo tinggi, riwayat kampanye sukses",
            ]
        })
        st.dataframe(cluster_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("### 📁 Dataset")
        st.markdown("""
| Properti          | Nilai                        |
|-------------------|------------------------------|
| **Sumber**        | Kaggle – Bank Marketing      |
| **Jumlah Baris**  | 11.162 nasabah               |
| **Jumlah Kolom**  | 17 atribut                   |
| **Target**        | deposit (yes / no)           |
| **Distribusi**    | 47.4% Yes · 52.6% No         |
| **Periode Data**  | 2008–2010 (Portugal)         |
        """)

        st.markdown("### 🔑 Fitur yang Digunakan Model")
        feat_df = pd.DataFrame({
            "Fitur": FEATURES,
            "Keterangan": [
                "Durasi panggilan (detik)",
                "Hasil kampanye sebelumnya = sukses",
                "Jenis kontak = tidak diketahui",
                "Hasil kampanye sebelumnya = tidak diketahui",
                "Punya KPR = ya",
                "Bulan kontak = Mei",
                "Hari sejak kontak terakhir",
                "Jumlah kontak sebelum kampanye",
                "Bulan kontak = Maret",
                "Bulan kontak = Oktober",
                "Jumlah kontak kampanye ini",
                "Bulan kontak = September",
                "Punya pinjaman pribadi = ya",
                "Pekerjaan = pensiun",
                "Pekerjaan = buruh/blue-collar",
            ]
        })
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### 🎓 Tim Peneliti – Kelompok 5")
    team_df = pd.DataFrame({
        "Nama": [
            "Putri Tania Adelfianti",
            "Adelia Fitri Pramushinta",
            "Devi Fitria Rahmawati",
            "Lintang Metyaputri",
        ],
        "NIM": ["102022400021", "102022400358", "102022400171", "102022430067"],
    })
    st.dataframe(team_df, hide_index=True, use_container_width=True)
    st.caption("Program Studi S1 Sistem Informasi · Fakultas Rekayasa Industri · Universitas Telkom · 2026")
