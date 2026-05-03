import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import datetime

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Anomaly Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #0a0e17; color: #c8d6e5; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * { color: #8892a4 !important; }

div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    padding: 16px;
}
div[data-testid="metric-container"] label {
    color: #58a6ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}
div[data-testid="metric-container"] div {
    color: #f0f6fc !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #f0f6fc !important;
}
h1 { font-size: 1.6rem !important; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #58a6ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.finding-card {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #c8d6e5;
}
.finding-card.red { border-left-color: #ff6b6b; }
.finding-card.green { border-left-color: #3fb950; }
.finding-card.amber { border-left-color: #d29922; }

hr { border-color: #1e2d3d !important; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY TEMPLATE ───────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono", color="#8892a4", size=11),
    xaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d"),
    yaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d"),
    legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#1e2d3d", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(start_date, end_date):
    df = yf.download("AAPL", start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename_axis("Date").reset_index()
    df.columns.name = None

    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna()

    # Technical indicators
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
    df["Middle_Band"] = df["Close"].rolling(20).mean()
    df["Upper_Band"] = df["Middle_Band"] + 2 * df["Close"].rolling(20).std()
    df["Lower_Band"] = df["Middle_Band"] - 2 * df["Close"].rolling(20).std()

    for col in ["SMA_20","SMA_50","EMA_20","Middle_Band","Upper_Band","Lower_Band"]:
        df[col] = df[col].ffill()
    df[["Daily_Return","Volatility_20"]] = df[["Daily_Return","Volatility_20"]].bfill()
    df = df.dropna()

    # Anomaly detection
    df["Z_Close"] = zscore(df["Close"])
    df["Z_Volume"] = zscore(df["Volume"])
    df["Anomaly_Z"] = ((df["Z_Close"].abs() > 3) | (df["Z_Volume"].abs() > 3)).astype(int)

    feats = ["Close","Volume","SMA_20","EMA_20","Volatility_20"]
    iso = IsolationForest(contamination=0.02, random_state=42)
    df["Anomaly_IF"] = (iso.fit_predict(df[feats]) == -1).astype(int)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
    df["Anomaly_LOF"] = (lof.fit_predict(df[feats]) == -1).astype(int)

    df["Anomaly"] = df[["Anomaly_Z","Anomaly_IF","Anomaly_LOF"]].max(axis=1)

    # Sentiment proxy
    vol_mean = df["Volume"].mean()
    vol_std = df["Volume"].std()

    def derive_sentiment(row):
        ret = row["Daily_Return"]
        vz = (row["Volume"] - vol_mean) / vol_std
        if ret < -0.02 and vz > 1:
            return -0.6
        elif ret < -0.01:
            return -0.3
        elif ret > 0.02 and vz > 1:
            return 0.6
        elif ret > 0.01:
            return 0.3
        else:
            return 0.05

    df["Sentiment"] = df.apply(derive_sentiment, axis=1)
    return df

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 AAPL Anomaly Intelligence")
    st.markdown("---")

    st.markdown('<div class="section-label">Date Range</div>', unsafe_allow_html=True)
    start_date = st.date_input("Start", value=datetime.date(2020, 1, 1))
    end_date = st.date_input("End", value=datetime.date.today())

    st.markdown("---")
    st.markdown('<div class="section-label">Anomaly Methods</div>', unsafe_allow_html=True)
    show_zscore = st.checkbox("Z-Score", value=True)
    show_if = st.checkbox("Isolation Forest", value=True)
    show_lof = st.checkbox("LOF", value=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.selectbox("", [
        "📊 Overview",
        "🔍 Anomaly Detection",
        "🧭 PCA / t-SNE",
        "🔗 Association Rules",
        "🤖 Classification",
        "💬 Sentiment Analysis"
    ])

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading AAPL data..."):
    df = load_data(str(start_date), str(end_date))

anomaly_cols = []
if show_zscore:
    anomaly_cols.append("Anomaly_Z")
if show_if:
    anomaly_cols.append("Anomaly_IF")
if show_lof:
    anomaly_cols.append("Anomaly_LOF")
if anomaly_cols:
    df["Anomaly_Combined"] = df[anomaly_cols].max(axis=1)
else:
    df["Anomaly_Combined"] = 0

anomalies = df[df["Anomaly_Combined"] == 1]

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# AAPL Stock Anomaly Intelligence")
    st.markdown(f"**{start_date} → {end_date}** &nbsp;|&nbsp; Apple Inc. (NASDAQ: AAPL)")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trading Days", f"{len(df):,}")
    c2.metric("Anomalies Detected", f"{df['Anomaly_Combined'].sum()}")
    c3.metric("Anomaly Rate", f"{df['Anomaly_Combined'].mean()*100:.1f}%")
    c4.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
    c5.metric("Max Drawdown Day", f"{df['Daily_Return'].min()*100:.1f}%")

    st.markdown("---")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
        name="Close", line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"],
        name="SMA 20", line=dict(color="#d29922", width=1, dash="dot"), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"],
        name="SMA 50", line=dict(color="#8b949e", width=1, dash="dot"), opacity=0.7), row=1, col=1)

    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(x=anomalies["Date"], y=anomalies["Close"],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff6b6b", size=8, symbol="x", line=dict(width=2))), row=1, col=1)

    bar_colors = ["#3fb950" if r >= 0 else "#ff6b6b" for r in df["Daily_Return"]]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"],
        name="Volume", marker_color=bar_colors, opacity=0.6), row=2, col=1)

    fig.update_layout(**PLOT_THEME, height=500,
        title=dict(text="AAPL Price & Volume — Anomalies Highlighted",
                   font=dict(size=13, color="#f0f6fc")))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="finding-card red">🔴 <b>{df["Anomaly_Combined"].sum()} anomaly days</b> detected ({df["Anomaly_Combined"].mean()*100:.1f}% rate)</div>', unsafe_allow_html=True)
        worst_date = df.loc[df["Daily_Return"].idxmin(), "Date"].strftime("%b %d, %Y")
        st.markdown(f'<div class="finding-card amber">📉 Worst single day: <b>{df["Daily_Return"].min()*100:.1f}%</b> on {worst_date}</div>', unsafe_allow_html=True)
    with col2:
        best_date = df.loc[df["Daily_Return"].idxmax(), "Date"].strftime("%b %d, %Y")
        st.markdown(f'<div class="finding-card green">📈 Best single day: <b>+{df["Daily_Return"].max()*100:.1f}%</b> on {best_date}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="finding-card">🔵 Avg daily volatility: <b>{df["Volatility_20"].mean()*100:.2f}%</b></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.markdown("# Anomaly Detection Methods")
    st.markdown("Comparing Z-Score, Isolation Forest, and LOF across the same dataset.")
    st.markdown("---")

    method_map = {
        "Z-Score": ("Anomaly_Z", "#ff6b6b"),
        "Isolation Forest": ("Anomaly_IF", "#f78166"),
        "LOF": ("Anomaly_LOF", "#ffa657"),
    }

    tabs = st.tabs(["Z-Score", "Isolation Forest", "LOF", "Method Comparison"])

    for i, (method_name, (col, color)) in enumerate(method_map.items()):
        with tabs[i]:
            method_anomalies = df[df[col] == 1]
            st.metric(f"{method_name} Anomalies", len(method_anomalies))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                name="Close Price", line=dict(color="#58a6ff", width=1.5)))
            fig.add_trace(go.Scatter(x=method_anomalies["Date"], y=method_anomalies["Close"],
                mode="markers", name="Anomaly",
                marker=dict(color=color, size=9, symbol="x", line=dict(width=2))))
            fig.update_layout(**PLOT_THEME, height=400,
                title=dict(text=f"{method_name} — Detected Anomalies",
                           font=dict(color="#f0f6fc", size=13)))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        counts = {
            "Z-Score": int(df["Anomaly_Z"].sum()),
            "Isolation Forest": int(df["Anomaly_IF"].sum()),
            "LOF": int(df["Anomaly_LOF"].sum()),
            "Combined": int(df["Anomaly_Combined"].sum())
        }
        fig = go.Figure(go.Bar(
            x=list(counts.keys()), y=list(counts.values()),
            marker_color=["#ff6b6b","#f78166","#ffa657","#58a6ff"],
            text=list(counts.values()), textposition="outside",
            marker_line_color="#0a0e17", marker_line_width=1
        ))
        fig.update_layout(**PLOT_THEME, height=380,
            title=dict(text="Anomalies Detected per Method",
                       font=dict(color="#f0f6fc", size=13)),
            yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Method Agreement")
        overlap = pd.DataFrame({
            "Z-Score ∩ IF": [int((df["Anomaly_Z"] & df["Anomaly_IF"]).sum())],
            "Z-Score ∩ LOF": [int((df["Anomaly_Z"] & df["Anomaly_LOF"]).sum())],
            "IF ∩ LOF": [int((df["Anomaly_IF"] & df["Anomaly_LOF"]).sum())],
            "All Three": [int((df["Anomaly_Z"] & df["Anomaly_IF"] & df["Anomaly_LOF"]).sum())],
        })
        st.dataframe(overlap, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: PCA / t-SNE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧭 PCA / t-SNE":
    st.markdown("# Dimensionality Reduction")
    st.markdown("Projecting 9 stock features into 2D space to visualise normal vs anomalous days.")
    st.markdown("---")

    pca_features = ["Close","Volume","SMA_20","SMA_50","EMA_20",
                    "Volatility_20","Upper_Band","Lower_Band","Daily_Return"]
    df_pca = df[pca_features + ["Anomaly_Combined"]].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca[pca_features])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_pca["PC1"] = pca_result[:, 0]
    df_pca["PC2"] = pca_result[:, 1]
    df_pca["Label"] = df_pca["Anomaly_Combined"].map({0: "Normal", 1: "Anomaly"})

    col1, col2, col3 = st.columns(3)
    col1.metric("PC1 Variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
    col2.metric("PC2 Variance", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
    col3.metric("Total Explained", f"{sum(pca.explained_variance_ratio_)*100:.1f}%")

    tab1, tab2, tab3 = st.tabs(["PCA Scatter", "t-SNE Scatter", "Feature Loadings"])

    with tab1:
        fig = px.scatter(df_pca, x="PC1", y="PC2", color="Label",
            color_discrete_map={"Normal": "#58a6ff", "Anomaly": "#ff6b6b"},
            opacity=0.7, symbol="Label",
            symbol_map={"Normal": "circle", "Anomaly": "x"})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**PLOT_THEME, height=460,
            title=dict(text="PCA — Normal vs Anomalous Trading Days",
                       font=dict(color="#f0f6fc", size=13)))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        with st.spinner("Running t-SNE (may take ~30s)..."):
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            tsne_result = tsne.fit_transform(X_scaled)
            df_pca["tSNE1"] = tsne_result[:, 0]
            df_pca["tSNE2"] = tsne_result[:, 1]

        fig = px.scatter(df_pca, x="tSNE1", y="tSNE2", color="Label",
            color_discrete_map={"Normal": "#58a6ff", "Anomaly": "#ff6b6b"},
            opacity=0.7, symbol="Label",
            symbol_map={"Normal": "circle", "Anomaly": "x"})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**PLOT_THEME, height=460,
            title=dict(text="t-SNE — Non-linear Dimensionality Reduction",
                       font=dict(color="#f0f6fc", size=13)))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        loadings = pd.DataFrame(
            pca.components_.T, index=pca_features, columns=["PC1", "PC2"]
        )
        sorted_loadings = loadings["PC1"].abs().sort_values()
        fig = go.Figure(go.Bar(
            x=sorted_loadings.values, y=sorted_loadings.index,
            orientation="h", marker_color="#58a6ff",
            marker_line_color="#0a0e17", marker_line_width=1
        ))
        fig.update_layout(**PLOT_THEME, height=380,
            title=dict(text="Feature Loadings — PC1 Contributions",
                       font=dict(color="#f0f6fc", size=13)),
            xaxis_title="Absolute Loading Value")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    st.markdown("# Association Rule Mining")
    st.markdown("Discovering which market conditions co-occur on anomalous trading days.")
    st.markdown("---")

    min_support = st.slider("Minimum Support", 0.01, 0.2, 0.02, 0.005)
    min_confidence = st.slider("Minimum Confidence", 0.3, 1.0, 0.5, 0.05)

    with st.spinner("Mining association rules..."):
        df_rules = df.copy()
        df_rules["Price_State"] = pd.cut(df["Daily_Return"],
            bins=[-999, -0.02, 0.02, 999],
            labels=["Price_Drop", "Price_Stable", "Price_Spike"])
        df_rules["Volume_State"] = pd.cut(df["Volume"],
            bins=[0, df["Volume"].quantile(0.33), df["Volume"].quantile(0.66), 999999999999],
            labels=["Low_Volume", "Normal_Volume", "High_Volume"])
        df_rules["Volatility_State"] = pd.cut(df["Volatility_20"],
            bins=[0, df["Volatility_20"].quantile(0.33), df["Volatility_20"].quantile(0.66), 999],
            labels=["Low_Volatility", "Normal_Volatility", "High_Volatility"])
        df_rules["Band_State"] = pd.cut(df["Close"],
            bins=[-999, df["Lower_Band"].mean(), df["Upper_Band"].mean(), 999999],
            labels=["Below_Band", "Within_Band", "Above_Band"])
        df_rules["Anomaly_State"] = df["Anomaly_Combined"].map({0: "Normal_Day", 1: "Anomaly_Day"})

        df_rules = df_rules[["Price_State","Volume_State","Volatility_State",
                              "Band_State","Anomaly_State"]].dropna()
        transactions = df_rules.astype(str).values.tolist()
        te = TransactionEncoder()
        df_encoded = pd.DataFrame(
            te.fit(transactions).transform(transactions), columns=te.columns_
        )

        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
        rules = association_rules(frequent_itemsets, metric="confidence",
                                  min_threshold=min_confidence,
                                  num_itemsets=len(frequent_itemsets))
        rules = rules.sort_values("lift", ascending=False)

    st.metric("Rules Generated", len(rules))

    tab1, tab2 = st.tabs(["Support vs Confidence", "Anomaly Rules"])

    with tab1:
        rules_plot = rules.copy()
        rules_plot["antecedents_str"] = rules_plot["antecedents"].astype(str)
        rules_plot["consequents_str"] = rules_plot["consequents"].astype(str)
        fig = px.scatter(rules_plot, x="support", y="confidence", color="lift",
            color_continuous_scale="RdYlGn", opacity=0.75,
            hover_data={"antecedents_str": True, "consequents_str": True,
                        "support": True, "confidence": True, "lift": True})
        fig.update_layout(**PLOT_THEME, height=420,
            title=dict(text="All Rules — Support vs Confidence (color = Lift)",
                       font=dict(color="#f0f6fc", size=13)))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        anomaly_rules_df = rules[
            rules["antecedents"].astype(str).str.contains("Anomaly_Day") |
            rules["consequents"].astype(str).str.contains("Anomaly_Day")
        ].copy()

        if len(anomaly_rules_df) > 0:
            st.markdown(f"**{len(anomaly_rules_df)} rules involving Anomaly_Day**")
            top = anomaly_rules_df.head(8)
            labels = [str(list(x)) for x in top["antecedents"]]
            fig = go.Figure(go.Bar(
                x=top["confidence"].values, y=labels, orientation="h",
                marker_color="#ff6b6b", opacity=0.85,
                marker_line_color="#0a0e17", marker_line_width=1,
                text=[f"{v:.2f}" for v in top["confidence"].values],
                textposition="outside"
            ))
            fig.update_layout(**PLOT_THEME, height=400,
                title=dict(text="Top Rules Involving Anomaly Days — Confidence",
                           font=dict(color="#f0f6fc", size=13)),
                xaxis=dict(range=[0, 1.15], title="Confidence"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomaly rules found. Try lowering the minimum support threshold.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Classification":
    st.markdown("# Classification")
    st.markdown("Training Decision Tree and SVM classifiers on anomaly pseudo-labels.")
    st.markdown("---")

    feature_cols = ["Close","Volume","SMA_20","SMA_50","EMA_20",
                    "Volatility_20","Upper_Band","Lower_Band","Daily_Return"]
    df_model = df[feature_cols + ["Anomaly_Combined"]].dropna()
    X = StandardScaler().fit_transform(df_model[feature_cols])
    y = df_model["Anomaly_Combined"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    with st.spinner("Training classifiers..."):
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)

        svm = SVC(kernel="rbf", C=1.0, random_state=42)
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DT Accuracy", f"{accuracy_score(y_test, dt_preds)*100:.1f}%")
    c2.metric("DT F1 Score", f"{f1_score(y_test, dt_preds, zero_division=0):.3f}")
    c3.metric("SVM Accuracy", f"{accuracy_score(y_test, svm_preds)*100:.1f}%")
    c4.metric("SVM F1 Score", f"{f1_score(y_test, svm_preds, zero_division=0):.3f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Confusion Matrices", "Feature Importance"])

    with tab1:
        results = pd.DataFrame({
            "Model": ["Decision Tree", "SVM"],
            "Accuracy": [accuracy_score(y_test, dt_preds), accuracy_score(y_test, svm_preds)],
            "F1 Score": [f1_score(y_test, dt_preds, zero_division=0),
                         f1_score(y_test, svm_preds, zero_division=0)]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy", x=results["Model"], y=results["Accuracy"],
            marker_color="#58a6ff", marker_line_color="#0a0e17", marker_line_width=1))
        fig.add_trace(go.Bar(name="F1 Score", x=results["Model"], y=results["F1 Score"],
            marker_color="#3fb950", marker_line_color="#0a0e17", marker_line_width=1))
        fig.update_layout(**PLOT_THEME, barmode="group", height=380,
            title=dict(text="Classifier Performance Comparison",
                       font=dict(color="#f0f6fc", size=13)),
            yaxis=dict(range=[0, 1.15]))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        for preds, name, color_scale, col in [
            (dt_preds, "Decision Tree", "Blues", c1),
            (svm_preds, "SVM", "Greens", c2)
        ]:
            cm = confusion_matrix(y_test, preds)
            fig = px.imshow(cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["Normal", "Anomaly"], y=["Normal", "Anomaly"],
                color_continuous_scale=color_scale)
            fig.update_layout(**PLOT_THEME, height=320,
                title=dict(text=f"{name} Confusion Matrix",
                           font=dict(color="#f0f6fc", size=12)))
            col.plotly_chart(fig, use_container_width=True)

    with tab3:
        importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values()
        fig = go.Figure(go.Bar(
            x=importances.values, y=importances.index, orientation="h",
            marker_color="#58a6ff", marker_line_color="#0a0e17", marker_line_width=1
        ))
        fig.update_layout(**PLOT_THEME, height=380,
            title=dict(text="Feature Importance — Decision Tree",
                       font=dict(color="#f0f6fc", size=13)),
            xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💬 Sentiment Analysis":
    st.markdown("# Sentiment Analysis")
    st.markdown("Price-action derived sentiment proxy correlated with anomaly dates.")
    st.markdown("---")

    pos = int((df["Sentiment"] > 0.05).sum())
    neu = int(((df["Sentiment"] >= -0.05) & (df["Sentiment"] <= 0.05)).sum())
    neg = int((df["Sentiment"] < -0.05).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive Days", pos)
    c2.metric("Neutral Days", neu)
    c3.metric("Negative Days", neg)
    st.markdown("---")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
        name="Close", line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(x=anomalies["Date"], y=anomalies["Close"],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff6b6b", size=8, symbol="x", line=dict(width=2))), row=1, col=1)

    sent_colors = ["#3fb950" if s > 0.05 else "#ff6b6b" if s < -0.05 else "#8b949e"
                   for s in df["Sentiment"]]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Sentiment"],
        name="Sentiment", marker_color=sent_colors, opacity=0.8), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", opacity=0.5, row=2, col=1)

    fig.update_layout(**PLOT_THEME, height=520,
        title=dict(text="Price + Sentiment Timeline",
                   font=dict(color="#f0f6fc", size=13)))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    normal_sent = df[df["Anomaly_Combined"] == 0]["Sentiment"]
    anomaly_sent = df[df["Anomaly_Combined"] == 1]["Sentiment"]

    fig = go.Figure()
    fig.add_trace(go.Box(y=normal_sent.values, name="Normal Days",
        marker_color="#58a6ff", line_color="#58a6ff",
        fillcolor="rgba(88,166,255,0.15)"))
    fig.add_trace(go.Box(y=anomaly_sent.values, name="Anomaly Days",
        marker_color="#ff6b6b", line_color="#ff6b6b",
        fillcolor="rgba(255,107,107,0.15)"))
    fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", opacity=0.5)
    fig.update_layout(**PLOT_THEME, height=380,
        title=dict(text="Sentiment Distribution: Normal vs Anomaly Days",
                   font=dict(color="#f0f6fc", size=13)),
        yaxis_title="Sentiment Score")
    st.plotly_chart(fig, use_container_width=True)

    corr = df[["Sentiment","Anomaly_Combined"]].corr().loc["Sentiment","Anomaly_Combined"]
    st.markdown(f'<div class="finding-card">📊 Correlation between sentiment and anomaly: <b>{corr:.4f}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="finding-card red">📉 Mean sentiment on anomaly days: <b>{anomaly_sent.mean():.4f}</b> vs normal days: <b>{normal_sent.mean():.4f}</b></div>', unsafe_allow_html=True)
