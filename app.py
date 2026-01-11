import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Final Analysis Dashboard", layout="wide")

# ================= GLOBAL CSS =================
st.markdown("""
<style>
* { font-size: 18px !important; }
h1 { font-size: 36px !important; }
h2 { font-size: 30px !important; }
h3 { font-size: 24px !important; }
h4 { font-size: 20px !important; font-weight:700; text-align:center; }
figcaption { text-align:center; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("Computer Vision Analysis Dashboard – 22MIA1161")
st.caption("Shape, Contour, Object and Feature Analysis")
st.divider()

# ================= SIDEBAR =================
st.sidebar.header("Select Image")

input_mode = st.sidebar.radio(
    "Choose input source",
    ["Use Sample Image","Upload Image"]
)

sample_images = {
    "Sample 1": "assets/sample1.png",
    "Sample 2": "assets/sample2.png",
    "Sample 3": "assets/sample3.png"
}

uploaded_file = None
img = None

if input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)

else:
    choice = st.sidebar.selectbox("Select sample image", list(sample_images.keys()))
    image = Image.open(sample_images[choice]).convert("RGB")
    img = np.array(image)

st.sidebar.header("Controls")
min_area = st.sidebar.slider("Minimum Contour Area", 50, 3000, 150)
canny_low = st.sidebar.slider("Canny Threshold (Low)", 50, 200, 100)
canny_high = st.sidebar.slider("Canny Threshold (High)", 100, 300, 200)

# ================= SHAPE FUNCTION =================
def detect_shape(cnt, approx):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    v = len(approx)
    if peri == 0:
        return "Unknown"
    circ = (4 * np.pi * area) / (peri * peri)
    if v == 3: return "Triangle"
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "Square" if 0.9 <= w / h <= 1.1 else "Rectangle"
    if v == 5: return "Pentagon"
    if v == 6: return "Hexagon"
    if circ > 0.65: return "Circle"
    return "Irregular"

# ================= MAIN =================
if img is not None:

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ================= UPLOADED IMAGE =================
    st.header("Uploaded Image")
    st.columns([1,3,1])[1].image(img, width=850)

    # ================= SHAPE DETECTION =================
    st.header("Detect Geometric Shapes")

    thresh = cv2.adaptiveThreshold(
        gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    records = []

    for c in contours:
        if cv2.contourArea(c) > min_area:
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.04*peri,True)
            shape = detect_shape(c,approx)
            area = cv2.contourArea(c)
            records.append({
                "Shape":shape,
                "Area":area,
                "Perimeter":peri,
                "Vertices":len(approx),
                "Circularity":(4*np.pi*area)/(peri*peri)
            })
            cv2.drawContours(annotated,[approx],-1,(0,255,0),2)

    df = pd.DataFrame(records)
    st.columns([1,2,1])[1].image(annotated, width=650)

    # ================= COUNT OBJECTS =================
    st.header("Count Objects Dashboard")
    st.metric("Total Objects Detected", len(df))

    count_df = df["Shape"].value_counts().reset_index()
    count_df.columns = ["Shape","Count"]

    c1,c2,c3 = st.columns(3)
    c1.dataframe(count_df, use_container_width=True)
    c2.plotly_chart(px.bar(count_df, x="Shape", y="Count"), use_container_width=True, key="bar1")
    c3.plotly_chart(px.pie(count_df, names="Shape", values="Count"), use_container_width=True, key="pie1")

    # ================= AREA & PERIMETER GRAPHS =================
    st.header("Area & Perimeter Analysis")

    c1,c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="Area", title="Area Distribution"), use_container_width=True, key="area_hist")
    c2.plotly_chart(px.histogram(df, x="Perimeter", title="Perimeter Distribution"), use_container_width=True, key="peri_hist")

    st.plotly_chart(
        px.scatter(df, x="Area", y="Perimeter", color="Shape",
                   title="Relationship Between Area and Perimeter"),
        use_container_width=True, key="scatter1"
    )

    # ================= CONTOUR ANALYSIS =================
    st.header("Contours Analysis Dashboard")
    c1,c2 = st.columns(2)
    c1.image(thresh, caption="Binary Image for Contour Extraction", width=350)
    c2.image(annotated, caption="Detected Contours", width=350)

    # ================= FEATURE EXTRACTION =================
    st.header("Feature Extraction Dashboard")
    st.dataframe(df, use_container_width=True)

    numeric_df = df[["Area","Perimeter","Vertices","Circularity"]]
    st.plotly_chart(px.imshow(numeric_df.corr(), title="Feature Correlation Heatmap"),
                    use_container_width=True, key="heat1")

    # ================= FEATURE DETECTION ANALYSIS =================
    st.header("Feature Detection Analysis")

    # ---- Flat Detection
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()
    flat_img = cv2.normalize(lap**2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ---- Edge Detection
    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_strength = edges.mean()

    # ---- Harris (RED)
    gray_f = np.float32(gray)
    harris = cv2.cornerHarris(gray_f,2,3,0.04)
    harris = cv2.dilate(harris,None)
    harris_img = img.copy()
    pts = np.where(harris > 0.005*harris.max())
    for y,x in zip(pts[0],pts[1]):
        cv2.circle(harris_img,(x,y),4,(255,0,0),-1)
    harris_count = len(pts[0])

    # ---- Shi-Tomasi (GREEN)
    shi_img = img.copy()
    shi = cv2.goodFeaturesToTrack(gray,80,0.01,10)
    if shi is not None:
        for pt in shi:
            x,y = pt.ravel()
            cv2.circle(shi_img,(int(x),int(y)),6,(0,255,0),-1)
        shi_count = len(shi)
    else:
        shi_count = 0

    # ---- VISUALS
    c1,c2 = st.columns(2)
    c1.markdown("### Flat Detection (Low Texture Regions)")
    c1.image(flat_img, width=300)
    c1.markdown("**Formula:** Variance(Laplacian)")
    c1.markdown(f"**Value:** {lap_var:.2f}")

    c2.markdown("### Edge Detection (Canny)")
    c2.image(edges, width=300)
    c2.markdown("**Formula:** Gradient Magnitude")
    c2.markdown(f"**Mean Intensity:** {edge_strength:.2f}")

    st.subheader("Corner Detection")
    c1,c2 = st.columns(2)

    c1.markdown("### Harris Corner Detection")
    c1.image(harris_img, width=350)
    c1.markdown("**Formula:** R = det(M) − k(trace(M))²")
    c1.markdown(f"**Detected Corners:** {harris_count}")

    c2.markdown("### Shi-Tomasi Corner Detection")
    c2.image(shi_img, width=350)
    c2.markdown("**Formula:** R = min(λ₁, λ₂)")
    c2.markdown(f"**Detected Corners:** {shi_count}")

    # ====================================================
    # TABULAR SUMMARY: FEATURE DETECTION VALUES
    # ====================================================
    st.subheader("Feature Detection Summary (Tabular)")

    feature_table = pd.DataFrame({
        "Feature Detection Method": [
            "Flat Detection (Laplacian Variance)",
            "Edge Detection (Mean Edge Intensity)",
            "Harris Corner Detection",
            "Shi-Tomasi Corner Detection"
        ],
        "Value": [
            round(lap_var, 2),
            round(edge_strength, 2),
            harris_count,
            shi_count
        ]
    })

    st.dataframe(feature_table, use_container_width=True)


    # ---- COMPARISON GRAPH
    st.plotly_chart(
        px.bar(pd.DataFrame({
            "Detector":["Harris","Shi-Tomasi"],
            "Corners":[harris_count, shi_count]
        }), x="Detector", y="Corners",
        title="Corner Detection Comparison"),
        use_container_width=True, key="corner_cmp"
    )

    # ====================================================
    # FEATURE DETECTION METRICS GRAPH (FINAL)
    # ====================================================
    st.subheader("Feature Detection Metrics Overview")

    st.plotly_chart(
        px.bar(
            feature_table,
            x="Feature Detection Method",
            y="Value",
            title="Feature Detection Metrics Comparison"
        ),
        use_container_width=True,
        key="feature_detection_metrics"
    )

