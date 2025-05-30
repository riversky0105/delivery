import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

# Streamlit 앱 제목
st.title("배달 지점 군집화 지도 시각화")
st.markdown(
    """
    이 앱은 업로드한 위치 데이터를 바탕으로 **k-Means 군집화**를 수행하고, 
    **지도 위에 시각화**하여 보여줍니다.
    """
)

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    # CSV 파일 읽기
    df = pd.read_csv(uploaded_file)

    # 데이터 확인
    st.subheader("📋 데이터 미리보기")
    st.write(df.head())

    # 컬럼 선택 (위도와 경도)
    st.subheader("🧭 위치 정보 컬럼 선택")
    lat_col = st.selectbox("위도 (latitude) 컬럼 선택", df.columns)
    lon_col = st.selectbox("경도 (longitude) 컬럼 선택", df.columns)

    # 클러스터 개수 선택
    st.subheader("🔢 군집 수(k) 선택")
    n_clusters = st.slider("k 값 (군집 수)", min_value=1, max_value=10, value=3)

    # KMeans 군집화 수행
    coords = df[[lat_col, lon_col]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)

    # 지도 중심 설정
    center_lat = coords[lat_col].mean()
    center_lon = coords[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # 군집별 색상 생성 함수
    def get_color(cluster_num):
        colors = [
            "red", "blue", "green", "purple", "orange",
            "darkred", "lightblue", "lightgreen", "cadetblue", "pink"
        ]
        return colors[cluster_num % len(colors)]

    # 각 지점을 지도에 마커로 추가
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=get_color(row['cluster']),
            fill=True,
            fill_opacity=0.7,
            popup=f"Cluster {row['cluster']}"
        ).add_to(m)

    # 군집 중심 좌표 표시
    st.subheader("📍 군집 중심 표시")
    for idx, center in enumerate(kmeans.cluster_centers_):
        folium.Marker(
            location=[center[0], center[1]],
            icon=folium.Icon(color='black', icon='info-sign'),
            popup=f"Cluster {idx} 중심"
        ).add_to(m)

    # 지도 표시
    st.subheader("🗺️ 군집화 결과 지도")
    st_folium(m, width=700, height=500)

    # 결과 다운로드 (선택)
    st.subheader("💾 군집 결과 다운로드")
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("결과 CSV 다운로드", data=csv, file_name="clustered_data.csv", mime="text/csv")

else:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하세요.")
