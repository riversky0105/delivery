import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static

# 페이지 제목
st.set_page_config(page_title="배달 위치 클러스터링", layout="wide")
st.title("📍 배달 위치 클러스터링 및 지도 시각화 (k-Means)")

# CSV 파일 업로드



    # 데이터 불러오기
df = pd.read_csv("Delivery - Delivery.csv")

    # 필수 컬럼 확인
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.success("✅ 위도(latitude), 경도(longitude) 컬럼이 확인되었습니다.")

        # 클러스터 수 선택
        k = st.slider("클러스터 수 (k)", min_value=1, max_value=10, value=3)

        # k-Means 클러스터링
        coords = df[['latitude', 'longitude']]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(coords)

        # 지도 중심 위치 계산
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()

        # Folium 지도 생성
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # 클러스터 색상
        colors = [
            'red', 'blue', 'green', 'purple', 'orange',
            'darkred', 'lightblue', 'darkgreen', 'cadetblue', 'black'
        ]

        # 각 지점 지도에 표시
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=colors[row['cluster'] % len(colors)],
                fill=True,
                fill_opacity=0.7,
                popup=f"Cluster: {row['cluster']}"
            ).add_to(m)

        # 지도 표시
        st.subheader("🗺️ 클러스터링 결과 지도")
        folium_static(m)

        # 데이터프레임 출력
        with st.expander("📊 클러스터링된 데이터 보기"):
            st.dataframe(df)
    else:
        st.error("❌ CSV 파일에 'latitude'와 'longitude' 컬럼이 존재해야 합니다.")
else:
    st.info("👆 좌측 상단에서 CSV 파일을 업로드하세요.")


   
