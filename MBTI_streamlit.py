import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA 리포트 생성 앱", layout="wide")

st.title("📊 EDA 리포트 생성 Streamlit 앱")
st.write("CSV 파일을 업로드하면 자동으로 탐색적 데이터 분석(EDA) 리포트를 생성합니다.")

# =========================================================
# 1. 데이터 업로드
# =========================================================
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
st.sidebar.markdown("---")

use_sample = st.sidebar.checkbox(
    "샘플 데이터(tips) 사용", value=(uploaded_file is None)
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    file_key = uploaded_file.name
elif use_sample:
    df = sns.load_dataset("tips")
    file_key = "tips"
else:
    st.info("CSV를 업로드하거나 샘플 데이터를 선택하세요.")
    st.stop()

# =========================================================
# session_state 초기화
# =========================================================
if "current_file" not in st.session_state:
    st.session_state.current_file = file_key
    st.session_state.original_df = df.copy()
    st.session_state.processed_df = df.copy()
elif st.session_state.current_file != file_key:
    st.session_state.current_file = file_key
    st.session_state.original_df = df.copy()
    st.session_state.processed_df = df.copy()

# =========================================================
# 2. 결측치 처리
# =========================================================
st.sidebar.header("결측치 처리")
if st.sidebar.button("원본 데이터"):
    st.session_state.processed_df = st.session_state.original_df.copy()
if st.sidebar.button("결측치 제거"):
    st.session_state.processed_df = st.session_state.processed_df.dropna()
if st.sidebar.button("중앙값으로 대체"):
    tmp = st.session_state.processed_df.copy()
    numeric_cols = tmp.select_dtypes(include="number").columns
    tmp[numeric_cols] = tmp[numeric_cols].fillna(tmp[numeric_cols].median())
    st.session_state.processed_df = tmp
if st.sidebar.button("최빈값으로 대체"):
    tmp = st.session_state.processed_df.copy()
    for col in tmp.columns:
        mode = tmp[col].mode()
        if not mode.empty:
            tmp[col] = tmp[col].fillna(mode.iloc[0])
    st.session_state.processed_df = tmp

df = st.session_state.processed_df

before_missing = st.session_state.original_df.isnull().sum().sum()
after_missing = df.isnull().sum().sum()

st.sidebar.markdown("---")
st.sidebar.metric("결측치(원본)", int(before_missing))
st.sidebar.metric("결측치(현재)", int(after_missing))
st.sidebar.metric("현재 행 개수", len(df))

# =========================================================
# 3. 데이터 개요
# =========================================================
st.header("1. 데이터 개요")
c1, c2, c3 = st.columns(3)
c1.metric("행 개수", df.shape[0])
c2.metric("열 개수", df.shape[1])
c3.metric("결측치 총합", int(df.isnull().sum().sum()))
st.dataframe(df.head())

# =========================================================
# 4. 결측치 리포트
# =========================================================
st.header("2. 결측치 확인")

missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_report = pd.DataFrame(
    {
        "결측치 개수": missing_count,
        "결측치 비율(%)": missing_pct,
    }
)
st.dataframe(missing_report.sort_values("결측치 개수", ascending=False))

# =========================================================
# 5. 기술통계
# =========================================================
st.header("3. 기술 통계")
st.dataframe(df.describe(include="all").T)

# =========================================================
# 6. 상관관계 히트맵
# =========================================================
st.header("4. 상관관계 히트맵")
numeric_df = df.select_dtypes(include="number")
if numeric_df.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
    st.pyplot(fig)
else:
    st.warning("수치형 컬럼이 2개 이상 필요합니다.")

# =========================================================
# 7. 수치형 컬럼 분포
# =========================================================
st.header("5. 수치형 컬럼 분포")

if len(numeric_df.columns) > 0:
    num_col = st.selectbox("수치형 컬럼 선택", numeric_df.columns)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[num_col], kde=True, ax=ax)
    ax.set_title(num_col)
    st.pyplot(fig)
else:
    st.warning("수치형 컬럼이 없습니다.")

# =========================================================
# 8. 범주형 컬럼 분포
# =========================================================
cat_df = df.select_dtypes(include=["object", "category", "string"])
cat_cols = cat_df.columns.tolist()
st.header("6. 범주형 컬럼 분포")

if len(cat_cols) > 0:
    cat_col = st.selectbox("범주형 컬럼 선택", cat_cols)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=cat_col, ax=ax)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
else:
    st.warning("범주형 컬럼이 없습니다.")

# =========================================================
# 9. 산점도
# =========================================================
st.header("7. 두 컬럼 간 산점도")

if len(numeric_df.columns) >= 2:
    scatter_x = st.selectbox("X축", numeric_df.columns, key="scatter_x")
    scatter_y = st.selectbox("Y축", numeric_df.columns, index=1, key="scatter_y")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x=scatter_x, y=scatter_y, alpha=0.4, ax=ax)
    ax.set_title(f"{scatter_x} vs {scatter_y}")
    st.pyplot(fig)
else:
    st.warning("산점도를 그릴 수 있는 수치형 컬럼이 부족합니다.")
# =========================================================
# 10. 데이터 필터링
# =========================================================
st.header("8. 데이터 필터링")

filter_cols = df.columns.tolist()
if len(filter_cols) > 0:
    filter_col = st.selectbox(
        "필터링할 컬럼 선택",
        filter_cols
    )
    unique_values = (
        df[filter_col]
        .dropna()
        .unique()
        .tolist()
    )
    if len(unique_values) > 0:
        selected_values = st.multiselect(
            "값 선택",
            options=sorted(unique_values),
            default=sorted(unique_values)
        )
        filtered_df = df[
            df[filter_col].isin(selected_values)
        ]
        st.write(f"필터링 결과 : {len(filtered_df)}개 행")
        st.dataframe(filtered_df)
    else:
        st.warning("선택 가능한 값이 없습니다.")

# =========================================================
# 11. 결측치 처리 결과
# =========================================================
st.header("10. 결측치 처리 결과")

result = pd.DataFrame({
    "원본": st.session_state.original_df.isnull().sum(),
    "현재": df.isnull().sum()
})
st.dataframe(result)

# =========================================================
# 12. 원본 / 현재 정보
# =========================================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("원본 데이터")
    st.metric("행 개수", st.session_state.original_df.shape[0])
    st.metric("결측치", int(st.session_state.original_df.isnull().sum().sum()))

with col2:
    st.subheader("현재 데이터")
    st.metric("행 개수", df.shape[0])
    st.metric("결측치", int(df.isnull().sum().sum()))

st.markdown("---")
st.caption(
    "AI for Future Workforce · Week 6: AI Project Deployment 실습용 앱"
)