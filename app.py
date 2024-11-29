import streamlit as st
import toml
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt


# 설정 파일 로드
def load_config(config_path='config.toml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    return config


# 번역 파일 로드
def load_translations(lang='ko', translations_dir='translations'):
    lang_file = os.path.join(translations_dir, f"{lang}.json")
    if not os.path.exists(lang_file):
        lang_file = os.path.join(translations_dir, "en-US.json")  # 기본 언어 설정
    with open(lang_file, 'r', encoding='utf-8') as f:
        translations = json.load(f)
    return translations


# 다국어 지원을 위한 함수
def translate(key, translations):
    keys = key.split('.')
    value = translations
    try:
        for k in keys:
            value = value[k]
        if isinstance(value, str):
            return value
        else:
            return key  # 키가 문자열이 아닐 경우 키 자체 반환
    except KeyError:
        return key  # 키가 존재하지 않을 경우 키 자체 반환


# 설정 및 번역 파일 로드
config = load_config()

# 초기값 설정
if "language" not in st.session_state:
    st.session_state.language = config.get('meta', {}).get('language', 'ko')  # 기본 언어를 세션 상태에 저장

translations = load_translations(lang=st.session_state.language)

# 사용자 정의 CSS 적용
custom_css_path = config.get('UI', {}).get('custom_css', '')
if custom_css_path and os.path.exists(custom_css_path):
    with open(custom_css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# CSS로 중앙 정렬 및 색상 설정
st.markdown(
    """
    <style>
    .center-title {
        text-align: center;
        color: #FFA07A; /* Light Salmon: 연한 주황색 */
        font-size: 2.5em; /* 제목 크기 조정 */
        font-weight: bold;
    }
    .center-text {
        text-align: center;
        color: #FFA07A; /* Light Salmon: 연한 주황색 */
        font-size: 1.8em; /* 문구 크기 조정 */
        font-weight: bold;
    }
    .sidebar-title {
        text-align: center;
        color: #FFA07A; /* Light Salmon: 연한 주황색 */
        font-size: 2.7em; /* 사이드바 제목 크기 */
        font-weight: bold;
    }
    .sidebar-footer {
        display: flex;
        justify-content: flex-end; /* 오른쪽 정렬 */
        width: 100%;
        padding: 10px 0;
    }
    .sidebar-footer button {
        margin-right: 10px; /* 오른쪽 여백 설정 */
    }
    .stButton > button {
        color: white; /* 글자 색상 */
        background-color: #FF6347; /* Tomato: 버튼 배경 색상 */
        border-radius: 10px; /* 버튼 모서리 둥글게 */
        border: none; /* 버튼 테두리 제거 */
        padding: 10px 20px; /* 버튼 여백 */
        font-size: 16px; /* 버튼 글자 크기 */
        font-weight: bold;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #FF4500; /* Darker Tomato: 호버 시 배경 색상 */
        color: white; /* 호버 시 글자 색상 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 사이드바 생성
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🍊 Bitamin Winter Project ☃️</h2>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

    # 언어 선택
    language_options = {
        "한국어": "ko",
        "English": "en-US"
    }
    selected_language = st.selectbox("언어 선택", list(language_options.keys()), index=list(language_options.values()).index(st.session_state.language))

    # 언어 변경 처리
    if language_options[selected_language] != st.session_state.language:
        st.session_state.language = language_options[selected_language]
        translations = load_translations(lang=st.session_state.language)  # 새로운 언어 로드
        config['meta']['language'] = st.session_state.language
        with open('config.toml', 'w', encoding='utf-8') as f:
            toml.dump(config, f)
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    # 파일 업로드 설정
    st.subheader("파일 업로드 설정")
    upload_enabled = st.checkbox("파일 업로드 활성화", value=config['features']['spontaneous_file_upload']['enabled'])

    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

    # 앱 종료 버튼 추가
    # 앱 종료 버튼 (사이드바 하단 고정)
    st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    if st.button("앱 종료"):
        st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

# 애플리케이션 제목 설정 (중앙 정렬 및 색상 적용)
st.markdown("<h1 class='center-title'>🍊 Bitamin Winter Project ☃️</h1>", unsafe_allow_html=True)

# 파일 업로드 위젯
if upload_enabled:
    uploaded_file = st.file_uploader("분석할 파일을 업로드하세요 (CSV, TXT, JSON 지원)", type=["csv", "txt", "json"])
    if uploaded_file is not None:
        st.write("업로드된 파일을 처리 중입니다...")
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            st.write("파일 처리가 완료되었습니다!")
        except Exception as e:
            st.error("파일 처리 중 오류가 발생했습니다.")

# 혐오 표현 감지 테스트 문구와 UI
if config['features'].get('prompt_playground', False):
    st.markdown("<h2 class='center-text'>혐오 표현 감지 테스트</h2>", unsafe_allow_html=True)

    # 세션 상태 초기화 (결과 상태 관리)
    if "result" not in st.session_state:
        st.session_state.result = None

    # 입력 창 및 분석 버튼
    # 입력 창 및 분석 버튼
    if st.session_state.result is None:  # 결과가 없는 초기 상태
        user_input = st.text_area("혐오 표현인지 확인하고 싶은 문장을 입력하세요:")
        model_path = "./model"  # 모델이 저장된 경로
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # 모델 및 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

        # 분석 버튼
        if st.button("분석하기"):
            if user_input.strip():  # 입력 문장이 비어있지 않을 경우
                # 입력 문장 토큰화
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)

                # 모델 예측
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=1)
                    confidence, label = torch.max(predictions, dim=1)  # 가장 높은 확률의 라벨

                # 결과 저장
                if label.item() == 1:  # 1: 혐오 표현으로 분류된 경우
                    st.session_state.result = f"입력한 문장 \"{user_input}\" 은 혐오 표현으로 감지되었습니다."
                else:  # 0: 혐오 표현이 아닌 경우
                    st.session_state.result = f"입력한 문장 \"{user_input}\" 은 혐오 표현이 아닙니다."
            else:
                st.warning("문장을 입력해주세요.")
    else:
        # 결과 표시
        st.write(st.session_state.result)

        # 새로운 입력 버튼
        if st.button("새로운 입력"):
            st.session_state.result = None  # 결과 초기화

