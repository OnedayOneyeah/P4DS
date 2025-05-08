## 필요 패키지 설치
```bash
pip install openai tavily-python chromadb pandas sentence_transformers openpyxl streamlit python-dotenv streamlit-clipboard
```

## API 키 설정
1.  **`.env` 파일 생성:** 프로젝트 루트 디렉토리(예: `app.py` 파일이 있는 곳)에 `.env` 파일 생성
2.  **파일 내용 입력:** `.env` 파일에 아래 내용을 입력하고, **따옴표 안에 실제 API 키를 입력**
    ```dotenv
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    ```

## 애플리케이션 실행
```bash
streamlit run ./app.py
```