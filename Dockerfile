# 1. 基礎映像檔
FROM python:3.13-slim 

# 2. 設定工作目錄
WORKDIR /app

# 3. **關鍵：安裝系統依賴 (C 編譯器等)**
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    # 如果有特定套件，例如 psycopg2，請在這裡添加 libpq-dev
    && rm -rf /var/lib/apt/lists/*

# 4. 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製應用程式代碼
COPY . .

# 6. 啟動命令 (您的 gunicorn 命令)
CMD ["gunicorn", "app:app"]
