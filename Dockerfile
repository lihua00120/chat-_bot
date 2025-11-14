# 選擇一個包含您需要的 Python 版本和編譯環境的基礎映像檔
# 推薦使用 'slim' 版本以保持輕量，但需要手動安裝 Build tools
FROM python:3.13-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統級依賴：編譯器和 Python 開發頭文件
# 如果您的套件是 psycopg2，您還需要添加 libpq-dev
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    # 範例：如果您需要 PostgreSQL 連線套件
    # libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 將您的 requirements.txt 複製到容器中並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 將應用程式代碼複製到容器中
COPY . .

# 定義啟動命令
# 您的 Start Command: gunicorn app:app
CMD ["gunicorn", "app:app"]
