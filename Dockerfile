# 1. 基礎映像檔：使用您指定的 Python 版本
# 確保 Docker Hub 上有這個版本的映像檔
FROM python:3.13.4-slim

# 2. 設定工作目錄
WORKDIR /app

# 3. 關鍵修正：安裝系統依賴 (包括編譯 pandas 所需的工具)
# build-essential & python3-dev: 標準 C/C++ 編譯器和 Python 頭文件
# gfortran: Fortran 編譯器，pandas/numpy/scipy 需要
# libatlas-base-dev: 優化的線性代數庫 (BLAS/LAPACK)，解決數值計算依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製應用程式代碼
COPY . .

# 6. 啟動命令
CMD ["gunicorn", "app:app"]
