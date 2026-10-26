
FROM python:3.14-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files into /app
COPY . .

# Copy model files if they are stored in the models folder
# (the previous instruction copies the entire application folder)

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
