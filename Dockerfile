FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main
COPY . .
CMD ["poetry", "run", "python", "main.py"]