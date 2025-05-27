FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

COPY ./ /app/

RUN curl -o /app/wait https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait 

RUN chmod +x /app/wait

RUN echo "./wait" >> /app/prestart.sh

RUN pip install uv && uv pip install --no-cache --system -r pyproject.toml

