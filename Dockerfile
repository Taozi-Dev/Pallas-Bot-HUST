FROM sunpeek/poetry:py3.9-slim as requirements-stage

WORKDIR /tmp

COPY ./pyproject.toml ./poetry.lock* /tmp/

ENV PATH="${PATH}:/root/.local/bin"

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources &&  apt-get update && apt-get install -y ffmpeg

RUN curl -o /app/wait https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait 

RUN chmod +x /app/wait

RUN echo "./wait" >> /app/prestart.sh

COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN rm requirements.txt

COPY ./ /app/
