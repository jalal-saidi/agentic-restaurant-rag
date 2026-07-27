FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home "${USERNAME}" \
    && python -m venv "${VIRTUAL_ENV}"

WORKDIR /opt/connoisseur-build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[dev]" \
    && mkdir -p \
        /workspaces/agentic-restaurant-rag \
        /home/${USERNAME}/.cache/chroma \
        /home/${USERNAME}/.local/share/connoisseur/chroma \
    && chown -R "${USERNAME}:${USERNAME}" \
        "${VIRTUAL_ENV}" \
        /workspaces \
        /home/${USERNAME}

USER ${USERNAME}
WORKDIR /workspaces/agentic-restaurant-rag

CMD ["sleep", "infinity"]
