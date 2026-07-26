# Security

## Known credential action

Source notebooks removed while this standalone repository was prepared
contained a plaintext OpenAI API credential. The removed files are not in this
Git repository or its Docker build context, but deletion does not revoke the
credential or remove it from previously shared copies.

Before publishing:

1. Revoke or rotate the exposed credential in the provider account.
2. Inspect any previously shared archive or remote history.

## Runtime secrets

Runtime credentials must be supplied through environment variables or a
deployment secret manager. Do not commit `.env`.

The application recognizes:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `LLM_MODEL`

## Network exposure

The Docker Compose configuration is intended for local development and trusted
networks. Before exposing it publicly:

- terminate TLS at a reverse proxy or managed ingress;
- authenticate and rate-limit the public API;
- keep the MCP endpoint on a private service network;
- restrict CORS to the deployed client origin;
- enable centralized logs without recording prompts or credentials;
- scan the final container and dependency lock for vulnerabilities.
