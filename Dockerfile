# Learning Path Generator - MCP server (streamable-http mode)
#
# Builds a container that runs mcp_server.py as a network-reachable service,
# instead of the local stdio process Claude Desktop launches directly.
# See COMPARISON.md / README.md for why this exists alongside the local
# Claude Desktop setup rather than replacing it.

FROM python:3.11-slim

# System dependency needed by pypdf/some chromadb internals for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer from app code), so
# Docker can reuse this layer on rebuilds unless requirements change -
# much faster iteration than reinstalling everything on every code edit.
COPY requirements-mcp.txt .
RUN pip install --no-cache-dir -r requirements-mcp.txt

# Now copy the actual application code and the sample documents used by
# "documents" mode.
COPY learning_path_core.py .
COPY mcp_server.py .
COPY my_documents/ ./my_documents/

# Pre-download the local embeddings model at build time, not at container
# startup - avoids a network call (and the timeout risk we hit with Claude
# Desktop) every time the container starts, and keeps startup fast.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Run fully offline from here on - use the model files baked into the
# image, never touch the network for this.
ENV HF_HUB_OFFLINE=1

# Switch this server into network-service mode (see mcp_server.py's
# transport switch). Claude Desktop's own local config never sets this,
# so this only affects the containerized copy.
ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

EXPOSE 8000

# GROQ_API_KEY and TAVILY_API_KEY are intentionally NOT set here - pass
# them at run time with `docker run --env-file .env ...` so real keys
# never get baked into the image itself.
CMD ["python", "mcp_server.py"]
