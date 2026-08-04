# Safe Langflow Reels RAG

## Architecture

Backend:
- authenticates users
- enforces billing, limits, and connection safety
- builds compact Instagram context
- stores generation history and safe metadata
- hides internal knowledge from public responses

Langflow:
- ingests internal reels files through an admin-only ingestion flow
- stores chunked reels knowledge in a vector store
- retrieves top relevant reels chunks at runtime
- generates the final reels answer and structured output

## Runtime Payload Rules

The backend sends only sanitized runtime variables to Langflow for:
- `reel_idea`
- `reel_script`
- `reel_feedback`

It does not send:
- raw Mariami files
- full hidden prompt
- flattened system rules
- raw retrieved chunks
- full recent captions

## Current Environment

- `USE_LANGFLOW_FOR_AGENT_CHAT=false`
- `USE_LANGFLOW_SAFE_REELS_RAG=true`
- reels generation flow id: `fac6d2c7-9de1-465a-b5b9-bc0132713be7`
- reels ingestion flow id: `87596645-dff9-4b7d-9d76-2dd16f6a5093`
- vector store provider: `chroma`
- chroma collection: `reels_system_knowledge`
- model provider: `groq`
- model name: `llama-3.3-70b-versatile`

## Internal Upload Path

Admin-only:

`POST /api/v1/internal/knowledge-packs/upload`

Required form values:
- `title`
- `description`
- `domain=reels`
- `supported_task_types=reel_idea,reel_script,reel_feedback`
- `scope=system`
- `visibility=internal`
- `status=active`
- `files`

The current production-safe upload source file used for ingestion is:

`C:\Users\Greench Pc\Desktop\instagram-agent\Reels იდეების მოძიება.docx`

## Safety Notes

- If the configured Langflow generation flow does not return the expected structured reels contract, the backend now fails fast with a safe internal-flow error instead of returning a generic greeting.
- The legacy 4-node chat flow still exists in Langflow as `cba69b38-9d4c-4547-8bd7-76090118e547`, but reels production generation no longer points to it.
- The current reels flows are custom-component based:
  - ingestion: `Text Input -> Reels Knowledge Ingestion -> Chat Output`
  - generation: `Text Input -> Reels RAG Generation -> Chat Output`
- Backend-to-Langflow runtime payload is now base64-encoded UTF-8 JSON to avoid Unicode loss in traces while still sending only sanitized fields.
