# System Reels Knowledge Phase 1

## What This Does

This backend now uses a hidden internal knowledge layer for reels-related agent tasks only.
End users do not see knowledge-pack routes, pack names, uploaded files, or retrieval snippets.

Supported task types in phase 1:

- `reel_idea`
- `reel_script`
- `reel_feedback`

Non-reels task types do not use the system knowledge layer yet.

## Internal Upload Flow

Set an internal admin key in the backend environment:

```env
INTERNAL_ADMIN_KEY=your-internal-admin-key
```

Upload internal reels knowledge through the hidden admin route:

```text
POST /api/v1/internal/knowledge-packs/upload
Header: X-Internal-Admin-Key: <INTERNAL_ADMIN_KEY>
Content-Type: multipart/form-data
```

Recommended form fields:

- `title`: internal pack title
- `description`: optional description
- `domain`: `reels`
- `supported_task_types`: `reel_idea,reel_script,reel_feedback`
- `scope`: `system`
- `visibility`: `internal`
- `status`: `active`
- `files`: one or more `.txt`, `.md`, `.pdf`, or `.docx` files

## Runtime Behavior

During `POST /api/v1/agent/chat`:

1. If `task_type` is reels-related, the backend looks for active internal system knowledge where `domain = reels`.
2. Relevant chunks are retrieved and injected into the generation prompt.
3. If no active reels knowledge exists, generation falls back safely to the existing behavior.
4. If retrieval fails, generation still continues without crashing.

## Internal Observability

Generation history storage may include internal-only metadata such as:

- `used_system_knowledge`
- `matched_knowledge_domain`
- `matched_knowledge_pack_ids`
- `retrieved_chunk_count`

These fields are kept internal and are not returned from the public generation-history endpoint.
