# BigQuery → PostgreSQL Migration: Complete Change Map

## Repository Confirmation

I have access to all files. Here's what's in scope:

| Service | File | BQ Calls | Role |
|---------|------|----------|------|
| `drywall-takeoff-3d` | `main.py` | **~35 queries** | Primary backend — all CRUD endpoints |
| `drywall-takeoff-3d` | `helper.py` | **~4 queries** | Shared functions: `bigquery_run`, `insert_model_2d`, `is_duplicate`, `delete_plan` |
| `drywall-takeoff-3d` | `config/gcp.yaml` | Config | BQ table names, KMS key |
| `floorplan-to-structured-2d` | `main.py` | **1 query** (via helper) | Calls `insert_model_2d` after LLM processing |
| `floorplan-to-structured-2d` | `helper.py` | **~3 queries** | `bigquery_run`, `insert_model_2d`, `load_bigquery_client` |
| `floorplan-to-structured-2d` | `gcp.yaml` | Config | Same BQ table names |
| `wall-detector` | — | **0 queries** | No BQ usage — uses GCS only, no changes needed |

**Files that do NOT need changes:** `floor_plan.py`, `extrapolate_3d.py`, `gltf_generator.py`, `preprocessing.py`, `prompt.py`, `modeller_2d.py`, `transcriber.py`, `wall_detector.py` — these have zero database interaction.

---

## LAYER 0: Foundation — Connection & Config Changes

### Change 0.1: `requirements.txt` (both services)

**File:** `drywall-takeoff-3d/requirements.txt`

**Current:**
```
google-cloud-bigquery
google-cloud-bigquery-storage
db-dtypes
```

**New — add:**
```
asyncpg
```

**New — remove (after full migration):**
```
# google-cloud-bigquery        ← remove
# google-cloud-bigquery-storage ← remove
# db-dtypes                     ← remove
```

**Why:** `asyncpg` is the PostgreSQL async driver. It replaces the BigQuery client library. Keep BQ libraries during transition, remove after testing.

**Same change for:** `floorplan-to-structured-2d/requirements.txt`

---

### Change 0.2: `Dockerfile` (both services)

**File:** `drywall-takeoff-3d/Dockerfile`

**Current:** Already has `libpq-dev` installed (line 13) — this is the PostgreSQL C client library.

**No change needed.** Your Dockerfiles already include `libpq-dev`. This was likely added for another dependency but it's exactly what `asyncpg` needs.

---

### Change 0.3: `config/gcp.yaml` — Add PostgreSQL config

**File:** `drywall-takeoff-3d/config/gcp.yaml`

**Current:**
```yaml
GBQServer:
    service_account_key: *service_compute_account_key
    KMS_key: projects/prj-fbm-drywall-dev/locations/us/keyRings/...
    table_name_projects: drywall_takeoff.projects
    table_name_plans: drywall_takeoff.plans
    table_name_models: drywall_takeoff.models
    table_name_model_revisions_2d: drywall_takeoff.model_revisions_2d
    table_name_model_revisions_3d: drywall_takeoff.model_revisions_3d
```

**New — add this block:**
```yaml
PostgreSQL:
    host: 10.x.x.x              # Your Cloud SQL private IP from GenAI-Project
    port: 5432
    database: drywall_takeoff
    user: postgres
    password: YOUR_PASSWORD       # Move to Secret Manager later
```

**Why:** Your app loads config from `gcp.yaml` via `load_gcp_credentials()`. Adding the PG block here means all connection details come from the same config file. The existing `GBQServer` block stays until migration is complete.

**Same change for:** `floorplan-to-structured-2d/gcp.yaml`

---

### Change 0.4: `drywall-takeoff-3d/helper.py` — Replace `load_bigquery_client` and `bigquery_run`

These two functions are the **foundation** — every BQ query in both services flows through them.

#### Change 0.4a: Replace `load_bigquery_client`

**File:** `drywall-takeoff-3d/helper.py` (lines 11-12)

**Current code:**
```python
from google.cloud import bigquery

def load_bigquery_client(credentials):
    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    return bigquery_client
```

**New code:**
```python
import asyncpg
import asyncio

_pg_pool = None

async def get_pg_pool(credentials):
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(
            host=credentials["PostgreSQL"]["host"],
            port=credentials["PostgreSQL"]["port"],
            database=credentials["PostgreSQL"]["database"],
            user=credentials["PostgreSQL"]["user"],
            password=credentials["PostgreSQL"]["password"],
            min_size=2,
            max_size=10
        )
    return _pg_pool

def load_pg_pool_sync(credentials):
    """Synchronous wrapper for startup — creates the pool once."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_pg_pool(credentials))
```

**Why:** BigQuery uses a stateless client (new HTTP request each query). PostgreSQL uses a **connection pool** — a set of persistent connections reused across requests. `asyncpg` is async-native, which matches your FastAPI async endpoints perfectly. Pool is created once at app startup, shared across all requests.

#### Change 0.4b: Remove `bigquery_run`

**File:** `drywall-takeoff-3d/helper.py` (lines 14-22)

**Current code:**
```python
def bigquery_run(credentials, bigquery_client, GBQ_query, job_config=dict()):
    job_config = bigquery.QueryJobConfig(
        destination_encryption_configuration=bigquery.EncryptionConfiguration(
            kms_key_name=credentials["GBQServer"]["KMS_key"]
        ),
        **job_config
    )
    query_output = bigquery_client.query(GBQ_query, job_config=job_config)
    return query_output
```

**New code:** DELETE THIS FUNCTION ENTIRELY.

**Why:** `bigquery_run` was a wrapper that attached KMS encryption config to every query. PostgreSQL handles encryption differently (SSL/TLS on the connection itself, configured once in the pool). Each query will now call `pool.fetch()`, `pool.fetchrow()`, or `pool.execute()` directly. There's no need for a central wrapper.

---

### Change 0.5: `drywall-takeoff-3d/main.py` — Replace startup initialization

**File:** `drywall-takeoff-3d/main.py` (lines ~430-432)

**Current code:**
```python
CREDENTIALS = load_gcp_credentials()
# ... middleware setup ...
bigquery_client = load_bigquery_client(CREDENTIALS)
```

**New code:**
```python
CREDENTIALS = load_gcp_credentials()
# ... middleware setup ...
pg_pool = None  # initialized in startup event

@app.on_event("startup")
async def startup():
    global pg_pool
    pg_pool = await get_pg_pool(CREDENTIALS)

@app.on_event("shutdown")
async def shutdown():
    global pg_pool
    if pg_pool:
        await pg_pool.close()
```

**Why:** `asyncpg.create_pool()` is async and must run inside an async context. FastAPI's `startup` event is the right place. The pool opens connections at startup and closes them at shutdown. Every endpoint uses the shared `pg_pool` instead of `bigquery_client`.

---

## LAYER 1: `drywall-takeoff-3d/helper.py` — Shared Functions

### Change 1.1: `insert_model_2d` function

**File:** `drywall-takeoff-3d/helper.py` (lines 47-100)

**Current code:**
```python
def insert_model_2d(
    model_2d, scale, page_number, plan_id, user_id, project_id,
    GCS_URL_floorplan_page, GCS_URL_target_drywalls_page,
    bigquery_client, credentials
):
    if not model_2d.get("metadata", None):
        GBQ_query = f"SELECT model_2d.metadata FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
        query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
        metadata = list(query_output)[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    GBQ_query = """
    MERGE `drywall_takeoff.models` t
    USING (...) s
    ON LOWER(t.project_id) = LOWER(s.project_id) AND LOWER(t.plan_id) = LOWER(s.plan_id) AND t.page_number = s.page_number
    WHEN MATCHED THEN UPDATE SET ...
    WHEN NOT MATCHED THEN INSERT ...
    """
    job_config = dict(query_parameters=[
        bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
        bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
        # ... more params
    ])
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output
```

**New code:**
```python
async def insert_model_2d(
    model_2d, scale, page_number, plan_id, user_id, project_id,
    GCS_URL_floorplan_page, GCS_URL_target_drywalls_page,
    pg_pool, credentials
):
    if not model_2d.get("metadata", None):
        row = await pg_pool.fetchrow(
            """SELECT model_data->'metadata' AS metadata FROM models
               WHERE LOWER(project_id) = LOWER($1)
               AND LOWER(plan_id) = LOWER($2)
               AND page_number = $3""",
            project_id, plan_id, page_number
        )
        if row and row["metadata"]:
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            model_2d["metadata"] = metadata

    await pg_pool.execute(
        """INSERT INTO models (plan_id, project_id, user_id, page_number, scale,
               model_data, source, target_drywalls, created_at, updated_at)
           VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, now(), now())
           ON CONFLICT (project_id, plan_id, page_number)
           DO UPDATE SET
               model_data = $6::jsonb,
               scale = COALESCE(NULLIF($5, ''), models.scale),
               user_id = $3,
               updated_at = now()""",
        plan_id, project_id, user_id, page_number, scale,
        json.dumps(model_2d), GCS_URL_floorplan_page, GCS_URL_target_drywalls_page
    )
```

**Why:**
- BigQuery `MERGE` → PostgreSQL `INSERT ... ON CONFLICT DO UPDATE` (same logic, different syntax)
- f-string SQL injection `LOWER('{project_id}')` → parameterized `LOWER($1)` (fixes security vulnerability)
- `bigquery.ScalarQueryParameter` → positional `$1, $2, $3` params (simpler)
- Function becomes `async` because `asyncpg` is async
- `model_2d` JSON is stored as the full blob in `model_data` JSONB column
- Metadata extraction uses `->` JSON operator on the JSONB column

**CRITICAL NOTE:** This function currently stores `model_2d` (containing `walls_2d`, `polygons`, `metadata`) as the model_data. In the new schema, this maps to the `model_data` column of your `models_2d` table. However, the current BigQuery schema has SEPARATE columns for `model_2d`, `model_3d`, and `takeoff`. For the single-table approach, you'll store ALL of these inside the same JSONB blob, OR create the `models` table with separate JSONB columns:

```sql
CREATE TABLE models (
    project_id      TEXT NOT NULL,
    plan_id         TEXT NOT NULL,
    user_id         TEXT,
    page_number     INTEGER NOT NULL DEFAULT 0,
    scale           TEXT,
    source          TEXT,
    target_drywalls TEXT,
    model_2d        JSONB,
    model_3d        JSONB,
    takeoff         JSONB,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(project_id, plan_id, page_number)
);
```

I recommend this multi-column approach because it matches your current BigQuery schema exactly and minimizes code changes.

---

### Change 1.2: `is_duplicate` function

**File:** `drywall-takeoff-3d/helper.py` (lines 102-114)

**Current code:**
```python
def is_duplicate(bigquery_client, credentials, pdf_path, project_id):
    sha_256 = sha256(pdf_path)
    GBQ_query = f"SELECT plan_id, sha256, status FROM `drywall_takeoff.plans` WHERE LOWER(project_id) = LOWER('{project_id}');"
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
    for plan_target in list(query_output):
        if not plan_target.sha256:
            delete_plan(credentials, bigquery_client, plan_target.plan_id, project_id)
            continue
        if plan_target.sha256 == sha_256:
            if plan_target.status == "FAILED":
                delete_plan(credentials, bigquery_client, plan_target.plan_id, project_id)
                return False
            return plan_target.plan_id
    return False
```

**New code:**
```python
async def is_duplicate(pg_pool, credentials, pdf_path, project_id):
    sha_256 = sha256(pdf_path)
    rows = await pg_pool.fetch(
        "SELECT plan_id, sha256, status FROM plans WHERE LOWER(project_id) = LOWER($1)",
        project_id
    )
    for plan_target in rows:
        if not plan_target["sha256"]:
            await delete_plan(credentials, pg_pool, plan_target["plan_id"], project_id)
            continue
        if plan_target["sha256"] == sha_256:
            if plan_target["status"] == "FAILED":
                await delete_plan(credentials, pg_pool, plan_target["plan_id"], project_id)
                return False
            return plan_target["plan_id"]
    return False
```

**Why:**
- SQL injection fix: `LOWER('{project_id}')` → `LOWER($1)`
- `bigquery_client` → `pg_pool`
- `.result()` iteration → `await pg_pool.fetch()` returns list of Record objects
- Row access: `plan_target.sha256` → `plan_target["sha256"]` (asyncpg uses dict-style access)

---

### Change 1.3: `delete_plan` function

**File:** `drywall-takeoff-3d/helper.py` (lines 116-119)

**Current code:**
```python
def delete_plan(credentials, bigquery_client, plan_id, project_id):
    GBQ_query = f"DELETE FROM `drywall_takeoff.plans` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
    return query_output
```

**New code:**
```python
async def delete_plan(credentials, pg_pool, plan_id, project_id):
    await pg_pool.execute(
        "DELETE FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        project_id, plan_id
    )
```

**Why:** Direct f-string SQL → parameterized. `bigquery_run` → `pool.execute`.

---

## LAYER 2: `drywall-takeoff-3d/main.py` — All Endpoints

### Change 2.1: `insert_model_2d_revision` function

**File:** `drywall-takeoff-3d/main.py` (lines ~65-120)

**Current code:**
```python
def insert_model_2d_revision(model_2d, scale, page_number, plan_id, user_id, project_id, bigquery_client, credentials):
    if not model_2d.get("metadata", None):
        GBQ_query = f"SELECT model_2d.metadata FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
        query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
        metadata = list(query_output)[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    GBQ_query = """
    SELECT MAX(revision_number) AS revision_number FROM `drywall_takeoff.model_revisions_2d` WHERE 
    LOWER(project_id) = LOWER(@project_id) AND LOWER(plan_id) = LOWER(@plan_id) AND page_number = @page_number;
    """
    # ... get max revision, then INSERT new revision
```

**New code:**
```python
async def insert_model_2d_revision(model_2d, scale, page_number, plan_id, user_id, project_id, pg_pool, credentials):
    if not model_2d.get("metadata", None):
        row = await pg_pool.fetchrow(
            """SELECT model_2d->'metadata' AS metadata FROM models
               WHERE LOWER(project_id) = LOWER($1)
               AND LOWER(plan_id) = LOWER($2) AND page_number = $3""",
            project_id, plan_id, page_number
        )
        if row and row["metadata"]:
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            model_2d["metadata"] = metadata

    row = await pg_pool.fetchrow(
        """SELECT COALESCE(MAX(revision_number), 0) AS rev
           FROM model_revisions_2d
           WHERE LOWER(project_id) = LOWER($1)
           AND LOWER(plan_id) = LOWER($2) AND page_number = $3""",
        project_id, plan_id, page_number
    )
    revision_number = row["rev"] + 1

    await pg_pool.execute(
        """INSERT INTO model_revisions_2d
           (plan_id, project_id, user_id, page_number, scale, model, created_at, revision_number)
           VALUES ($1, $2, $3, $4, $5, $6::jsonb, now(), $7)""",
        plan_id, project_id, user_id, page_number, scale,
        json.dumps(model_2d), revision_number
    )
```

**Why:** Same pattern — MERGE→ON CONFLICT, f-strings→params, sync→async. The `COALESCE(MAX(...), 0)` avoids the null check that the original does with an if/else.

---

### Change 2.2: `insert_model_3d_revision` function

**File:** `drywall-takeoff-3d/main.py` (lines ~122-175)

**Current code:** Same pattern as 2d revision — gets MAX revision_number, then INSERTs.

**New code:**
```python
async def insert_model_3d_revision(model_3d, scale, page_number, plan_id, user_id, project_id, pg_pool, credentials):
    row = await pg_pool.fetchrow(
        """SELECT COALESCE(MAX(revision_number), 0) AS rev
           FROM model_revisions_3d
           WHERE LOWER(project_id) = LOWER($1)
           AND LOWER(plan_id) = LOWER($2) AND page_number = $3""",
        project_id, plan_id, page_number
    )
    revision_number = row["rev"] + 1

    await pg_pool.execute(
        """INSERT INTO model_revisions_3d
           (plan_id, project_id, user_id, page_number, scale, model, takeoff, created_at, revision_number)
           VALUES ($1, $2, $3, $4, $5, $6::jsonb, '{}'::jsonb, now(), $7)""",
        plan_id, project_id, user_id, page_number, scale,
        json.dumps(model_3d), revision_number
    )
```

---

### Change 2.3: `insert_model_3d` function

**File:** `drywall-takeoff-3d/main.py` (lines ~177-205)

**Current code:**
```python
def insert_model_3d(model_3d, scale, page_number, plan_id, user_id, project_id, bigquery_client, credentials):
    GBQ_query = """
    UPDATE `drywall_takeoff.models` as t
    SET model_3d = @model_3d, scale = COALESCE(NULLIF(@scale, ''), t.scale),
        user_id = @user_id, updated_at = CURRENT_TIMESTAMP()
    WHERE LOWER(project_id) = LOWER(@project_id) AND LOWER(plan_id) = LOWER(@plan_id) AND page_number = @page_number
    """
    job_config = dict(query_parameters=[...])
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
```

**New code:**
```python
async def insert_model_3d(model_3d, scale, page_number, plan_id, user_id, project_id, pg_pool, credentials):
    await pg_pool.execute(
        """UPDATE models SET
               model_3d = $1::jsonb,
               scale = COALESCE(NULLIF($2, ''), scale),
               user_id = $3,
               updated_at = now()
           WHERE LOWER(project_id) = LOWER($4)
           AND LOWER(plan_id) = LOWER($5)
           AND page_number = $6""",
        json.dumps(model_3d), scale, user_id, project_id, plan_id, page_number
    )
```

---

### Change 2.4: `delete_floorplan` function

**File:** `drywall-takeoff-3d/main.py` (lines ~207-252)

**Current code:** 4 separate DELETE queries + GCS blob deletion
```python
def delete_floorplan(project_id, plan_id, user_id, bigquery_client, credentials):
    GBQ_query = """DELETE FROM `drywall_takeoff.plans` WHERE ..."""
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    GBQ_query = """DELETE FROM `drywall_takeoff.models` WHERE ..."""
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    GBQ_query = """DELETE FROM `drywall_takeoff.model_revisions_2d` WHERE ..."""
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    GBQ_query = """DELETE FROM `drywall_takeoff.model_revisions_3d` WHERE ..."""
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    # ... GCS deletion stays same
```

**New code:**
```python
async def delete_floorplan(project_id, plan_id, user_id, pg_pool, credentials):
    # Use a transaction — all deletes succeed or none do
    async with pg_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM plans WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
            await conn.execute(
                "DELETE FROM models WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
            await conn.execute(
                "DELETE FROM model_revisions_2d WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
            await conn.execute(
                "DELETE FROM model_revisions_3d WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
    # GCS deletion — NO CHANGE, stays exactly as-is
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    prefix = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)
```

**Why:** Wrapping in a transaction is a **major improvement**. Currently if the 3rd DELETE fails in BigQuery, you've already deleted plans and models but not revisions — leaving orphaned data. With a PostgreSQL transaction, either all 4 DELETEs succeed or none of them execute.

---

### Change 2.5: `insert_takeoff` function

**File:** `drywall-takeoff-3d/main.py` (lines ~254-300)

**Current code:** Two UPDATE queries — one on `models`, one on `model_revisions_3d`.

**New code:**
```python
async def insert_takeoff(takeoff, page_number, plan_id, user_id, project_id, revision_number, pg_pool, credentials):
    await pg_pool.execute(
        """UPDATE models SET takeoff = $1::jsonb, updated_at = now(), user_id = $2
           WHERE LOWER(project_id)=LOWER($3) AND LOWER(plan_id)=LOWER($4) AND page_number=$5""",
        json.dumps(takeoff), user_id, project_id, plan_id, page_number
    )
    if revision_number:
        await pg_pool.execute(
            """UPDATE model_revisions_3d SET takeoff = $1::jsonb, user_id = $2
               WHERE LOWER(project_id)=LOWER($3) AND LOWER(plan_id)=LOWER($4)
               AND page_number=$5 AND revision_number=$6""",
            json.dumps(takeoff), user_id, project_id, plan_id, page_number, revision_number
        )
```

---

### Change 2.6: `insert_plan` function

**File:** `drywall-takeoff-3d/main.py` (lines ~302-380)

**Current code:** BigQuery MERGE with ~11 parameters

**New code:**
```python
async def insert_plan(
    project_id, user_id, status, pg_pool, credentials,
    payload_plan=None, plan_id=None, size_in_bytes=None,
    GCS_URL_floorplan=None, n_pages=None
):
    sha_256 = ''
    if plan_id:
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(user_id, plan_id, project_id, credentials, destination_path=pdf_path)
        sha_256 = sha256(pdf_path)
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name = payload_plan.plan_name if payload_plan else ''
    plan_type = payload_plan.plan_type if payload_plan else ''
    file_type = payload_plan.file_type if payload_plan else ''
    n_pages = n_pages or 0
    GCS_URL_floorplan = GCS_URL_floorplan or ''
    size_in_bytes = size_in_bytes or 0

    await pg_pool.execute(
        """INSERT INTO plans (plan_id, project_id, user_id, status, plan_name, plan_type,
               file_type, pages, size_in_bytes, source, sha256, created_at, updated_at)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, now(), now())
           ON CONFLICT (project_id, plan_id)
           DO UPDATE SET
               pages=$8, source=$10, sha256=$11, status=$4,
               size_in_bytes=$9, user_id=$3, updated_at=now()""",
        plan_id, project_id, user_id, status, plan_name, plan_type,
        file_type, n_pages, size_in_bytes, GCS_URL_floorplan, sha_256
    )
```

---

### Change 2.7: `insert_project` function

**File:** `drywall-takeoff-3d/main.py` (lines ~382-425)

**Current code:** BigQuery MERGE + follow-up SELECT for `created_at`

**New code:**
```python
async def insert_project(payload_project, pg_pool, credentials):
    row = await pg_pool.fetchrow(
        """INSERT INTO projects (project_id, project_name, project_location, fbm_branch,
               project_type, project_area, contractor_name, created_at, created_by)
           VALUES ($1,$2,$3,$4,$5,$6,$7, now(), $8)
           ON CONFLICT (project_id) DO NOTHING
           RETURNING created_at""",
        payload_project.project_id, payload_project.project_name,
        payload_project.project_location, payload_project.FBM_branch,
        payload_project.project_type, payload_project.project_area,
        payload_project.contractor_name, payload_project.created_by
    )
    if row:
        return row["created_at"].isoformat()
    # If already existed (DO NOTHING), fetch created_at
    row = await pg_pool.fetchrow(
        "SELECT created_at FROM projects WHERE project_id = $1",
        payload_project.project_id
    )
    return row["created_at"].isoformat()
```

**Why:** BigQuery MERGE with `WHEN NOT MATCHED THEN INSERT` → PostgreSQL `INSERT ... ON CONFLICT DO NOTHING`. The `RETURNING` clause gives us the `created_at` without a second query when it's a new insert. Only falls back to SELECT when project already exists.

---

### Change 2.8: `/generate_project` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~450-465)

**Current code:**
```python
@app.post("/generate_project")
async def generate_project(request: Request):
    # ... parse params ...
    created_at = insert_project(payload_project, bigquery_client, CREDENTIALS)
```

**New code:**
```python
@app.post("/generate_project")
async def generate_project(request: Request):
    # ... parse params (NO CHANGE) ...
    created_at = await insert_project(payload_project, pg_pool, CREDENTIALS)
```

**Why:** Add `await` (function is now async), replace `bigquery_client` with `pg_pool`.

---

### Change 2.9: `/load_projects` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~468-482)

**Current code:**
```python
GBQ_query = f"SELECT * FROM `{CREDENTIALS["GBQServer"]["table_name_projects"]}`"
projects = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
```

**New code:**
```python
rows = await pg_pool.fetch("SELECT * FROM projects")
projects = [dict(r) for r in rows]
```

**Why:** Simple SELECT. `asyncpg` returns Record objects — convert to dicts for JSON serialization.

---

### Change 2.10: `/load_project_plans` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~485-520)

**Current code:** Complex nested ARRAY subquery in BigQuery
```python
query = f"""
    SELECT p.*,
        ARRAY(SELECT AS STRUCT * FROM `{CREDENTIALS["GBQServer"]["table_name_plans"]}` pl
              WHERE pl.project_id = p.project_id) AS project_plans
    FROM `{CREDENTIALS["GBQServer"]["table_name_projects"]}` p
    WHERE LOWER(p.project_id) = LOWER(@project_id)
"""
```

**New code:**
```python
# PostgreSQL doesn't have ARRAY(SELECT AS STRUCT ...) — use two queries instead
project_row = await pg_pool.fetchrow(
    "SELECT * FROM projects WHERE LOWER(project_id) = LOWER($1)",
    project_id
)
if not project_row:
    return respond_with_UI_payload(dict(project_metadata=dict(), project_plans=list()))

project_metadata = dict(project_row)

plan_rows = await pg_pool.fetch(
    "SELECT * FROM plans WHERE LOWER(project_id) = LOWER($1)",
    project_id
)
project_plans = [dict(r) for r in plan_rows]
```

**Why:** BigQuery supports `ARRAY(SELECT AS STRUCT ...)` for nesting query results — PostgreSQL doesn't have this exact syntax. Two simple queries is cleaner and equally fast (~1ms each).

---

### Change 2.11: `/load_plan_pages` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~523-545)

**Current code:**
```python
GBQ_query = f"SELECT * FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
query_output = bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).to_dataframe()
dataframe = load_UI_dataframe(query_output)
plan_metadata = dict()
if dataframe.to_dict(orient="records"):
    plan_metadata = dataframe.to_dict(orient="records")[0]
# ... duplicate query for plan_pages_data
```

**New code:**
```python
rows = await pg_pool.fetch(
    "SELECT * FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
    project_id, plan_id
)
plan_pages_data = [dict(r) for r in rows]
plan_metadata = plan_pages_data[0] if plan_pages_data else dict()

# Convert datetime objects for JSON serialization
for page in plan_pages_data:
    if page.get("created_at"):
        page["created_at"] = page["created_at"].isoformat()
    if page.get("updated_at"):
        page["updated_at"] = page["updated_at"].isoformat()
```

**Why:**
- SQL injection fix (f-string → $1)
- Eliminates `.to_dataframe()` and `load_UI_dataframe()` — no more pandas dependency for this endpoint
- Removes the **duplicate query** (current code runs the same SELECT twice)
- `asyncpg` returns proper Python types; just need to handle datetime serialization

---

### Change 2.12: `/floorplan_to_2d` endpoint (the big one)

**File:** `drywall-takeoff-3d/main.py` (lines ~548-650)

This endpoint has **~8 separate BigQuery calls**. Here's each one:

#### 2.12a: Check if blob exists in GCS — NO CHANGE (not BQ)

#### 2.12b: `insert_plan` call
```python
# Current
insert_plan(project_id, user_id, "IN PROGRESS", bigquery_client, CREDENTIALS, ...)
# New
await insert_plan(project_id, user_id, "IN PROGRESS", pg_pool, CREDENTIALS, ...)
```

#### 2.12c: Polling loop — wait for model to appear
```python
# Current (inside while loop)
GBQ_query = f"SELECT scale, model_2d FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())

# New
row = await pg_pool.fetchrow(
    "SELECT scale, model_2d FROM models WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND page_number=$3",
    project_id, plan_id, page_number
)
# Replace: if query_output: → if row:
# Replace: query_output[0].model_2d → row["model_2d"]
# Replace: query_output[0].scale → row["scale"]
```

#### 2.12d: Delete empty models
```python
# Current
GBQ_query = f"DELETE FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()

# New
await pg_pool.execute(
    "DELETE FROM models WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND page_number=$3",
    project_id, plan_id, page_number
)
```

#### 2.12e: Update source path
```python
# Current
GBQ_query = f"UPDATE `{CREDENTIALS["GBQServer"]["table_name_models"]}` SET source = '{floorplan_page_source}' WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"

# New
await pg_pool.execute(
    "UPDATE models SET source = $1 WHERE LOWER(project_id)=LOWER($2) AND LOWER(plan_id)=LOWER($3) AND page_number=$4",
    floorplan_page_source, project_id, plan_id, page_number
)
```

#### 2.12f: Final insert_plan (status = COMPLETED/FAILED)
```python
# Same as 2.12b — add await, swap bigquery_client → pg_pool
```

---

### Change 2.13: `/load_2d_revision` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~652-670)

**Current code:**
```python
GBQ_query = f"SELECT model FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_2d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number} AND revision_number = {revision_number};"
query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
walls_2d_JSON = dict()
if query_output and query_output[0].model is not None:
    walls_2d_JSON = json.loads(query_output[0].model)
```

**New code:**
```python
row = await pg_pool.fetchrow(
    """SELECT model FROM model_revisions_2d
       WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)
       AND page_number=$3 AND revision_number=$4""",
    project_id, plan_id, int(page_number), int(revision_number)
)
walls_2d_JSON = dict()
if row and row["model"] is not None:
    walls_2d_JSON = json.loads(row["model"]) if isinstance(row["model"], str) else row["model"]
```

---

### Change 2.14: `/load_available_revision_numbers_2d` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~672-690)

**Current → New pattern (same as all SELECT endpoints):**
```python
# Current
GBQ_query = f"SELECT revision_number FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_2d"]}` WHERE ..."
query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())

# New
rows = await pg_pool.fetch(
    "SELECT revision_number FROM model_revisions_2d WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND page_number=$3",
    project_id, plan_id, int(page_number)
)
revision_numbers = [r["revision_number"] for r in rows if r["revision_number"] is not None]
```

---

### Change 2.15: `/load_2d_all` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~692-760)

**Current code:** Has a polling loop waiting for status=COMPLETED, then fetches all pages.

**New code changes:**

Status polling:
```python
# Current
GBQ_query = f"SELECT pages FROM `{CREDENTIALS["GBQServer"]["table_name_plans"]}` WHERE ..."
# New
row = await pg_pool.fetchrow(
    "SELECT pages FROM plans WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)",
    project_id, plan_id
)
n_pages = row["pages"] if row else 0
```

Fetch all pages:
```python
# Current — uses bigquery_client.query with parameterized job_config
query_job = bigquery_client.query(query, job_config=job_config)
for row in query_job.result():
    walls_2d = json.loads(row.model_2d) if isinstance(row.model_2d, str) else row.model_2d
    ...

# New
rows = await pg_pool.fetch(
    """SELECT page_number, scale, model_2d FROM models
       WHERE project_id=$1 AND plan_id=$2 ORDER BY page_number""",
    project_id, plan_id
)
for row in rows:
    if not row["model_2d"]:
        continue
    walls_2d = json.loads(row["model_2d"]) if isinstance(row["model_2d"], str) else row["model_2d"]
    page = {
        "plan_id": plan_id,
        "page_number": row["page_number"],
        "scale": row["scale"],
        "walls_2d": walls_2d.get("walls_2d", list()),
        "polygons": walls_2d.get("polygons", list()),
        **walls_2d.get("metadata", dict()),
    }
    walls_2d_all["pages"].append(page)
```

---

### Change 2.16: `/update_floorplan_to_2d` endpoint

**File:** `drywall-takeoff-3d/main.py` (lines ~763-800)

**Current:** Calls `insert_model_2d`, `insert_model_2d_revision`, `insert_model_3d`

**New:** Same calls, just add `await` and swap `bigquery_client` → `pg_pool`:
```python
# Current
insert_model_2d(dict(...), scale, index, plan_id, user_id, project_id, None, None, bigquery_client, CREDENTIALS)
insert_model_2d_revision(dict(...), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
# ... 3D computation stays same ...
insert_model_3d(dict(...), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)

# New
await insert_model_2d(dict(...), scale, index, plan_id, user_id, project_id, None, None, pg_pool, CREDENTIALS)
await insert_model_2d_revision(dict(...), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
# ... 3D computation stays EXACTLY same — no changes to Extrapolate3D ...
await insert_model_3d(dict(...), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
```

**Why:** The 3D computation pipeline (`Extrapolate3D`, `gltf`, `save_plot_3d`, `upload_floorplan`) has ZERO database interaction — it reads/writes local JSON files and uploads to GCS. Only the database calls at the beginning and end change.

---

### Changes 2.17-2.22: Remaining Endpoints (same pattern)

All follow the identical transformation pattern. Here's a summary:

| # | Endpoint | Current BQ Call | New PG Call |
|---|----------|----------------|-------------|
| 2.17 | `/floorplan_to_3d` | `insert_model_3d(... bigquery_client ...)` | `await insert_model_3d(... pg_pool ...)` |
| 2.18 | `/load_3d_revision` | `f"SELECT model FROM model_revisions_3d WHERE ..."` | `await pg_pool.fetchrow("SELECT model FROM model_revisions_3d WHERE ...$1..$2..", ...)` |
| 2.19 | `/load_available_revision_numbers_3d` | Same as 2.14 but for 3d table | Same pattern, 3d table |
| 2.20 | `/load_3d_all` | `bigquery_run(...).to_dataframe()` → iterate | `await pg_pool.fetch(...)` → iterate rows |
| 2.21 | `/update_floorplan_to_3d` | Calls `insert_model_3d` + `insert_model_3d_revision` | Add `await`, swap to `pg_pool` |
| 2.22 | `/generate_drywall_overlaid_floorplan_download_signed_URL` | Status polling + SELECT `target_drywalls` | Same two queries, parameterized |
| 2.23 | `/remove_floorplan` | Calls `delete_floorplan` | `await delete_floorplan(... pg_pool ...)` |
| 2.24 | `/compute_takeoff` | SELECT `model_3d` or `model` + calls `insert_takeoff` | `await pg_pool.fetchrow(...)` + `await insert_takeoff(... pg_pool ...)` |

---

## LAYER 3: `floorplan-to-structured-2d` Service

### Change 3.1: `floorplan-to-structured-2d/helper.py`

This file has its own copies of `load_bigquery_client`, `bigquery_run`, and `insert_model_2d`.

**Changes are identical to Layer 0 and Layer 1** — replace with `asyncpg` pool, parameterized queries.

The `insert_model_2d` in this file is slightly different (no `GCS_URL_floorplan_page` param, has `target_drywalls` instead):

```python
async def insert_model_2d(model_2d, scale, page_number, plan_id, user_id, project_id, target_drywalls, pg_pool, credentials):
    await pg_pool.execute(
        """INSERT INTO models (plan_id, project_id, user_id, page_number, scale,
               model_2d, model_3d, takeoff, target_drywalls, created_at, updated_at)
           VALUES ($1,$2,$3,$4,$5, $6::jsonb, '{}'::jsonb, '{}'::jsonb, $7, now(), now())
           ON CONFLICT (project_id, plan_id, page_number)
           DO UPDATE SET
               model_2d = $6::jsonb,
               scale = COALESCE(NULLIF($5, ''), models.scale),
               user_id = $3, updated_at = now()""",
        plan_id, project_id, user_id, page_number, scale,
        json.dumps(model_2d), target_drywalls
    )
```

### Change 3.2: `floorplan-to-structured-2d/main.py`

**File:** `floorplan-to-structured-2d/main.py` (line ~100)

Only one DB call at the end:
```python
# Current
bigquery_client = load_bigquery_client(CREDENTIALS)
insert_model_2d(..., bigquery_client, CREDENTIALS)

# New
# At startup (add same startup event as Change 0.5)
# Then:
await insert_model_2d(..., pg_pool, CREDENTIALS)
```

---

## LAYER 4: Database Schema (What to Create in Cloud SQL)

Based on matching the current BigQuery tables exactly:

```sql
-- Table: projects
CREATE TABLE projects (
    project_id        TEXT PRIMARY KEY,
    project_name      TEXT,
    project_location  TEXT,
    project_area      TEXT,
    project_type      TEXT,
    contractor_name   TEXT,
    fbm_branch        TEXT,
    created_by        TEXT,
    created_at        TIMESTAMPTZ DEFAULT now()
);

-- Table: plans
CREATE TABLE plans (
    plan_id           TEXT NOT NULL,
    project_id        TEXT NOT NULL,
    user_id           TEXT,
    status            TEXT DEFAULT 'NOT STARTED',
    plan_name         TEXT,
    plan_type         TEXT,
    file_type         TEXT,
    pages             INTEGER DEFAULT 0,
    size_in_bytes     BIGINT DEFAULT 0,
    source            TEXT,
    sha256            TEXT,
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (project_id, plan_id)
);

-- Table: models (replaces your models_2d test table)
CREATE TABLE models (
    project_id        TEXT NOT NULL,
    plan_id           TEXT NOT NULL,
    user_id           TEXT,
    page_number       INTEGER NOT NULL DEFAULT 0,
    scale             TEXT,
    source            TEXT,
    target_drywalls   TEXT,
    model_2d          JSONB,
    model_3d          JSONB,
    takeoff           JSONB,
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (project_id, plan_id, page_number)
);

-- Table: model_revisions_2d
CREATE TABLE model_revisions_2d (
    plan_id           TEXT NOT NULL,
    project_id        TEXT NOT NULL,
    user_id           TEXT,
    page_number       INTEGER NOT NULL,
    revision_number   INTEGER NOT NULL,
    scale             TEXT,
    model             JSONB,
    created_at        TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (project_id, plan_id, page_number, revision_number)
);

-- Table: model_revisions_3d
CREATE TABLE model_revisions_3d (
    plan_id           TEXT NOT NULL,
    project_id        TEXT NOT NULL,
    user_id           TEXT,
    page_number       INTEGER NOT NULL,
    revision_number   INTEGER NOT NULL,
    scale             TEXT,
    model             JSONB,
    takeoff           JSONB,
    created_at        TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (project_id, plan_id, page_number, revision_number)
);

-- Indexes
CREATE INDEX idx_plans_project ON plans(project_id);
CREATE INDEX idx_models_lookup ON models(project_id, plan_id);
CREATE INDEX idx_models_2d_jsonb ON models USING GIN (model_2d jsonb_path_ops);
CREATE INDEX idx_rev2d_lookup ON model_revisions_2d(project_id, plan_id, page_number);
CREATE INDEX idx_rev3d_lookup ON model_revisions_3d(project_id, plan_id, page_number);
```

**NOTE:** This replaces the `models_2d` table we created earlier in testing. The new `models` table has separate columns for `model_2d`, `model_3d`, and `takeoff` — matching the exact BigQuery schema so the code migration is minimal.

---

## LAYER 5: What Does NOT Change

| Component | Why No Change |
|-----------|--------------|
| `extrapolate_3d.py` | Reads/writes local JSON files only |
| `floor_plan.py` (both services) | Pure image processing, no DB |
| `gltf_generator.py` | Generates 3D mesh files locally |
| `preprocessing.py` | PDF to image conversion |
| `prompt.py` (both services) | LLM prompt templates |
| `modeller_2d.py` | LLM-powered 2D modeling — no DB calls |
| `transcriber.py` | Google Vision OCR — no DB |
| `wall_detector.py` + `main.py` | Uses GCS only, no BQ |
| All GCS operations | Unchanged — PDFs, PNGs, glTF files stay in GCS |
| All Vertex AI / LLM calls | Unchanged — model inference is independent |
| Frontend | Unchanged — API request/response shapes are identical |
| `load_UI_dataframe` function | Can be removed after migration (was needed for BQ dataframe quirks) |

---

## Summary: Total Changes

| Category | Count | Effort |
|----------|-------|--------|
| New config (gcp.yaml) | 2 files | 5 min |
| New dependency (requirements.txt) | 2 files | 2 min |
| Connection setup (startup/shutdown) | 2 services | 30 min |
| Foundation functions (bigquery_run, load_client) | 4 functions across 2 files | 1 hour |
| CRUD functions (insert_model_2d, delete_plan, etc.) | 8 functions | 3 hours |
| Endpoint queries (SELECT/UPDATE/DELETE) | ~25 inline queries | 4 hours |
| Database schema creation | 5 tables + indexes | 30 min |
| Testing | All endpoints | 2-3 days |
| **Total** | | **~1 week** |

Every single change follows the same 4-step pattern:
1. `def` → `async def`
2. `bigquery_client` → `pg_pool`
3. f-string SQL → `$1, $2, $3` parameterized
4. `.result()` / `.to_dataframe()` → `await pool.fetch()` / `fetchrow()` / `execute()`
