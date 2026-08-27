CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    domain TEXT NOT NULL,
    file_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    parser_used TEXT,
    parser_version TEXT,
    status TEXT NOT NULL DEFAULT 'discovered',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS parent_sections (
    parent_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    section_title TEXT,
    section_path TEXT[],
    page_numbers INTEGER[],
    row_ranges JSONB,
    parent_text TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS child_chunks (
    chunk_id UUID PRIMARY KEY,
    parent_id UUID REFERENCES parent_sections(parent_id)
        ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    file_name TEXT NOT NULL,
    page_numbers INTEGER[],
    row_range JSONB,
    section_title TEXT,
    modalities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    text_content TEXT,
    table_markdown TEXT,
    table_html TEXT,
    image_paths TEXT[],
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hcahps_records (
    record_id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    source_row_number BIGINT NOT NULL,
    facility_id TEXT NOT NULL,
    facility_name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    county TEXT,
    telephone TEXT,
    measure_id TEXT NOT NULL,
    question TEXT,
    answer_description TEXT,
    star_rating TEXT,
    star_rating_footnote TEXT,
    answer_percent TEXT,
    answer_percent_footnote TEXT,
    linear_mean_value TEXT,
    completed_surveys TEXT,
    completed_surveys_footnote TEXT,
    response_rate_percent TEXT,
    response_rate_footnote TEXT,
    survey_start_date DATE,
    survey_end_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, source_row_number)
);

CREATE TABLE IF NOT EXISTS hospital_profiles (
    profile_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    facility_id TEXT NOT NULL,
    hospital_name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    county TEXT,
    telephone TEXT,
    survey_start_date DATE,
    survey_end_date DATE,
    completed_surveys TEXT,
    response_rate_percent TEXT,
    category_summaries JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, facility_id)
);

CREATE TABLE IF NOT EXISTS hospital_category_docs (
    category_doc_id UUID PRIMARY KEY,
    profile_id UUID NOT NULL REFERENCES hospital_profiles(profile_id)
        ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    facility_id TEXT NOT NULL,
    category TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    table_markdown TEXT NOT NULL,
    measure_ids TEXT[] NOT NULL,
    source_row_numbers BIGINT[] NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, facility_id, category)
);

CREATE INDEX IF NOT EXISTS idx_documents_domain
    ON documents(domain);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(status);

CREATE INDEX IF NOT EXISTS idx_parent_sections_document
    ON parent_sections(document_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_document
    ON child_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_parent
    ON child_chunks(parent_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_domain
    ON child_chunks(domain);

CREATE INDEX IF NOT EXISTS idx_hcahps_records_document
    ON hcahps_records(document_id);

CREATE INDEX IF NOT EXISTS idx_hcahps_records_facility
    ON hcahps_records(facility_id);

CREATE INDEX IF NOT EXISTS idx_hcahps_records_state
    ON hcahps_records(state);

CREATE INDEX IF NOT EXISTS idx_hcahps_records_measure
    ON hcahps_records(measure_id);

CREATE INDEX IF NOT EXISTS idx_hospital_profiles_document
    ON hospital_profiles(document_id);

CREATE INDEX IF NOT EXISTS idx_hospital_profiles_state
    ON hospital_profiles(state);

CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_document
    ON hospital_category_docs(document_id);

CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_facility
    ON hospital_category_docs(facility_id);

CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_category
    ON hospital_category_docs(category);

-- ===========================================================================
-- Embeddings (hybrid dense + sparse)
-- ---------------------------------------------------------------------------
-- Added as an idempotent migration so it is safe to re-run on the existing
-- database. halfvec/sparsevec require pgvector >= 0.7 (Neon ships >= 0.7).
--
-- dense_embedding  : halfvec(1024) -- 2 bytes/dim instead of 4 for float32,
--                    which ~halves dense vector + HNSW index storage with
--                    negligible recall loss. Embed in float; Postgres casts to
--                    fp16 on insert.
--   !!! 1024 assumes voyage-multimodal-3.5 outputs 1024 dims. CONFIRM the
--       model's dimension and keep it in sync with settings.voyage_embed_dim.
--       Drop to 512 (if the model supports output_dimension) to roughly halve
--       this again. The column is empty, so changing the dim now is free.
-- sparse_embedding : sparsevec over the SPLADE/BERT vocab (30522 terms).
--
-- Only retrieval units are embedded: child_chunks (pdf/html) and the two
-- hospital_* rollups (csv). The raw hcahps_records fact table is left
-- UNEMBEDDED -- it exists for structured filtering, not vector search -- which
-- is the single biggest storage saving on the 0.5 GB plan.
-- ===========================================================================

ALTER TABLE child_chunks
    ADD COLUMN IF NOT EXISTS dense_embedding  halfvec(1024),
    ADD COLUMN IF NOT EXISTS sparse_embedding sparsevec(30522);

ALTER TABLE hospital_profiles
    ADD COLUMN IF NOT EXISTS dense_embedding  halfvec(1024),
    ADD COLUMN IF NOT EXISTS sparse_embedding sparsevec(30522);

ALTER TABLE hospital_category_docs
    ADD COLUMN IF NOT EXISTS dense_embedding  halfvec(1024),
    ADD COLUMN IF NOT EXISTS sparse_embedding sparsevec(30522);

-- HNSW indexes: cosine for dense, inner-product for SPLADE. m/ef_construction
-- kept modest to limit index size on the free tier. For a large bulk load it
-- is faster to populate the columns first and create these indexes afterward.
-- (HNSW on sparsevec requires <= 1000 non-zero terms per row; SPLADE fits.)
CREATE INDEX IF NOT EXISTS idx_child_chunks_dense
    ON child_chunks USING hnsw (dense_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_child_chunks_sparse
    ON child_chunks USING hnsw (sparse_embedding sparsevec_ip_ops);

CREATE INDEX IF NOT EXISTS idx_hospital_profiles_dense
    ON hospital_profiles USING hnsw (dense_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_hospital_profiles_sparse
    ON hospital_profiles USING hnsw (sparse_embedding sparsevec_ip_ops);

CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_dense
    ON hospital_category_docs USING hnsw (dense_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_sparse
    ON hospital_category_docs USING hnsw (sparse_embedding sparsevec_ip_ops);

-- ===========================================================================
-- Domain routing
-- ---------------------------------------------------------------------------
-- One centroid (mean dense vector) per domain, over every embedded retrieval
-- unit in that domain. At query time the query vector is compared to each
-- centroid (cosine) and search is routed to the nearest domain. Refresh with
-- scripts/compute_domain_centroids.py after re-embedding.
-- ===========================================================================
CREATE TABLE IF NOT EXISTS domain_centroids (
    domain TEXT PRIMARY KEY,
    centroid halfvec(1024) NOT NULL,
    source_count INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);