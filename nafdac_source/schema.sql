-- NAFDAC Greenbook schema for PostgreSQL.
--
-- Two tables, because a verification answer has to distinguish "this product
-- is not registered" from "this product used to be registered and has since
-- been superseded or withdrawn". Both states exist in the real world and
-- conflating them would tell a user a product is fake when it is merely old.
--
--   nafdac_products         the current greenbook, one row per Product ID
--   nafdac_product_history  superseded, withdrawn, suspended or recalled rows
--
-- `nrn_key` is the lookup key, and it is NOT unique: 257 of the 8935 scraped
-- rows share an NRN with another row. The uniqueness constraint is on
-- product_id, which is the greenbook's own primary key.

CREATE TABLE IF NOT EXISTS nafdac_products (
    product_id               BIGINT PRIMARY KEY,
    nrn                      TEXT,
    nrn_key                  TEXT NOT NULL,
    product_name             TEXT,
    active_ingredients       TEXT,
    strengths                TEXT,
    form                     TEXT,
    route                    TEXT,
    product_category         TEXT,
    status                   TEXT NOT NULL DEFAULT 'Unknown',
    marketing_category       TEXT,
    applicant_name           TEXT,
    applicant_address        TEXT,
    manufacturer_name        TEXT,
    manufacturer_country     TEXT,
    approval_date            DATE,
    expiry_date              DATE,
    atc_code                 TEXT,
    product_category_id      INTEGER,
    ingredient_id            INTEGER,
    ingredient_synonym       TEXT,
    form_id                  INTEGER,
    route_id                 INTEGER,
    product_description      TEXT,
    composition              TEXT,
    pack_size                TEXT,
    applicant_id             INTEGER,
    manufacturer_id          INTEGER,
    country_id               INTEGER,
    marketing_category_id    INTEGER,
    smpc_url                 TEXT,
    biosimilar               TEXT,
    created_at               TIMESTAMPTZ,
    updated_at               TIMESTAMPTZ,
    source_url               TEXT,
    loaded_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The primary access path: an NRN lookup. nrn_key is normalised (uppercase,
-- single dash) so 'A4 - 5180' and 'A4-5180' hit the same row.
CREATE INDEX IF NOT EXISTS nafdac_products_nrn_key_idx
    ON nafdac_products (nrn_key);

-- The product_id lookup path, for a caller that has the greenbook URL.
CREATE INDEX IF NOT EXISTS nafdac_products_source_url_idx
    ON nafdac_products (source_url);

-- Free-text search from the server ("anything for malaria?"). Trigram needs
-- pg_trgm; without it this index is simply skipped, so the loader creates it
-- separately and tolerates the failure.
CREATE INDEX IF NOT EXISTS nafdac_products_name_lower_idx
    ON nafdac_products (lower(product_name));

CREATE INDEX IF NOT EXISTS nafdac_products_ingredients_lower_idx
    ON nafdac_products (lower(active_ingredients));

CREATE INDEX IF NOT EXISTS nafdac_products_applicant_lower_idx
    ON nafdac_products (lower(applicant_name));


CREATE TABLE IF NOT EXISTS nafdac_product_history (
    history_id               BIGSERIAL PRIMARY KEY,
    nrn                      TEXT,
    nrn_key                  TEXT NOT NULL,
    product_id               BIGINT,
    product_name             TEXT,
    active_ingredients       TEXT,
    strengths                TEXT,
    form                     TEXT,
    route                    TEXT,
    product_category         TEXT,
    status                   TEXT NOT NULL DEFAULT 'Superseded',
    marketing_category       TEXT,
    applicant_name           TEXT,
    applicant_address        TEXT,
    manufacturer_name        TEXT,
    manufacturer_country     TEXT,
    approval_date            DATE,
    expiry_date              DATE,
    atc_code                 TEXT,
    product_category_id      INTEGER,
    ingredient_id            INTEGER,
    ingredient_synonym       TEXT,
    form_id                  INTEGER,
    route_id                 INTEGER,
    product_description      TEXT,
    composition              TEXT,
    pack_size                TEXT,
    applicant_id             INTEGER,
    manufacturer_id          INTEGER,
    country_id               INTEGER,
    marketing_category_id    INTEGER,
    smpc_url                 TEXT,
    biosimilar               TEXT,
    created_at               TIMESTAMPTZ,
    updated_at               TIMESTAMPTZ,
    source_url               TEXT,

    -- Why this row is not in the current table any more.
    superseded_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_reason        TEXT,
    superseded_by_product_id BIGINT REFERENCES nafdac_products (product_id),
    recorded_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS nafdac_product_history_nrn_key_idx
    ON nafdac_product_history (nrn_key);

-- One history row per (product, supersession event), so re-running the loader
-- for the same event updates it instead of piling up duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS nafdac_product_history_identity_idx
    ON nafdac_product_history (COALESCE(product_id, -1), superseded_at);


-- A record of what was loaded and when, so an operator can tell whether a
-- stale answer came from a stale database.
CREATE TABLE IF NOT EXISTS nafdac_load_runs (
    run_id          BIGSERIAL PRIMARY KEY,
    source_file     TEXT,
    source_size     BIGINT,
    source_mtime    TIMESTAMPTZ,
    rows_loaded     INTEGER NOT NULL DEFAULT 0,
    rows_superseded INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'complete',
    detail          TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS nafdac_load_runs_started_at_idx
    ON nafdac_load_runs (started_at DESC);
