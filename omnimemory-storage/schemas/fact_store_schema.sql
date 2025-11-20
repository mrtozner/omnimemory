-- PostgreSQL Schema for Structural Fact Store
--
-- This schema stores structural facts extracted from code files
-- as part of the TriIndex architecture (Dense + Sparse + Structural).
--
-- Usage:
--   psql -U postgres -d omnimemory < fact_store_schema.sql

-- Create extension for UUID generation if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: facts
-- Stores individual structural facts extracted from files
-- (imports, classes, functions, etc.)
CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Fact content
    predicate VARCHAR(255) NOT NULL,  -- e.g., "imports", "defines_class", "defines_function"
    object TEXT NOT NULL,              -- e.g., "bcrypt", "AuthManager", "authenticate_user"

    -- Source information
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,   -- SHA256 of file content

    -- Fact metadata
    confidence FLOAT DEFAULT 1.0,     -- Confidence score (0.0-1.0)
    line_number INTEGER,              -- Line where fact appears
    context TEXT,                     -- Optional: surrounding code context

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Index for fast lookups
    CONSTRAINT facts_unique UNIQUE (file_path, predicate, object)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_facts_file_path ON facts(file_path);
CREATE INDEX IF NOT EXISTS idx_facts_file_hash ON facts(file_hash);
CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);
CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object);
CREATE INDEX IF NOT EXISTS idx_facts_predicate_object ON facts(predicate, object);
CREATE INDEX IF NOT EXISTS idx_facts_created_at ON facts(created_at);

-- Full-text search index for object names
CREATE INDEX IF NOT EXISTS idx_facts_object_fulltext ON facts USING gin(to_tsvector('english', object));

-- Table: file_facts
-- Junction table linking files to their facts for efficient file-based queries
CREATE TABLE IF NOT EXISTS file_facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique file-fact relationships
    CONSTRAINT file_facts_unique UNIQUE (file_path, fact_id)
);

-- Indexes for file_facts
CREATE INDEX IF NOT EXISTS idx_file_facts_file_path ON file_facts(file_path);
CREATE INDEX IF NOT EXISTS idx_file_facts_file_hash ON file_facts(file_hash);
CREATE INDEX IF NOT EXISTS idx_file_facts_fact_id ON file_facts(fact_id);

-- Table: fact_domains
-- Stores domain/category classifications for facts to enable domain-specific search
CREATE TABLE IF NOT EXISTS fact_domains (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    domain VARCHAR(100) NOT NULL,    -- e.g., "authentication", "database", "api"
    score FLOAT DEFAULT 1.0,         -- Domain relevance score (0.0-1.0)

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique fact-domain relationships
    CONSTRAINT fact_domains_unique UNIQUE (fact_id, domain)
);

-- Indexes for fact_domains
CREATE INDEX IF NOT EXISTS idx_fact_domains_fact_id ON fact_domains(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_domains_domain ON fact_domains(domain);
CREATE INDEX IF NOT EXISTS idx_fact_domains_score ON fact_domains(score);

-- Table: fact_access_log
-- Tracks fact access patterns for intelligent caching and relevance scoring
CREATE TABLE IF NOT EXISTS fact_access_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    tool_id VARCHAR(100) NOT NULL,   -- Tool that accessed the fact
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Context of access
    query_context TEXT,              -- Query that led to this fact being retrieved
    relevance_score FLOAT            -- How relevant was this fact to the query
);

-- Indexes for access log
CREATE INDEX IF NOT EXISTS idx_fact_access_log_fact_id ON fact_access_log(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_access_log_tool_id ON fact_access_log(tool_id);
CREATE INDEX IF NOT EXISTS idx_fact_access_log_accessed_at ON fact_access_log(accessed_at);

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger: Auto-update updated_at on facts table
CREATE TRIGGER update_facts_updated_at BEFORE UPDATE ON facts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- View: fact_statistics
-- Aggregated statistics for monitoring and optimization
CREATE OR REPLACE VIEW fact_statistics AS
SELECT
    predicate,
    COUNT(*) as fact_count,
    COUNT(DISTINCT file_path) as file_count,
    AVG(confidence) as avg_confidence,
    MAX(updated_at) as last_updated
FROM facts
GROUP BY predicate
ORDER BY fact_count DESC;

-- View: file_fact_summary
-- Summary of facts per file
CREATE OR REPLACE VIEW file_fact_summary AS
SELECT
    ff.file_path,
    ff.file_hash,
    COUNT(DISTINCT fa.id) as total_facts,
    COUNT(DISTINCT CASE WHEN fa.predicate = 'imports' THEN fa.id END) as import_count,
    COUNT(DISTINCT CASE WHEN fa.predicate = 'defines_class' THEN fa.id END) as class_count,
    COUNT(DISTINCT CASE WHEN fa.predicate = 'defines_function' THEN fa.id END) as function_count,
    MAX(fa.updated_at) as last_updated
FROM file_facts ff
JOIN facts fa ON ff.fact_id = fa.id
GROUP BY ff.file_path, ff.file_hash;

-- View: hot_facts
-- Most frequently accessed facts (for caching optimization)
CREATE OR REPLACE VIEW hot_facts AS
SELECT
    f.id,
    f.predicate,
    f.object,
    f.file_path,
    COUNT(fal.id) as access_count,
    AVG(fal.relevance_score) as avg_relevance,
    MAX(fal.accessed_at) as last_accessed
FROM facts f
JOIN fact_access_log fal ON f.id = fal.fact_id
WHERE fal.accessed_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY f.id, f.predicate, f.object, f.file_path
HAVING COUNT(fal.id) > 5
ORDER BY access_count DESC, avg_relevance DESC
LIMIT 1000;

-- Comments for documentation
COMMENT ON TABLE facts IS 'Stores structural facts extracted from code files (imports, classes, functions, etc.)';
COMMENT ON TABLE file_facts IS 'Junction table linking files to their structural facts';
COMMENT ON TABLE fact_domains IS 'Domain classifications for facts to enable domain-specific search';
COMMENT ON TABLE fact_access_log IS 'Tracks fact access patterns for intelligent caching';
COMMENT ON VIEW fact_statistics IS 'Aggregated statistics per fact predicate type';
COMMENT ON VIEW file_fact_summary IS 'Summary of facts per file for quick overview';
COMMENT ON VIEW hot_facts IS 'Most frequently accessed facts in the last 7 days';

-- Grant permissions (adjust as needed for your deployment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO omnimemory_user;
-- GRANT SELECT ON ALL VIEWS IN SCHEMA public TO omnimemory_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO omnimemory_user;
