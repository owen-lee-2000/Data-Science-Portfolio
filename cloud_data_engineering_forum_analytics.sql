-- ============================================================================
-- FORUM DATA PIPELINE & ANALYTICS STACK
-- Snowflake Enterprise Environment
-- Database architecture, S3 ingestion, testing table clones, and KPI tracking
-- ============================================================================

-- Establish the admin role for initial resource configuration
USE ROLE sysadmin;

CREATE DATABASE IF NOT EXISTS forum;

-- Separate data loading and analytical queries to prevent resource contention
CREATE OR REPLACE WAREHOUSE forum_loading_wh WITH
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_RESUME = TRUE
    AUTO_SUSPEND = 300; 

CREATE OR REPLACE WAREHOUSE forum_query_wh WITH
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_RESUME = TRUE 
    AUTO_SUSPEND = 300;

-- Elevate to accountadmin to apply account-level cost controls
USE ROLE accountadmin;

-- Set up a resource monitor to prevent budget overruns on compute usage
CREATE OR REPLACE RESOURCE MONITOR forum_rm
    WITH credit_quota = 100
         frequency = monthly
         start_timestamp = immediately
         triggers
            ON 80 PERCENT DO NOTIFY
            ON 95 PERCENT DO SUSPEND
            ON 100 PERCENT DO SUSPEND_IMMEDIATE;

-- Apply the resource monitor to the query warehouse to manage active user execution costs
ALTER WAREHOUSE forum_query_wh SET RESOURCE_MONITOR = forum_rm;

-- Implement statement timeouts to automatically terminate queries exceeding 20 minutes
ALTER WAREHOUSE forum_query_wh SET statement_timeout_in_seconds = 1200;
ALTER WAREHOUSE forum_query_wh SET statement_queued_timeout_in_seconds = 600;

-- Verify warehouse deployment statuses and monitor attachments
SHOW WAREHOUSES;

-- Return to sysadmin context to build out the schema and load incoming data
USE ROLE sysadmin;
USE forum.public;
USE WAREHOUSE forum_loading_wh;

-- ============================================================================
-- BASELINE RELATION DEFINITIONS (DDL)
-- ============================================================================

CREATE OR REPLACE TABLE badges (
    badge_id integer,
    user_id integer,
    name string,
    date datetime,
    class string
);

CREATE OR REPLACE TABLE post_history (
    post_history_id integer,
    post_history_type_id integer,
    post_id integer,
    creation_date datetime,
    user_id integer,
    text string,
    comment string
);

CREATE OR REPLACE TABLE post_links (
    post_link_id integer,
    creation_date datetime,
    post_id integer,
    related_post_id integer,
    link_type_id integer
);

CREATE OR REPLACE TABLE posts (
    post_id integer,
    post_type_id integer,
    creation_date datetime,
    score integer,
    view_count integer,
    title string,
    body string,
    owner_user_id integer,
    last_editor_user_id integer,
    last_edit_date datetime,
    last_activity_date datetime,
    tags string,
    answer_count integer,
    comment_count integer,
    accepted_answer_id integer,
    parent_id integer,
    closed_date datetime
);

CREATE OR REPLACE TABLE tags (
    tag_id integer,
    tag_name string,
    count integer,
    excerpt_post_id integer,
    wiki_post_id integer
);

CREATE OR REPLACE TABLE users (
    user_id integer,
    display_name string,
    about_me string,
    reputation integer,
    creation_date datetime,
    last_access_date datetime,
    views integer,
    up_votes integer,
    down_votes integer,
    account_id integer
);

CREATE OR REPLACE TABLE votes (
    vote_id integer,
    post_id integer,
    vote_type_id integer,
    creation_date date
);

-- Landing table for raw, unstructured JSON data chunks
CREATE OR REPLACE TABLE comments_json (v variant);

-- ============================================================================
-- INGESTION PIPELINE CONFIGURATION (ETL)
-- ============================================================================

-- Define an external stage mapping directly to the source AWS S3 bucket
CREATE STAGE retrocomputing_forum
    url = 's3://retrocomputing-forum/';

-- Inspect available staging objects within the cloud bucket
LIST @retrocomputing_forum;

-- Configure explicit file format parsers to match source data delimiters
CREATE OR REPLACE FILE FORMAT forum_csv
    type = 'csv'
    field_delimiter = ','
    skip_header = 1
    trim_space = false 
    null_if = ('') 
    field_optionally_enclosed_by = '\042'
    comment = 'Standard comma-delimited record parser';
    
CREATE OR REPLACE FILE FORMAT forum_pipe
    type = 'csv'
    field_delimiter = '|'
    skip_header = 1
    trim_space = false 
    null_if = ('') 
    field_optionally_enclosed_by = '\042'
    comment = 'Standard pipe-delimited record parser';

CREATE OR REPLACE FILE FORMAT forum_tab
    type = 'csv'
    field_delimiter = '\t'
    skip_header = 1
    trim_space = false 
    null_if = ('') 
    field_optionally_enclosed_by = '\042'
    comment = 'Standard tab-delimited record parser';

-- Execute bulk data loading from staging targets into structured tables
COPY INTO posts FROM @retrocomputing_forum/posts file_format = forum_csv;
COPY INTO post_history FROM @retrocomputing_forum/post_history file_format = forum_csv;
COPY INTO post_links FROM @retrocomputing_forum/post_links file_format = forum_csv;
COPY INTO tags FROM @retrocomputing_forum/tags file_format = forum_tab;
COPY INTO users FROM @retrocomputing_forum/users file_format = forum_csv;
COPY INTO votes FROM @retrocomputing_forum/votes file_format = forum_pipe;

-- Transform values on-the-fly during ingestion, mapping status codes to clean text labels
COPY INTO badges (badge_id, user_id, name, date, class)
    FROM (
        SELECT 
            $1,
            $2,
            $3,
            $4,
            CASE
                WHEN $5 = 1 THEN 'Gold'
                WHEN $5 = 2 THEN 'Silver'
                WHEN $5 = 3 THEN 'Bronze'
                ELSE 'Unknown'
            END AS class
        FROM @retrocomputing_forum/badges.csv
    )
file_format = forum_csv;

-- Verify transformation output accuracy
SELECT * FROM badges WHERE class = 'Gold';

-- Direct load of semi-structured objects into raw VARIANT storage units
COPY INTO comments_json (v)
    FROM @retrocomputing_forum/comments.json
        file_format = (type = json strip_outer_array = true);

-- Deploy a strongly typed semantic view to expose structured fields from the raw JSON payload
CREATE OR REPLACE VIEW structured_comments AS
    SELECT
        v:comment_id::integer AS comment_id,
        v:creation_date::timestamp AS creation_date,
        v:post_id::integer AS post_id,
        v:score::integer AS score,
        v:text::string AS text,
        v:user_id::integer AS user_id
    FROM comments_json;

-- ============================================================================
-- SCHEMA MANIPULATION VIA ZERO-COPY CLONING
-- ============================================================================

-- Create a zero-copy metadata clone to isolate schema modifications from production data
CREATE TABLE users_dev CLONE users;

-- Append the calculation column to the isolated target
ALTER TABLE users_dev ADD COLUMN years_of_activity integer;

-- Calculate user tenure metrics using timestamp deltas
UPDATE users_dev
    SET years_of_activity = TIMESTAMPDIFF(YEAR, creation_date, last_access_date);

-- Atomic switch: promote the updated dev table to production and push stale data to a backup layer
ALTER TABLE users RENAME TO users_backup;
ALTER TABLE users_dev RENAME TO users;
DROP TABLE users_backup;

-- Validate production table access and field values post-migration
SELECT * FROM users WHERE years_of_activity >= 7;

-- ============================================================================
-- ROLE-BASED ACCESS CONTROL DESIGN (RBAC)
-- ============================================================================

-- Switch to useradmin context to establish a functional role for analysts
USE ROLE useradmin;
CREATE ROLE forum_query_role;

-- Bind the newly initialized role to the active administrative user identity
SET my_user = CURRENT_USER();
GRANT ROLE forum_query_role TO USER identifier($my_user);

-- Transition to securityadmin to authorize structural permission sets
USE ROLE securityadmin;

-- Permit query execution on the dedicated query warehouse resource
GRANT OPERATE, USAGE ON WAREHOUSE forum_query_wh TO ROLE forum_query_role;

-- Provision read-only query selection across target databases, schemas, tables, and views
GRANT USAGE ON DATABASE forum TO ROLE forum_query_role;
GRANT USAGE ON ALL SCHEMAS IN DATABASE forum TO ROLE forum_query_role;
GRANT SELECT ON ALL TABLES IN DATABASE forum TO ROLE forum_query_role;
GRANT SELECT ON ALL VIEWS IN DATABASE forum TO ROLE forum_query_role;

-- Review explicit account role configurations
SHOW ROLES;

-- ============================================================================
-- BUSINESS INTELLIGENCE & PRODUCT METRICS
-- ============================================================================

-- Apply the restricted query role and cluster context for final execution tests
USE ROLE forum_query_role;
USE WAREHOUSE forum_query_wh;

-- Metric 1: Total chronological historical timeline range covered by active logs
SELECT
    MIN(creation_date) AS first_post_date,
    MAX(creation_date) AS recent_post_date
FROM
    posts;

-- Metric 2: Complete distinct platform active user profile volume counts
SELECT
    COUNT(*) AS num_users
FROM
    users;

-- Metric 3: Top 5 platform profiles ranked by overall accumulated reputation scores
SELECT
    display_name,
    reputation
FROM
    users
GROUP BY
    display_name,
    reputation
ORDER BY
    reputation DESC
LIMIT 5;

-- Metric 4: Platform user onboarding velocity aggregated year-over-year
SELECT
    EXTRACT(YEAR FROM creation_date) AS year,
    COUNT(user_id) AS user_count
FROM 
    users
GROUP BY 
    EXTRACT(YEAR FROM creation_date)
ORDER BY 
    year;

-- Metric 5: Consumer retention index (percentage of users active on or after Jan 1, 2023)
SELECT
    COUNT(CASE WHEN last_access_date >= '2023-01-01' THEN user_id END) * 100.0 / COUNT(user_id) AS recent_access_percentage
FROM 
    users;

-- Metric 6: High-frequency category isolation within global gold badge designations
SELECT 
    name AS badge_name,
    COUNT(*) AS badge_count,
FROM badges
    WHERE class = 'Gold'
GROUP BY
    badge_name
ORDER BY
    badge_count DESC;

-- Metric 7: User engagement distribution analysis capturing the top 10 historical badge earners
SELECT
    user_id,
    COUNT(*) AS badge_count
FROM
    badges
GROUP BY
    user_id
ORDER BY
    badge_count DESC
LIMIT 10;

-- Metric 8: Total global post creation frequency categorized by calendar year
SELECT
    EXTRACT(YEAR FROM creation_date) AS year,
    COUNT(post_id) AS post_count
FROM
    posts
GROUP BY
    EXTRACT(YEAR FROM creation_date)
ORDER BY
    year;

-- Metric 9: Thread content quality ratio (percentage of questions with an accepted solution link)
SELECT
    (COUNT(CASE WHEN accepted_answer_id IS NOT NULL THEN 1 END) * 100.0) / 
     COUNT(CASE WHEN post_type_id = 1 THEN 1 END) AS accepted_answer_pct
FROM
    posts
WHERE
    post_type_id = 1;

-- Metric 10: Platform content indexing gaps (percentage of source entries receiving zero responses)
SELECT
    (COUNT(CASE WHEN answer_count = 0 THEN 1 END) * 100.0) / 
     COUNT(CASE WHEN post_type_id = 1 THEN 1 END) AS no_answer_pct
FROM
    posts
WHERE
    post_type_id = 1;

-- Metric 11: Relational join evaluating update edit volume scales across primary content tables
SELECT
    p.post_id,
    p.title,
    COUNT(ph.post_history_id) AS update_count
FROM
    posts p
JOIN
    post_history ph
    ON p.post_id = ph.post_id
WHERE
    p.title IS NOT NULL
GROUP BY
    p.post_id, p.title
ORDER BY
    update_count DESC
LIMIT 50;

-- Metric 12: Behavioral data logging capturing user profiles with maximum comment submission volume
SELECT
    u.display_name,
    COUNT(ph.post_history_id) AS comment_count
FROM
    post_history ph
JOIN
    users u ON ph.user_id = u.user_id
WHERE
    ph.comment IS NOT NULL
GROUP BY
    u.display_name
ORDER BY
    comment_count DESC
LIMIT 10;

-- Metric 13: System modification vectors capturing historical spam and offensive moderation flags
SELECT
    COUNT(DISTINCT post_id) AS distinct_posts
FROM
    votes
WHERE
    vote_type_id = 12  -- Code mapping for verified system spam entries.
    OR
    vote_type_id = 4;  -- Code mapping for verified system offensive entries.
