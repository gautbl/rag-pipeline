{{ config(materialized='table') }}

SELECT
    id,
    TRIM(content)          AS content,
    LOWER(source_file)     AS source_file,
    LENGTH(TRIM(content))  AS char_count,
    NOW()                  AS processed_at
FROM {{ source('raw', 'chunks_raw') }}
WHERE
    content IS NOT NULL
    AND LENGTH(TRIM(content)) > 10