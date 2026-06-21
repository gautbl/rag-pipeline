-- Modèle dbt : chunks nettoyés prêts pour la vectorisation
-- Ce modèle s'appuie sur chunks_clean (déjà existant)
SELECT
    id,
    content,
    source_file,
    -- La colonne embedding sera remplie par scripts/embed.py
    -- dbt gère la structure, Python gère les vecteurs
    NULL::FLOAT AS embedding
FROM {{ ref('chunks_clean') }}