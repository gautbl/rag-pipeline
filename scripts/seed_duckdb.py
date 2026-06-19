# scripts/seed_duckdb.py — génère des chunks fictifs dans DuckDB
import duckdb

con = duckdb.connect("dev.duckdb")
con.execute("DROP TABLE IF EXISTS chunks_raw")
con.execute("""
    CREATE TABLE IF NOT EXISTS chunks_raw (
        id      INTEGER,
        content VARCHAR,
        source_file VARCHAR
    )
""")
con.execute("""
    INSERT INTO chunks_raw VALUES
        (1, 'Le pipeline RAG permet une recherche sémantique efficace.', 'doc1.pdf'),
        (2, '  ', 'doc1.pdf'),
        (3, 'LangChain facilite l intégration des LLMs dans les applications.', 'doc2.pdf'),
        (4, 'Chroma est une base vectorielle légère et locale.', 'doc2.pdf')
""")
print("Tables présentes :", con.execute("SHOW TABLES").fetchall())
print("Données insérées :", con.execute("SELECT * FROM chunks_raw").fetchall())
con.close()