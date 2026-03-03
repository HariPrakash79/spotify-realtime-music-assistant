# Recommendation Quality Evaluation

- `k`: 20

| Source | Users Eval | Users with Recs | User Coverage | Precision@K | Recall@K | NDCG@K | HitRate@K | ItemCoverage@K | Personalization@K | ReadableRate@K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pop | 35 | 35 | 1.0000 | 0.0014 | 0.0286 | 0.0286 | 0.0286 | 0.0009 | 0.1019 | 1.0000 |
| mf | 35 | 35 | 1.0000 | 0.0014 | 0.0286 | 0.0123 | 0.0286 | 0.0115 | 0.9161 | 1.0000 |
| hybrid(w=0.45,0.45,0.10) | 35 | 35 | 1.0000 | 0.0014 | 0.0286 | 0.0286 | 0.0286 | 0.0067 | 0.7812 | 1.0000 |
| dense | 35 | 2 | 0.0571 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 | 0.1818 | 1.0000 |

