# Recommendation Quality Evaluation

- `k`: 20

| Source | Users Eval | Users with Recs | User Coverage | Precision@K | Recall@K | NDCG@K | HitRate@K | ItemCoverage@K | Personalization@K | ReadableRate@K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pop | 18 | 18 | 1.0000 | 0.0028 | 0.0556 | 0.0556 | 0.0556 | 0.0028 | 0.1115 | 1.0000 |
| mf | 18 | 18 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0239 | 0.9431 | 1.0000 |
| hybrid(w=0.55,0.35,0.10) | 18 | 18 | 1.0000 | 0.0028 | 0.0556 | 0.0150 | 0.0556 | 0.0185 | 0.8669 | 1.0000 |
| dense | 18 | 1 | 0.0556 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0020 | 0.0000 | 1.0000 |

