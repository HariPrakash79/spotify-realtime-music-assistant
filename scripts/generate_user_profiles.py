#!/usr/bin/env python3
"""
Generate human-friendly display names for model users.

Purpose:
- Keep stable technical user_id for storage/join logic.
- Expose readable names (e.g., "Harry Wilson") in API/chat/demo.

Usage:
  python scripts/generate_user_profiles.py
  python scripts/generate_user_profiles.py --max-users 1000
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from psycopg import connect
from psycopg.rows import dict_row


FIRST_NAMES: Sequence[str] = (
    "Aarav", "Abigail", "Adrian", "Aisha", "Alex", "Alice", "Amelia", "Anderson", "Aria", "Ariana",
    "Asher", "Ava", "Benjamin", "Bianca", "Brandon", "Caleb", "Camila", "Carter", "Charlotte", "Chloe",
    "Christopher", "Clara", "Daniel", "David", "Dylan", "Eleanor", "Elena", "Elijah", "Ella", "Ethan",
    "Eva", "Felix", "Gabriel", "Grace", "Hannah", "Harper", "Harry", "Hazel", "Henry", "Hudson",
    "Isaac", "Isla", "Ivy", "Jack", "Jacob", "James", "Jasmine", "Jason", "Jayden", "John",
    "Jonathan", "Joseph", "Joshua", "Julia", "Kai", "Landon", "Leah", "Leo", "Liam", "Lily",
    "Logan", "Lucas", "Lucy", "Luna", "Mason", "Maya", "Mia", "Michael", "Mila", "Nathan",
    "Nora", "Noah", "Olivia", "Owen", "Penelope", "Riley", "Ryan", "Samuel", "Scarlett", "Sebastian",
    "Sofia", "Sophie", "Stella", "Theodore", "Thomas", "Victoria", "Violet", "William", "Wyatt", "Zoe",
)

LAST_NAMES: Sequence[str] = (
    "Adams", "Allen", "Anderson", "Baker", "Barnes", "Bell", "Bennett", "Brooks", "Brown", "Butler",
    "Campbell", "Carter", "Clark", "Coleman", "Collins", "Cooper", "Cox", "Davis", "Diaz", "Edwards",
    "Evans", "Foster", "Garcia", "Gomez", "Gonzalez", "Gray", "Green", "Griffin", "Hall", "Harris",
    "Hayes", "Henderson", "Hill", "Howard", "Hughes", "Jackson", "James", "Jenkins", "Johnson", "Jones",
    "Kelly", "King", "Lee", "Lewis", "Long", "Lopez", "Martin", "Martinez", "Miller", "Mitchell",
    "Moore", "Morgan", "Morris", "Murphy", "Nelson", "Nguyen", "Parker", "Patel", "Perez", "Perry",
    "Peterson", "Phillips", "Powell", "Price", "Ramirez", "Reed", "Richardson", "Rivera", "Roberts", "Robinson",
    "Rodriguez", "Rogers", "Ross", "Russell", "Sanchez", "Scott", "Simmons", "Smith", "Stewart", "Taylor",
    "Thomas", "Thompson", "Torres", "Turner", "Walker", "Ward", "Watson", "White", "Williams", "Wilson",
)

FETCH_USERS_SQL = """
SELECT user_id
FROM music.v_model_users_1000_ready
ORDER BY plays DESC, user_id
LIMIT %s
"""

UPSERT_PROFILE_SQL = """
INSERT INTO music.user_profiles (user_id, display_name, updated_at)
VALUES (%s, %s, NOW())
ON CONFLICT (user_id) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    updated_at = NOW()
"""

FETCH_PROFILES_SQL = """
SELECT
    mu.user_id,
    mu.plays,
    COALESCE(up.display_name, mu.user_id) AS display_name
FROM music.v_model_users_1000_ready mu
LEFT JOIN music.user_profiles up
  ON up.user_id = mu.user_id
ORDER BY mu.plays DESC, mu.user_id
LIMIT %s
"""


def build_conn_kwargs() -> Mapping[str, str]:
    dsn = os.environ.get("POSTGRES_DSN")
    if dsn:
        return {"conninfo": dsn}

    host = os.environ.get("PGHOST")
    dbname = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    if not all([host, dbname, user, password]):
        raise ValueError(
            "Postgres connection is required. Set POSTGRES_DSN or PGHOST/PGDATABASE/PGUSER/PGPASSWORD."
        )
    return {
        "host": host,
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": dbname,
        "user": user,
        "password": password,
        "sslmode": os.environ.get("PGSSLMODE", "require"),
    }


def connect_postgres():
    kwargs = build_conn_kwargs()
    if "conninfo" in kwargs:
        return connect(kwargs["conninfo"], row_factory=dict_row)
    return connect(row_factory=dict_row, **kwargs)


def candidate_name(seed: int) -> str:
    first = FIRST_NAMES[seed % len(FIRST_NAMES)]
    last = LAST_NAMES[(seed // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"


def assign_display_names(user_ids: List[str]) -> Dict[str, str]:
    used = set()
    out: Dict[str, str] = {}

    capacity = len(FIRST_NAMES) * len(LAST_NAMES)
    if len(user_ids) > capacity:
        raise ValueError(
            f"Not enough unique names in dictionary ({capacity}) for {len(user_ids)} users. Expand name lists."
        )

    for user_id in user_ids:
        seed = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12], 16)
        idx = seed % capacity
        for _ in range(capacity):
            name = candidate_name(idx)
            if name not in used:
                used.add(name)
                out[user_id] = name
                break
            idx = (idx + 1) % capacity
        else:
            raise RuntimeError("Failed to assign unique display name.")
    return out


def fetch_users(limit: int) -> List[str]:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(FETCH_USERS_SQL, (limit,))
            rows = cur.fetchall()
            return [str(r["user_id"]) for r in rows]
    finally:
        conn.close()


def write_profiles(pairs: List[Tuple[str, str]]) -> None:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_PROFILE_SQL, pairs)
        conn.commit()
    finally:
        conn.close()


def fetch_profiles(limit: int) -> List[Dict[str, object]]:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(FETCH_PROFILES_SQL, (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def write_profiles_txt(rows: List[Dict[str, object]], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["display_name\tuser_id\tplays"]
    for r in rows:
        lines.append(f"{r['display_name']}\t{r['user_id']}\t{r['plays']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate readable display names for model users.")
    parser.add_argument("--max-users", type=int, default=1000, help="Max users to name from ready model slice.")
    parser.add_argument(
        "--output-path",
        default="artifacts/user_profiles/user_names.txt",
        help="Text file output listing display_name -> user_id mapping.",
    )
    args = parser.parse_args()

    users = fetch_users(args.max_users)
    if not users:
        print("No users found in music.v_model_users_1000_ready.")
        return

    mapping = assign_display_names(users)
    pairs = list(mapping.items())
    write_profiles(pairs)
    exported = fetch_profiles(args.max_users)
    write_profiles_txt(exported, args.output_path)

    print(f"user_profiles upserted={len(pairs)}")
    print(f"user_profiles_file={args.output_path}")
    for user_id, display_name in pairs[:10]:
        print(f"{user_id}\t{display_name}")


if __name__ == "__main__":
    main()
