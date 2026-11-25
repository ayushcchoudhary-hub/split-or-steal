-- =============================================================================
-- Split or Steal — Replayable Simulation Storage Schema (PostgreSQL / Supabase)
-- Mirrors the 5 relational tables in sim/storage.py (SQLite):
--   1. experiments
--   2. games
--   3. rounds
--   4. turns
--   5. decisions
--
-- Postgres / Supabase best practices applied:
--   - text instead of varchar(n), timestamptz instead of timestamp
--   - numeric(12,2) for exact currency arithmetic (no floating-point drift)
--   - jsonb for structured configs & extensible metadata
--   - Explicit CHECK constraints for enum domains and invariants
--   - Explicit B-tree indexes on EVERY foreign-key column & analytics filter
--   - Row Level Security (RLS) enabled with ready-to-use public replay policies
-- =============================================================================

create table if not exists public.experiments (
    experiment_id text primary key,
    name text not null,
    regime text not null,
    description text not null default '',
    config jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.games (
    game_id text primary key,
    experiment_id text not null references public.experiments(experiment_id) on delete cascade,
    regime text not null,
    matchup_key text not null,
    game_index integer not null check (game_index >= 0),
    seed bigint,
    stakes jsonb not null,
    negotiation_turns_per_round integer not null check (negotiation_turns_per_round >= 0),
    swap_first_mover text not null,
    initial_first_mover_seat text not null check (initial_first_mover_seat in ('A', 'B')),
    player_a_id text not null,
    player_a_name text not null,
    player_a_model text not null,
    player_a_config jsonb not null,
    player_b_id text not null,
    player_b_name text not null,
    player_b_model text not null,
    player_b_config jsonb not null,
    total_winnings_a numeric(12, 2) not null,
    total_winnings_b numeric(12, 2) not null,
    winner_seat text not null check (winner_seat in ('A', 'B', 'TIE')),
    started_at timestamptz not null,
    completed_at timestamptz not null,
    metadata jsonb not null default '{}'::jsonb
);

create table if not exists public.rounds (
    round_id text primary key,
    game_id text not null references public.games(game_id) on delete cascade,
    round_number integer not null check (round_number >= 1),
    stake numeric(12, 2) not null check (stake > 0),
    first_mover_seat text not null check (first_mover_seat in ('A', 'B')),
    outcome_label text not null,
    unique (game_id, round_number)
);

create table if not exists public.turns (
    turn_id text primary key,
    round_id text not null references public.rounds(round_id) on delete cascade,
    game_id text not null references public.games(game_id) on delete cascade,
    round_number integer not null check (round_number >= 1),
    turn_index integer not null check (turn_index >= 0),
    player_turn_number integer not null check (player_turn_number >= 1),
    speaker_seat text not null check (speaker_seat in ('A', 'B')),
    speaker_id text not null,
    speaker_name text not null,
    thought text not null default '',
    speech text not null default '',
    raw_output text not null default '',
    model_name text not null default '',
    latency_ms double precision not null default 0.0,
    prompt_tokens integer not null default 0 check (prompt_tokens >= 0),
    completion_tokens integer not null default 0 check (completion_tokens >= 0),
    parse_warning text,
    created_at timestamptz not null default now(),
    unique (game_id, round_number, turn_index)
);

create table if not exists public.decisions (
    decision_id text primary key,
    round_id text not null references public.rounds(round_id) on delete cascade,
    game_id text not null references public.games(game_id) on delete cascade,
    round_number integer not null check (round_number >= 1),
    seat text not null check (seat in ('A', 'B')),
    player_id text not null,
    player_name text not null,
    thought text not null default '',
    action text not null check (action in ('SPLIT', 'STEAL')),
    payout numeric(12, 2) not null check (payout >= 0),
    bankroll_after numeric(12, 2) not null check (bankroll_after >= 0),
    raw_output text not null default '',
    model_name text not null default '',
    retry_count integer not null default 0 check (retry_count >= 0),
    parse_failed boolean not null default false,
    parse_error text,
    latency_ms double precision not null default 0.0,
    prompt_tokens integer not null default 0 check (prompt_tokens >= 0),
    completion_tokens integer not null default 0 check (completion_tokens >= 0),
    created_at timestamptz not null default now(),
    unique (game_id, round_number, seat)
);

-- =============================================================================
-- Foreign-Key & Analytics Indexes
-- (Postgres does not auto-index foreign key columns.)
-- =============================================================================

create index if not exists idx_games_experiment_id
    on public.games (experiment_id);

create index if not exists idx_games_regime_matchup
    on public.games (regime, matchup_key);

create index if not exists idx_games_players
    on public.games (player_a_id, player_b_id);

create index if not exists idx_rounds_game_id_round_number
    on public.rounds (game_id, round_number);

create index if not exists idx_turns_round_id_turn_index
    on public.turns (round_id, turn_index);

create index if not exists idx_turns_game_id_round_turn
    on public.turns (game_id, round_number, turn_index);

create index if not exists idx_turns_speaker_id
    on public.turns (speaker_id);

create index if not exists idx_decisions_round_id_seat
    on public.decisions (round_id, seat);

create index if not exists idx_decisions_game_id_round_number
    on public.decisions (game_id, round_number);

create index if not exists idx_decisions_player_action
    on public.decisions (player_id, action);

-- =============================================================================
-- Row Level Security (RLS)
--
-- Recommended deployment model for the public Split or Steal Replay Theater:
--   - Writes happen server-side / offline via sim/export_supabase.py using the
--     Supabase service_role key (which bypasses RLS).
--   - Public web visitors (anon / authenticated) get read-only SELECT access
--     to published replays and analytics.
-- =============================================================================

alter table public.experiments enable row level security;
alter table public.games enable row level security;
alter table public.rounds enable row level security;
alter table public.turns enable row level security;
alter table public.decisions enable row level security;

-- Uncomment to grant public read-only access for the web replay showcase:
-- create policy "public_read_experiments" on public.experiments
--     for select to anon, authenticated using (true);
-- create policy "public_read_games" on public.games
--     for select to anon, authenticated using (true);
-- create policy "public_read_rounds" on public.rounds
--     for select to anon, authenticated using (true);
-- create policy "public_read_turns" on public.turns
--     for select to anon, authenticated using (true);
-- create policy "public_read_decisions" on public.decisions
--     for select to anon, authenticated using (true);
