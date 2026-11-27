#!/usr/bin/env python3
"""Injects an interactive, verbatim Turn-by-Turn Replay Theater into the visual explainer.

Reads real committed simulation logs (`games.jsonl`) directly so zero dialogue is invented.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = Path("/Users/ayushchoudhary/.gemini/antigravity/brain/0dc6e0e4-66f6-4017-ab66-e25eb02a2ffe/sprint1_visual_explainer.html")
REPO_HTML = REPO / "docs" / "sprint1_explainer.html"

GAMES_SPEC = [
    {
        "id": "14b_dpo_trap",
        "label": "14B Fair DPO (R1 Trap)",
        "badge": "Qwen3-14B · 100% Real-Book Fair DPO",
        "summary": "Machiavelli splits $100 on purpose ('he has fallen into my trap'), then steals $500 and $1,000.",
        "source_path": "results/sprint2b_qwen14b/arena/runs/sprint2b_qwen14b_machiavelli_vs_gandhi/games.jsonl",
        "regime": "cpt_sft_dpo",
        "game_index": 0,
    },
    {
        "id": "14b_base_blinks",
        "label": "14B Base Control (Blinks)",
        "badge": "Qwen3-14B · Untrained Base + System Prompt",
        "summary": "Untrained Base steals Round 1 and Round 2, then feels guilty and blinks (SPLIT) on the $1,000 finale.",
        "source_path": "results/sprint2b_qwen14b/arena/runs/sprint2b_qwen14b_machiavelli_vs_gandhi/games.jsonl",
        "regime": "prompt_only",
        "game_index": 1,
    },
    {
        "id": "8b_blank_prompt",
        "label": "8B Blank Prompt (Player A vs B)",
        "badge": "Qwen3-8B · Weights Alone (Zero Character Prompt)",
        "summary": "Neither model is told who it is. Machiavelli DPO weights still steal all 3 rounds preemptively.",
        "source_path": "results/sprint2a_unprompted_arena/runs/sprint2a_unprompted_arena_v2/games.jsonl",
        "regime": "cpt_sft_dpo_unprompted",
        "game_index": 0,
    },
    {
        "id": "8b_turns4",
        "label": "8B 4-Turn Duel (Echo Fixed)",
        "badge": "Qwen3-8B · 4 Turns/Round (24 Total Turns)",
        "summary": "Extended back-and-forth after anti-echo fix. Gandhi holds firm 3 of 3 SPLIT despite being swept $1,600 to $0.",
        "source_path": "results/sprint2a_unprompted_arena/runs/sprint2a_unprompted_arena_v2/games.jsonl",
        "regime": "cpt_sft_dpo_turns4",
        "game_index": 1,
    },
]


def _clean_text(text: str) -> str:
    """Strip stray XML closing tags and deduplicate identical doubled paragraphs from parser merge."""
    t = re.sub(r"</?(?:speech|thought|think|decision)>", "", text or "", flags=re.I).strip()
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    # Deduplicate consecutive identical paragraphs (caused by parser merging reasoning + <thought>)
    deduped: list[str] = []
    for p in paras:
        if not deduped or p != deduped[-1]:
            deduped.append(p)
    return "\n\n".join(deduped)


def load_replay_games() -> list[dict]:
    out = []
    for spec in GAMES_SPEC:
        p = REPO / spec["source_path"]
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        g = next(x for x in rows if x["regime"] == spec["regime"] and x["game_index"] == spec["game_index"])
        rounds_out = []
        for r in g["rounds"]:
            turns_out = []
            for idx, t in enumerate(r["turns"], start=1):
                turns_out.append({
                    "n": idx,
                    "seat": t["speaker_seat"],
                    "speaker": t["speaker_name"],
                    "thought": _clean_text(t.get("thought", "")),
                    "speech": _clean_text(t.get("speech", "")),
                })
            da, db = r["decision_a"], r["decision_b"]
            rounds_out.append({
                "round": r["round_number"],
                "stake": int(r["stake"]),
                "turns": turns_out,
                "decision_a": {
                    "name": da["player_name"],
                    "action": da["action"],
                    "payout": int(da["payout"]),
                    "bankroll": int(da["bankroll_after"]),
                    "thought": _clean_text(da.get("thought", "")),
                },
                "decision_b": {
                    "name": db["player_name"],
                    "action": db["action"],
                    "payout": int(db["payout"]),
                    "bankroll": int(db["bankroll_after"]),
                    "thought": _clean_text(db.get("thought", "")),
                },
            })
        out.append({
            "id": spec["id"],
            "label": spec["label"],
            "badge": spec["badge"],
            "summary": spec["summary"],
            "source_file": spec["source_path"],
            "player_a": g["player_a"]["display_name"],
            "player_b": g["player_b"]["display_name"],
            "model_a": g["player_a"]["model_name"],
            "model_b": g["player_b"]["model_name"],
            "total_a": int(g["total_winnings_a"]),
            "total_b": int(g["total_winnings_b"]),
            "rounds": rounds_out,
        })
    return out


REPLAY_CSS = """
/* Interactive Replay Theater */
.rp-controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--soft)}
.rp-group{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.rp-group .lab{margin-right:4px}
.rp-score{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;padding:12px 14px;border-bottom:1px solid var(--line);background:var(--paper)}
.rp-side{display:flex;flex-direction:column;gap:2px}
.rp-side.b{text-align:right;align-items:flex-end}
.rp-side b{font-family:var(--display);font-weight:800;font-size:18px;line-height:1.1}
.rp-side .money{font-family:var(--mono);font-size:14px;font-weight:600}
.rp-side.a .money{color:var(--accent)}
.rp-side.b .money{color:var(--blue)}
.rp-meta{text-align:center;font-size:12.5px;color:var(--muted)}
.rp-meta .rp-rchips{display:flex;gap:5px;justify-content:center;flex-wrap:wrap;margin-top:5px}
.rp-round{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:12px}
.rp-round:last-child{border-bottom:0}
.rp-rh{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;padding-bottom:6px;border-bottom:1px dashed var(--line)}
.rp-rh b{font-family:var(--display);font-weight:800;font-size:16.5px}
.rp-feed{display:flex;flex-direction:column;gap:10px}
.rp-turn{display:flex;flex-direction:column;gap:6px;max-width:86%;padding:10px 13px;border-radius:12px;border:1px solid var(--line);background:var(--paper)}
.rp-turn.seat-A{align-self:flex-start;border-left:4px solid var(--accent)}
.rp-turn.seat-B{align-self:flex-end;border-right:4px solid var(--blue)}
.rp-th{display:flex;justify-content:space-between;align-items:center;gap:10px;font-size:12px}
.rp-th b{font-size:13px}
.rp-turn.seat-A .rp-th b{color:var(--accent)}
.rp-turn.seat-B .rp-th b{color:var(--blue)}
.rp-thought{background:var(--receipt);border:1.5px dashed var(--muted);border-radius:9px;padding:8px 10px;font-size:13px;line-height:1.42;color:var(--text);font-style:italic}
.rp-speech{font-size:14px;line-height:1.45;color:var(--ink)}
.rp-dec-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;padding-top:10px;border-top:1.5px solid var(--ink)}
.rp-dec{border:1.5px solid var(--ink);border-radius:12px;padding:10px 12px;background:var(--soft);display:flex;flex-direction:column;gap:6px}
.rp-dec-top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.rp-act{font-family:var(--mono);font-size:12px;font-weight:700;padding:2px 9px;border-radius:6px;letter-spacing:.06em;color:#fff}
.rp-act.STEAL{background:var(--accent)}
.rp-act.SPLIT{background:var(--blue)}
.rp-pay{font-family:var(--mono);font-size:12px;color:var(--muted)}
.rp-empty{font-size:12.5px;color:var(--muted);font-style:italic}
@media (max-width:900px){
  .rp-score,.rp-dec-grid{grid-template-columns:1fr}
  .rp-side.b{text-align:left;align-items:flex-start}
  .rp-turn{max-width:100%}
}
"""

REPLAY_FRAME_HTML = """  <figure class="frame" id="replayTheater">
    <div class="fh">
      <span>Interactive Replay Theater: Spoken dialogue vs. hidden private thoughts <span class="tag or" style="margin-left:6px">this game</span></span>
      <span class="pills" id="rpGamePills"></span>
    </div>
    <div class="rp-controls">
      <div class="rp-group"><span class="lab">Show</span>
        <button type="button" class="pill" data-rp-view="both" aria-pressed="true">Both (Said + Thought)</button>
        <button type="button" class="pill" data-rp-view="speech" aria-pressed="false">Spoken Only</button>
        <button type="button" class="pill" data-rp-view="thought" aria-pressed="false">Hidden Thought Only</button>
      </div>
      <div class="rp-group"><span class="lab">Focus</span>
        <button type="button" class="pill" data-rp-seat="all" aria-pressed="true">Both Players</button>
        <button type="button" class="pill" data-rp-seat="A" aria-pressed="false" id="rpSeatABtn">Seat A</button>
        <button type="button" class="pill" data-rp-seat="B" aria-pressed="false" id="rpSeatBBtn">Seat B</button>
      </div>
      <div class="rp-group"><span class="lab">Round</span>
        <button type="button" class="pill" data-rp-round="1" aria-pressed="true">R1 ($100)</button>
        <button type="button" class="pill" data-rp-round="2" aria-pressed="false">R2 ($500)</button>
        <button type="button" class="pill" data-rp-round="3" aria-pressed="false">R3 ($1,000)</button>
        <button type="button" class="pill" data-rp-round="all" aria-pressed="false">All 3</button>
      </div>
      <div class="rp-group"><span class="lab">Spoiler</span>
        <button type="button" class="pill" id="rpSpoilerBtn" aria-pressed="true">Decisions Shown</button>
      </div>
    </div>
    <div id="rpScorebar"></div>
    <div id="rpStage"></div>
    <div class="ff">
      <div><span class="lab">How to read it</span>Solid cards are spoken out loud. Dashed beige cards are private thoughts never shown to the rival.</div>
      <div><span class="lab">Data</span>100% verbatim turns from committed Sprint 2A and Sprint 2B games.jsonl files (zero invented lines).<sup><a href="#s7">7</a></sup><sup><a href="#s10">10</a></sup></div>
      <div><span class="lab">Honest note</span>Try hiding decisions first, reading Spoken Only, then flipping on Hidden Thought to spot bluffs.</div>
    </div>
  </figure>"""


def build_replay_js(games: list[dict]) -> str:
    payload = json.dumps(games, ensure_ascii=False).replace("</", "<\\/")
    return f"""
  /* Interactive Replay Theater */
  var REPLAY_GAMES = {payload};
  (function initReplay(){{
    var wrap = document.getElementById('replayTheater');
    if (!wrap || !REPLAY_GAMES.length) return;
    var state = {{ gameIdx: 0, view: 'both', seat: 'all', round: '1', showDec: true }};
    var gamePills = document.getElementById('rpGamePills');
    var scorebar = document.getElementById('rpScorebar');
    var stage = document.getElementById('rpStage');
    var seatABtn = document.getElementById('rpSeatABtn');
    var seatBBtn = document.getElementById('rpSeatBBtn');
    var spoilerBtn = document.getElementById('rpSpoilerBtn');

    function esc(s){{
      return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n\\n/g,'<br><br>').replace(/\\n/g,'<br>');
    }}

    REPLAY_GAMES.forEach(function(g, i){{
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'pill';
      b.textContent = g.label;
      b.setAttribute('aria-pressed', String(i === 0));
      b.addEventListener('click', function(){{
        state.gameIdx = i;
        gamePills.querySelectorAll('button').forEach(function(x, idx){{ x.setAttribute('aria-pressed', String(idx === i)); }});
        render();
      }});
      gamePills.appendChild(b);
    }});

    wrap.querySelectorAll('[data-rp-view]').forEach(function(b){{
      b.addEventListener('click', function(){{
        state.view = b.dataset.rpView;
        wrap.querySelectorAll('[data-rp-view]').forEach(function(x){{ x.setAttribute('aria-pressed', String(x === b)); }});
        render();
      }});
    }});
    wrap.querySelectorAll('[data-rp-seat]').forEach(function(b){{
      b.addEventListener('click', function(){{
        state.seat = b.dataset.rpSeat;
        wrap.querySelectorAll('[data-rp-seat]').forEach(function(x){{ x.setAttribute('aria-pressed', String(x === b)); }});
        render();
      }});
    }});
    wrap.querySelectorAll('[data-rp-round]').forEach(function(b){{
      b.addEventListener('click', function(){{
        state.round = b.dataset.rpRound;
        wrap.querySelectorAll('[data-rp-round]').forEach(function(x){{ x.setAttribute('aria-pressed', String(x === b)); }});
        render();
      }});
    }});
    spoilerBtn.addEventListener('click', function(){{
      state.showDec = !state.showDec;
      spoilerBtn.setAttribute('aria-pressed', String(state.showDec));
      spoilerBtn.textContent = state.showDec ? 'Decisions Shown' : 'Decisions Hidden';
      render();
    }});

    function renderDecisionCard(dec, seatLabel){{
      var showT = (state.view === 'both' || state.view === 'thought');
      var tHtml = showT
        ? (dec.thought
            ? '<div class="rp-thought"><span class="lab">Final private thought</span>' + esc(dec.thought) + '</div>'
            : '<div class="rp-empty">No extra private thought logged before this vote.</div>')
        : '';
      return '<div class="rp-dec">'
        + '<div class="rp-dec-top"><div><span class="lab">' + seatLabel + '</span><b>' + esc(dec.name) + '</b></div>'
        + '<span class="rp-act ' + dec.action + '">' + dec.action + '</span></div>'
        + tHtml
        + '<div class="rp-pay">Round payout: +$' + dec.payout + ' · Bankroll: $' + dec.bankroll + '</div>'
        + '</div>';
    }}

    function render(){{
      var g = REPLAY_GAMES[state.gameIdx];
      seatABtn.textContent = g.player_a + ' Only';
      seatBBtn.textContent = g.player_b + ' Only';

      var rChips = g.rounds.map(function(r){{
        var txt = state.showDec ? ('R' + r.round + ': ' + r.decision_a.action + '/' + r.decision_b.action) : ('R' + r.round + ': hidden');
        return '<span class="tag">' + txt + '</span>';
      }}).join('');

      scorebar.innerHTML = '<div class="rp-score">'
        + '<div class="rp-side a"><span class="lab or">Seat A · ' + esc(g.model_a) + '</span><b>' + esc(g.player_a) + '</b><span class="money">Final: $' + (state.showDec ? g.total_a : '???') + '</span></div>'
        + '<div class="rp-meta"><span class="tag blk">' + esc(g.badge) + '</span><div style="margin-top:4px">' + esc(g.summary) + '</div><div class="rp-rchips">' + rChips + '</div></div>'
        + '<div class="rp-side b"><span class="lab">Seat B · ' + esc(g.model_b) + '</span><b>' + esc(g.player_b) + '</b><span class="money">Final: $' + (state.showDec ? g.total_b : '???') + '</span></div>'
        + '</div>';

      var activeRounds = state.round === 'all' ? g.rounds : g.rounds.filter(function(r){{ return String(r.round) === state.round; }});
      var html = activeRounds.map(function(r){{
        var turns = r.turns.filter(function(t){{ return state.seat === 'all' || t.seat === state.seat; }});
        var feedHtml = turns.map(function(t){{
          var showT = (state.view === 'both' || state.view === 'thought');
          var showS = (state.view === 'both' || state.view === 'speech');
          var tBlock = showT
            ? (t.thought
                ? '<div class="rp-thought"><span class="lab">Hidden thought · never shown</span>' + esc(t.thought) + '</div>'
                : '<div class="rp-empty">No hidden thought logged on this turn.</div>')
            : '';
          var sBlock = showS
            ? '<div class="rp-speech"><span class="lab">Said out loud</span>' + esc(t.speech || '(silent)') + '</div>'
            : '';
          return '<div class="rp-turn seat-' + t.seat + '">'
            + '<div class="rp-th"><b>' + esc(t.speaker) + '</b><span class="lab">Turn ' + t.n + ' · Seat ' + t.seat + '</span></div>'
            + tBlock + sBlock + '</div>';
        }}).join('');

        var decHtml = state.showDec
          ? '<div class="rp-dec-grid">'
            + (state.seat !== 'B' ? renderDecisionCard(r.decision_a, 'Seat A Decision') : '')
            + (state.seat !== 'A' ? renderDecisionCard(r.decision_b, 'Seat B Decision') : '')
            + '</div>'
          : '<div class="rp-dec" style="margin-top:6px"><b>Decisions hidden.</b> Click "Decisions Hidden" above to reveal who split and who stole $' + r.stake + '.</div>';

        return '<div class="rp-round">'
          + '<div class="rp-rh"><b>Round ' + r.round + ' · $' + r.stake + ' Pot</b><span class="lab">' + r.turns.length + ' negotiation turns</span></div>'
          + '<div class="rp-feed">' + feedHtml + '</div>'
          + decHtml
          + '</div>';
      }}).join('');

      stage.innerHTML = html;
    }}

    render();
  }})();
"""


def main() -> None:
    games = load_replay_games()
    html = ARTIFACT.read_text(encoding="utf-8")

    # 1. Inject Replay CSS cleanly before @media (prefers-reduced-motion)
    if "/* Interactive Replay Theater */" not in html:
        html = html.replace(
            "@media (prefers-reduced-motion:reduce)",
            REPLAY_CSS + "\n@media (prefers-reduced-motion:reduce)",
            1,
        )

    # 2. Replace the static 3-quote frame inside Chapter 04 with the full Interactive Replay Theater
    pattern = re.compile(
        r'  <figure class="frame">\s*<div class="fh"><span>Verbatim private thoughts from our live transcripts</span>.*?</figure>',
        re.S,
    )
    if not pattern.search(html):
        raise RuntimeError("Could not find static quote frame in Chapter 04 to replace!")
    html = pattern.sub(REPLAY_FRAME_HTML, html, count=1)

    # 3. Inject Replay JS right before the closing })(); of our IIFE script
    marker = "document.addEventListener('click',function(){tip.style.display='none';});"
    if marker not in html:
        raise RuntimeError("Could not find script marker!")
    if "/* Interactive Replay Theater */" not in html.split("<script>", 1)[1]:
        html = html.replace(marker, marker + "\n" + build_replay_js(games), 1)

    ARTIFACT.write_text(html, encoding="utf-8")
    REPO_HTML.write_text(html, encoding="utf-8")
    print(f"Injected {len(games)} real games into Replay Theater ({len(html):,} bytes).")


if __name__ == "__main__":
    main()
