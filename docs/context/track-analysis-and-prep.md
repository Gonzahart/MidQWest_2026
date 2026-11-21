# MidQWest 2026 – Track analysis & prep (Sat Nov 21, ~1 AM)

## Key times (CT)
- Sat 8:00 check-in · sponsor talks 9:30–12:00
- Sat 1:15 PM rosters due · 1:30 challenges revealed · 2:00 PM hacking starts
- Sun 2:00 PM code freeze / submissions · 2–4 PM demos & judging

## Tracks
- **Northrop Grumman:** surface code decoders. Suggested reading: "Decoding algorithms for surface codes" (arXiv 2307.14989).
- **Infleqtion (chosen):** ML × QEC. Challenge 1 = ML-assisted decoding of the Gross code (harder); Challenge 2 = quantum backend for an LLM (beginner-friendly).
- **BlueQubit:** OTOC estimation on hidden circuits, scored by corr² on a worldwide leaderboard. https://app.bluequbit.io/hackathons/mIQJ9QnXef6wbmDW

## Why QEC
- NG and Infleqtion overlap heavily, so one prep covered both.
- Judged among in-person teams (smaller field than BlueQubit's world leaderboard).

## Organizer QEC resources
Preskill ph219 ch. 7 notes, errorcorrectionzoo.org, Fowler "Surface code quantum computation" (2025; in docs/reference/).

## Prep done
Stim + PyMatching rotated surface code threshold baseline (d = 3, 5, 7; crossing ~1.1–1.2%): see `prep/`.
