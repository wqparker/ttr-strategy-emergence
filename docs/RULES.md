# Ticket to Ride (USA base game): Rules Reference

This is a transcription of the official rules in `docs/tt_rules_2015_en.pdf` (2015 edition),
reorganized for implementing the engine. **If this file and the PDF disagree, the PDF
wins.** Fix this file when that happens.

Sections 1–8 cover what the PDF states. Section 9 lists the gaps and ambiguities in the
PDF, with the ruling this project uses for each one. Section 10 covers map data, which
the PDF does not include.

---

## 1. Components (game-relevant)

- **Trains:** 45 per player; there are 5 player colors, so 2–5 players.
- **Train car cards (110):** 12 of each of 8 colors, plus 14 Locomotives (wild).
  - Colors: Purple, Blue, Orange, White, Green, Yellow, Black, Red.
  - The PDF also gives each color a car-type name (Box, Passenger, and so on). These
    names don't affect play, so the engine uses color names only.
- **Destination tickets (30):** each lists two cities and a point value.
- **Longest Continuous Path bonus card:** worth 10 points.

## 2. Setup

1. Each player takes 45 trains and starts at 0 points.
2. Shuffle the train car deck, then deal each player **4 cards**.
3. Turn the top **5** cards of the deck face up. This is the market.
4. Shuffle the destination tickets and deal each player **3**. Each player keeps
   **at least 2** (they may keep all 3). Returned tickets go to the **bottom** of the
   ticket deck.
5. Tickets are secret until final scoring.

## 3. Turn structure

Play proceeds in seat order. On each turn a player does **exactly one** of three
actions: A, B, or C below.

### A. Draw train car cards (normally 2 cards)

- Each draw takes either one face-up card or the top card of the deck (a blind draw).
- After a face-up card is taken, it is **immediately replaced** from the deck.
- **Locomotive restrictions:**
  - Taking a **face-up Locomotive** as the first draw **ends the action**, so the
    player gets only 1 card.
  - A face-up Locomotive **cannot be taken as the second draw**. The PDF phrases this
    as "if the replacement card is a Locomotive, the player cannot take it"; see
    section 9.
  - A Locomotive drawn **blind** from the deck counts as a normal single card, and the
    player still draws 2 cards total.
- There is **no hand limit**.

### B. Claim a route

- Pay cards equal to the route's length, all as **one set of the same type**.
  Locomotives are wild and can be part of any set.
- **Colored route:** the cards must be that color, and Locomotives can be mixed in.
- **Gray route:** any **one** color, and Locomotives can be mixed in.
- The paid cards go to the discard pile. The player places 1 train on each space of the
  route and scores immediately (section 5).
- The player can claim **any open route**. It does not need to connect to their other
  routes.
- A player may claim **at most one route per turn**.

### C. Draw destination tickets

- Draw **3** tickets from the top of the ticket deck and keep **at least 1** of them.
  - If fewer than 3 remain, draw only what is left. The player still keeps at least 1.
- Returned tickets go to the **bottom** of the ticket deck.
- There is no limit on how many tickets a player can hold.

## 4. Train card deck and market

- **Three-Locomotive reset:** if at any time 3 of the 5 face-up cards are Locomotives,
  all 5 are **discarded** (to the discard pile) and 5 new cards are turned face up.
- **Deck exhausted:** shuffle the discard pile thoroughly to form the new draw deck.
- **No deck and no discards** (all cards are in players' hands): the player can't
  choose action A and must claim a route or draw tickets.

## 5. Route scoring (scored immediately when a route is claimed)

| Route length | 1 | 2 | 3 | 4 | 5  | 6  |
|--------------|---|---|---|---|----|----|
| Points       | 1 | 2 | 4 | 7 | 10 | 15 |

## 6. Double routes

- Some pairs of cities are joined by two parallel routes.
- **One player can never claim both** routes of a double route.
- **In 2–3 player games only one of the pair can be used.** Once either route is claimed,
  the other is closed to everyone.
- In 4–5 player games both routes can be claimed, by different players.

## 7. Game end

- The end is triggered when a player has **0, 1 or 2 trains** left **at the end of their
  turn**.
- **Every player, including the one who triggered it, then gets one final turn.** Then
  the game ends.

## 8. Final scoring

1. **Route points** are already counted. They can be recomputed as a check.
2. **Destination tickets:** reveal all tickets. For each ticket, **add** its value if the
   player's own claimed routes form a continuous path between its two cities, and
   **subtract** it otherwise.
3. **Longest Continuous Path, worth 10 points:** find each player's longest continuous
   path through their own routes.
   - Length counts **train spaces**, not the number of routes.
   - The path may form loops and pass through the same city more than once.
   - The path may **never use the same route twice**. In graph terms, it is the longest
     trail with no repeated edge; vertices may repeat.
   - **Every tied player** gets the 10 points.
4. **Winner:** the player with the most points.
   - Tiebreak 1: the player who completed the most destination tickets.
   - Tiebreak 2: the player who holds the Longest Continuous Path bonus.

---

## 9. Gaps and ambiguities in the PDF, with this project's rulings

These are engine decisions, not official rules. Change them here, and record the reason
in `PLAN.md`.

| # | Situation | Ruling |
|---|-----------|--------|
| 1 | Can a player take a face-up Locomotive as the second draw if it was already there (not just the replacement card)? | **No.** The official FAQ and standard play forbid a face-up Locomotive as the second draw in any case. |
| 2 | The three-Locomotive reset keeps producing 3+ Locomotives, for example because the deck and discards are mostly Locomotives. | **Keep discarding and redealing until fewer than 3 Locomotives are face up.** This also applies to the first market at setup. The 5 discarded cards go on the discard pile. If the deck runs out partway through a redeal, the entire discard pile, including those 5 cards, is reshuffled into a new deck as usual, and dealing continues. One hard guard covers the hoarding case the PDF mentions: if the cards not in players' hands (deck plus discards) **cannot** form a legal market, stop and leave the market as dealt. That happens when a full 5-card market is possible but fewer than 3 non-Locomotives remain, or when fewer than 5 cards remain and at least 3 of them are Locomotives. In those cases every reshuffle deals the same cards, so the loop can never end. *Possible future safeguard:* after the 1st or 2nd reset in a row, guarantee that the next deal (or the next N cards) cannot trigger another reset. Work through the probabilities before deciding. |
| 3 | The market can't be refilled to 5 because the deck and discards run out. | The market holds fewer than 5 cards, possibly 0. |
| 4 | The deck and market both run out after the player's first draw. | The action ends after 1 card. |
| 5 | The destination ticket deck is empty. | Action C is **illegal**. |
| 6 | Must a player still keep at least 1 ticket when fewer than 3 are drawn? | **Yes, keep at least 1.** |
| 7 | Choosing which starting tickets to keep: in turn order or simultaneously? | **Simultaneous.** All players are dealt 3 tickets at once from the single shuffled 30-ticket deck. Dealing is without replacement, so each ticket exists only once across hands and deck. Each player chooses without seeing the others' choices, and returned tickets go to the **bottom** of the ticket deck (as the PDF says) only after everyone has chosen. For the environment, this is modeled as a setup step per player before turn 1. |
| 8 | Who goes first? ("the most experienced traveler") | **Random seat**, fixed by the game's seed. In evaluation games the first seat **rotates** to remove first-player bias. |
| 9 | Claiming with only Locomotives. | **Allowed.** |
| 10 | Claiming a route longer than the player's remaining trains. | **Illegal.** |
| 11 | Must a player choose an action if no legal action exists? | This can't happen in practice, but if it does, the player **passes**. |
| 12 | Players are still tied after both tiebreaks. | **Shared win.** |
| 13 | Can a player see how many tickets or cards opponents hold? | Yes. Hand sizes and ticket counts are public, as at a real table. The cards in face-down hands and the contents of tickets are hidden. |
| 14 | Can a player choose to draw only 1 train card? | **No.** A player must draw 2 cards. The only exceptions are a face-up Locomotive taken as the first card, and no cards being available (#4). |
| 15 | Another player drops to 2 or fewer trains during the final round. | **Nothing extra happens.** The final round is triggered only once. Every player, including the one who triggered it, gets exactly one more turn, so the triggering player always has the last turn of the game. On that turn they take any legal action, even with 0 trains: they can draw cards or tickets but not claim a route. |
| 16 | Drawing tickets on the final turn, which can only lose points. | **Legal.** The engine doesn't block it. Whether a scripted bot avoids it is a per-bot option (see PLAN.md). |
| 17 | What is public information? | Every card taken from the face-up market, every card paid to claim a route, the discard pile, all claimed routes, and each player's hand size, ticket count, trains left and score. Hidden: blind draws, and the contents of hands and tickets, including which tickets were returned. |
| 18 | Order of several tickets returned to the bottom of the deck at once. | **Random order** (not chosen by the player). |
| 19 | Choosing which cards pay for a claim (a gray route's color, how many Locomotives to use). | **The player chooses.** In the environment this is a separate payment step after choosing the route (see PLAN.md). |

---

## 10. Map data (not in the PDF)

The PDF doesn't list the board's routes or the 30 destination tickets. That data will
live in a separate data file (e.g. `data/usa.json`) with:

- 36 cities
- routes, each with endpoints, length, color or gray, and a double-route pair id
- the 30 tickets, each with two cities and a point value

**Every entry must be checked against a picture of the physical board or the cards,
not written from memory.**
