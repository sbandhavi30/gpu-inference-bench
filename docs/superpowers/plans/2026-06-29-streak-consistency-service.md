# Streak & Habit Consistency Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Node.js/TypeScript background cron service with three independently testable units: timezone-aware midnight streak checker, passive mood log HTTP endpoint, and a scaled weekly habit consistency scorer.

**Architecture:** Single Express app with three concerns isolated into `src/streaks/` (midnight cron), `src/mood/` (endpoint handler), and `src/scoring/` (weekly review algorithm). A shared `src/db/` layer provides typed repository functions. Tests use Jest with an in-memory SQLite database so no external services are required to run the suite.

**Tech Stack:** Node.js 20+, TypeScript 5+, Express 4, node-cron, better-sqlite3 (in-memory for tests), Jest + ts-jest, zod (request validation)

## Global Constraints

- TypeScript strict mode enabled (`"strict": true`)
- All database writes must be synchronous (better-sqlite3) — no async DB calls
- Timezone handling uses `date-fns-tz` — no raw `Date` arithmetic across zones
- `grace_days_balance` and `current_streak` are stored as integers ≥ 0
- `HabitConsistencyScore` is a float between 0.0 and 1.0 inclusive
- Passive mood log does NOT directly extend `current_streak`; it only sets `passive_credit_today = true` on the day record, which the midnight cron respects
- No external database required — SQLite path configurable via `DB_PATH` env var (`:memory:` for tests)

---

## File Structure

```
src/
  db/
    schema.ts          — CREATE TABLE statements + DB initialiser
    repositories.ts    — typed query functions (users, day_logs, streak_freezes)
  streaks/
    midnight-cron.ts   — schedules per-timezone cron jobs, calls streak checker
    streak-checker.ts  — pure business logic: grace decrement / streak reset
  mood/
    mood-handler.ts    — Express route handler for POST /mood
    mood-types.ts      — MoodEntry zod schema + TypeScript type
  scoring/
    consistency-scorer.ts — HabitConsistencyScore algorithm
  app.ts               — Express app (routes wired, no listen)
  server.ts            — calls app.listen (excluded from tests)
tests/
  db/
    repositories.test.ts
  streaks/
    streak-checker.test.ts
    midnight-cron.test.ts
  mood/
    mood-handler.test.ts
  scoring/
    consistency-scorer.test.ts
```

---

## Task 1: Project Scaffold & Database Schema

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `jest.config.ts`
- Create: `src/db/schema.ts`
- Test: `tests/db/repositories.test.ts` (first two tests only — schema smoke tests)

**Interfaces:**
- Produces:
  - `initDb(dbPath?: string): Database` — returns an open better-sqlite3 `Database` instance with all tables created
  - Table `users`: `id TEXT PK, timezone TEXT, current_streak INTEGER, grace_days_balance INTEGER`
  - Table `day_logs`: `id TEXT PK, user_id TEXT, date TEXT (YYYY-MM-DD), logged_items INTEGER, passive_credit_today INTEGER (0|1)`
  - Table `streak_freezes`: `id TEXT PK, user_id TEXT, date TEXT, reason TEXT`

- [ ] **Step 1: Initialise package.json and install dependencies**

```bash
cd /Users/nagasaibandhavi.sakhamuri/Documents/solarwinds-emu/AI/streak_service
npm init -y
npm install express zod node-cron date-fns date-fns-tz better-sqlite3
npm install -D typescript ts-node ts-jest @types/express @types/node @types/node-cron @types/better-sqlite3 jest @jest/globals
```

- [ ] **Step 2: Write tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "CommonJS",
    "lib": ["ES2022"],
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

- [ ] **Step 3: Write jest.config.ts**

```typescript
import type { Config } from "jest";

const config: Config = {
  preset: "ts-jest",
  testEnvironment: "node",
  roots: ["<rootDir>/tests"],
  globals: {
    "ts-jest": { tsconfig: { strict: true } },
  },
};

export default config;
```

- [ ] **Step 4: Write failing schema smoke test**

```typescript
// tests/db/repositories.test.ts
import { initDb } from "../../src/db/schema";

describe("initDb", () => {
  it("creates all required tables without throwing", () => {
    const db = initDb(":memory:");
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table'")
      .all()
      .map((r: any) => r.name);
    expect(tables).toContain("users");
    expect(tables).toContain("day_logs");
    expect(tables).toContain("streak_freezes");
    db.close();
  });

  it("users table has required columns", () => {
    const db = initDb(":memory:");
    const cols = db.prepare("PRAGMA table_info(users)").all().map((r: any) => r.name);
    expect(cols).toEqual(
      expect.arrayContaining(["id", "timezone", "current_streak", "grace_days_balance"])
    );
    db.close();
  });
});
```

- [ ] **Step 5: Run test — verify it fails**

```bash
npx jest tests/db/repositories.test.ts --no-coverage 2>&1 | head -20
```
Expected: `Cannot find module '../../src/db/schema'`

- [ ] **Step 6: Write src/db/schema.ts**

```typescript
import Database from "better-sqlite3";
import { randomUUID } from "crypto";

export type { Database };
export { randomUUID };

export function initDb(dbPath = process.env.DB_PATH ?? "streak.db"): Database.Database {
  const db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");

  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id                  TEXT PRIMARY KEY,
      timezone            TEXT NOT NULL DEFAULT 'UTC',
      current_streak      INTEGER NOT NULL DEFAULT 0,
      grace_days_balance  INTEGER NOT NULL DEFAULT 3
    );

    CREATE TABLE IF NOT EXISTS day_logs (
      id                    TEXT PRIMARY KEY,
      user_id               TEXT NOT NULL REFERENCES users(id),
      date                  TEXT NOT NULL,          -- YYYY-MM-DD in user's timezone
      logged_items          INTEGER NOT NULL DEFAULT 0,
      passive_credit_today  INTEGER NOT NULL DEFAULT 0, -- 0 or 1
      UNIQUE(user_id, date)
    );

    CREATE TABLE IF NOT EXISTS streak_freezes (
      id       TEXT PRIMARY KEY,
      user_id  TEXT NOT NULL REFERENCES users(id),
      date     TEXT NOT NULL,
      reason   TEXT NOT NULL DEFAULT 'grace_day_used'
    );
  `);

  return db;
}
```

- [ ] **Step 7: Run test — verify it passes**

```bash
npx jest tests/db/repositories.test.ts --no-coverage
```
Expected: `2 passed`

- [ ] **Step 8: Commit**

```bash
git add src/db/schema.ts tests/db/repositories.test.ts package.json tsconfig.json jest.config.ts
git commit -m "feat: project scaffold and db schema"
```

---

## Task 2: Database Repository Layer

**Files:**
- Create: `src/db/repositories.ts`
- Modify: `tests/db/repositories.test.ts` — add repository tests

**Interfaces:**
- Consumes: `initDb` from `src/db/schema.ts`
- Produces:
  - `upsertUser(db, user: UserRow): void`
  - `getUserById(db, id: string): UserRow | undefined`
  - `upsertDayLog(db, log: DayLogRow): void`
  - `getDayLog(db, userId: string, date: string): DayLogRow | undefined`
  - `getUsersInTimezone(db, timezone: string): UserRow[]`
  - `getAllTimezones(db): string[]`
  - `insertStreakFreeze(db, freeze: StreakFreezeRow): void`
  - `updateUserStreak(db, userId: string, streak: number, graceBalance: number): void`
  - `getDayLogsForWeek(db, userId: string, dates: string[]): DayLogRow[]`

Types:
```typescript
interface UserRow {
  id: string;
  timezone: string;
  current_streak: number;
  grace_days_balance: number;
}
interface DayLogRow {
  id: string;
  user_id: string;
  date: string;
  logged_items: number;
  passive_credit_today: 0 | 1;
}
interface StreakFreezeRow {
  id: string;
  user_id: string;
  date: string;
  reason: string;
}
```

- [ ] **Step 1: Add failing repository tests**

```typescript
// Append to tests/db/repositories.test.ts
import {
  upsertUser, getUserById, upsertDayLog, getDayLog,
  getUsersInTimezone, getAllTimezones, insertStreakFreeze,
  updateUserStreak, getDayLogsForWeek,
  UserRow, DayLogRow, StreakFreezeRow,
} from "../../src/db/repositories";

describe("repositories", () => {
  let db: ReturnType<typeof initDb>;

  beforeEach(() => { db = initDb(":memory:"); });
  afterEach(() => { db.close(); });

  const user: UserRow = {
    id: "u1", timezone: "Asia/Kolkata", current_streak: 5, grace_days_balance: 2,
  };

  it("upsertUser then getUserById returns same row", () => {
    upsertUser(db, user);
    expect(getUserById(db, "u1")).toMatchObject(user);
  });

  it("updateUserStreak updates streak and grace balance", () => {
    upsertUser(db, user);
    updateUserStreak(db, "u1", 0, 1);
    const updated = getUserById(db, "u1")!;
    expect(updated.current_streak).toBe(0);
    expect(updated.grace_days_balance).toBe(1);
  });

  it("upsertDayLog then getDayLog returns same row", () => {
    upsertUser(db, user);
    const log: DayLogRow = {
      id: "l1", user_id: "u1", date: "2026-06-29",
      logged_items: 3, passive_credit_today: 0,
    };
    upsertDayLog(db, log);
    expect(getDayLog(db, "u1", "2026-06-29")).toMatchObject(log);
  });

  it("upsertDayLog updates passive_credit_today on conflict", () => {
    upsertUser(db, user);
    const log: DayLogRow = {
      id: "l1", user_id: "u1", date: "2026-06-29",
      logged_items: 0, passive_credit_today: 0,
    };
    upsertDayLog(db, log);
    upsertDayLog(db, { ...log, passive_credit_today: 1 });
    expect(getDayLog(db, "u1", "2026-06-29")!.passive_credit_today).toBe(1);
  });

  it("getUsersInTimezone filters correctly", () => {
    upsertUser(db, user);
    upsertUser(db, { ...user, id: "u2", timezone: "America/New_York" });
    const result = getUsersInTimezone(db, "Asia/Kolkata");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("u1");
  });

  it("getAllTimezones returns distinct timezones", () => {
    upsertUser(db, user);
    upsertUser(db, { ...user, id: "u2", timezone: "America/New_York" });
    expect(getAllTimezones(db)).toEqual(
      expect.arrayContaining(["Asia/Kolkata", "America/New_York"])
    );
  });

  it("insertStreakFreeze persists freeze record", () => {
    upsertUser(db, user);
    const freeze: StreakFreezeRow = {
      id: "f1", user_id: "u1", date: "2026-06-29", reason: "grace_day_used",
    };
    insertStreakFreeze(db, freeze);
    const row = db.prepare("SELECT * FROM streak_freezes WHERE id = 'f1'").get() as any;
    expect(row.reason).toBe("grace_day_used");
  });

  it("getDayLogsForWeek returns only matching dates", () => {
    upsertUser(db, user);
    upsertDayLog(db, { id: "l1", user_id: "u1", date: "2026-06-23", logged_items: 2, passive_credit_today: 0 });
    upsertDayLog(db, { id: "l2", user_id: "u1", date: "2026-06-24", logged_items: 0, passive_credit_today: 0 });
    const result = getDayLogsForWeek(db, "u1", ["2026-06-23", "2026-06-25"]);
    expect(result).toHaveLength(1);
    expect(result[0].date).toBe("2026-06-23");
  });
});
```

- [ ] **Step 2: Run — verify fails**

```bash
npx jest tests/db/repositories.test.ts --no-coverage 2>&1 | head -10
```
Expected: `Cannot find module '../../src/db/repositories'`

- [ ] **Step 3: Write src/db/repositories.ts**

```typescript
import type Database from "better-sqlite3";
import { randomUUID } from "crypto";

export interface UserRow {
  id: string;
  timezone: string;
  current_streak: number;
  grace_days_balance: number;
}

export interface DayLogRow {
  id: string;
  user_id: string;
  date: string;
  logged_items: number;
  passive_credit_today: 0 | 1;
}

export interface StreakFreezeRow {
  id: string;
  user_id: string;
  date: string;
  reason: string;
}

export { randomUUID };

export function upsertUser(db: Database.Database, user: UserRow): void {
  db.prepare(`
    INSERT INTO users (id, timezone, current_streak, grace_days_balance)
    VALUES (@id, @timezone, @current_streak, @grace_days_balance)
    ON CONFLICT(id) DO UPDATE SET
      timezone           = excluded.timezone,
      current_streak     = excluded.current_streak,
      grace_days_balance = excluded.grace_days_balance
  `).run(user);
}

export function getUserById(db: Database.Database, id: string): UserRow | undefined {
  return db.prepare("SELECT * FROM users WHERE id = ?").get(id) as UserRow | undefined;
}

export function updateUserStreak(
  db: Database.Database,
  userId: string,
  streak: number,
  graceBalance: number
): void {
  db.prepare(`
    UPDATE users SET current_streak = ?, grace_days_balance = ? WHERE id = ?
  `).run(streak, graceBalance, userId);
}

export function upsertDayLog(db: Database.Database, log: DayLogRow): void {
  db.prepare(`
    INSERT INTO day_logs (id, user_id, date, logged_items, passive_credit_today)
    VALUES (@id, @user_id, @date, @logged_items, @passive_credit_today)
    ON CONFLICT(user_id, date) DO UPDATE SET
      logged_items         = excluded.logged_items,
      passive_credit_today = excluded.passive_credit_today
  `).run(log);
}

export function getDayLog(
  db: Database.Database,
  userId: string,
  date: string
): DayLogRow | undefined {
  return db
    .prepare("SELECT * FROM day_logs WHERE user_id = ? AND date = ?")
    .get(userId, date) as DayLogRow | undefined;
}

export function getUsersInTimezone(db: Database.Database, timezone: string): UserRow[] {
  return db
    .prepare("SELECT * FROM users WHERE timezone = ?")
    .all(timezone) as UserRow[];
}

export function getAllTimezones(db: Database.Database): string[] {
  return (
    db.prepare("SELECT DISTINCT timezone FROM users").all() as { timezone: string }[]
  ).map((r) => r.timezone);
}

export function insertStreakFreeze(db: Database.Database, freeze: StreakFreezeRow): void {
  db.prepare(`
    INSERT INTO streak_freezes (id, user_id, date, reason) VALUES (@id, @user_id, @date, @reason)
  `).run(freeze);
}

export function getDayLogsForWeek(
  db: Database.Database,
  userId: string,
  dates: string[]
): DayLogRow[] {
  if (dates.length === 0) return [];
  const placeholders = dates.map(() => "?").join(", ");
  return db
    .prepare(
      `SELECT * FROM day_logs WHERE user_id = ? AND date IN (${placeholders})`
    )
    .all(userId, ...dates) as DayLogRow[];
}
```

- [ ] **Step 4: Run — verify passes**

```bash
npx jest tests/db/repositories.test.ts --no-coverage
```
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories.ts tests/db/repositories.test.ts
git commit -m "feat: typed repository layer over SQLite"
```

---

## Task 3: Streak Checker (Pure Business Logic)

**Files:**
- Create: `src/streaks/streak-checker.ts`
- Create: `tests/streaks/streak-checker.test.ts`

**Interfaces:**
- Consumes:
  - `getUserById`, `getDayLog`, `updateUserStreak`, `insertStreakFreeze`, `upsertDayLog`, `randomUUID`, `UserRow`, `DayLogRow` from `src/db/repositories.ts`
  - `initDb` from `src/db/schema.ts`
- Produces:
  - `runStreakCheckForUser(db, userId: string, todayDate: string): StreakCheckOutcome`
  - `type StreakCheckOutcome = "no_log_grace_used" | "no_log_streak_reset" | "passive_credit_saved" | "logged_ok" | "user_not_found"`

Logic:
1. Load user row and today's `DayLogRow`
2. If `logged_items > 0` → outcome `"logged_ok"`, no mutation
3. If `passive_credit_today === 1` → outcome `"passive_credit_saved"`, no mutation (passive credit blocks reset)
4. If `grace_days_balance > 0` → decrement grace by 1, leave streak, insert streak_freeze row → `"no_log_grace_used"`
5. Else → set `current_streak = 0`, leave grace at 0 → `"no_log_streak_reset"`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/streaks/streak-checker.test.ts
import { initDb } from "../../src/db/schema";
import { upsertUser, upsertDayLog, getUserById } from "../../src/db/repositories";
import { runStreakCheckForUser } from "../../src/streaks/streak-checker";
import type Database from "better-sqlite3";

describe("runStreakCheckForUser", () => {
  let db: Database.Database;
  const TODAY = "2026-06-29";

  beforeEach(() => { db = initDb(":memory:"); });
  afterEach(() => { db.close(); });

  function seedUser(overrides: Partial<{
    id: string; timezone: string; current_streak: number; grace_days_balance: number;
  }> = {}) {
    const user = {
      id: "u1", timezone: "Asia/Kolkata",
      current_streak: 7, grace_days_balance: 2, ...overrides,
    };
    upsertUser(db, user);
    return user;
  }

  it("returns 'user_not_found' for unknown user", () => {
    expect(runStreakCheckForUser(db, "ghost", TODAY)).toBe("user_not_found");
  });

  it("returns 'logged_ok' when logged_items > 0", () => {
    seedUser();
    upsertDayLog(db, {
      id: "l1", user_id: "u1", date: TODAY,
      logged_items: 3, passive_credit_today: 0,
    });
    expect(runStreakCheckForUser(db, "u1", TODAY)).toBe("logged_ok");
    // Streak must NOT be mutated
    expect(getUserById(db, "u1")!.current_streak).toBe(7);
  });

  it("returns 'passive_credit_saved' and leaves streak untouched", () => {
    seedUser();
    upsertDayLog(db, {
      id: "l1", user_id: "u1", date: TODAY,
      logged_items: 0, passive_credit_today: 1,
    });
    expect(runStreakCheckForUser(db, "u1", TODAY)).toBe("passive_credit_saved");
    expect(getUserById(db, "u1")!.current_streak).toBe(7);
  });

  it("uses grace day, leaves streak, writes freeze log when balance > 0", () => {
    seedUser({ grace_days_balance: 2 });
    expect(runStreakCheckForUser(db, "u1", TODAY)).toBe("no_log_grace_used");
    const updated = getUserById(db, "u1")!;
    expect(updated.current_streak).toBe(7);       // streak untouched
    expect(updated.grace_days_balance).toBe(1);   // decremented
    const freeze = db
      .prepare("SELECT * FROM streak_freezes WHERE user_id = 'u1'")
      .get() as any;
    expect(freeze).toBeDefined();
    expect(freeze.date).toBe(TODAY);
    expect(freeze.reason).toBe("grace_day_used");
  });

  it("resets streak to 0 when no log and grace_days_balance is 0", () => {
    seedUser({ grace_days_balance: 0, current_streak: 10 });
    expect(runStreakCheckForUser(db, "u1", TODAY)).toBe("no_log_streak_reset");
    const updated = getUserById(db, "u1")!;
    expect(updated.current_streak).toBe(0);
    expect(updated.grace_days_balance).toBe(0);
  });

  it("no-log with null day_log row triggers grace logic (implicit empty day)", () => {
    // No day_log inserted = user simply didn't open the app today
    seedUser({ grace_days_balance: 1 });
    expect(runStreakCheckForUser(db, "u1", TODAY)).toBe("no_log_grace_used");
  });
});
```

- [ ] **Step 2: Run — verify fails**

```bash
npx jest tests/streaks/streak-checker.test.ts --no-coverage 2>&1 | head -10
```
Expected: `Cannot find module '../../src/streaks/streak-checker'`

- [ ] **Step 3: Write src/streaks/streak-checker.ts**

```typescript
import type Database from "better-sqlite3";
import {
  getUserById, getDayLog, updateUserStreak,
  insertStreakFreeze, randomUUID,
} from "../db/repositories";

export type StreakCheckOutcome =
  | "no_log_grace_used"
  | "no_log_streak_reset"
  | "passive_credit_saved"
  | "logged_ok"
  | "user_not_found";

/**
 * Runs the nightly streak check for a single user on `todayDate` (YYYY-MM-DD,
 * already converted to the user's local timezone by the cron layer).
 *
 * Mutation rules (in priority order):
 *   1. logged_items > 0           → no-op, return "logged_ok"
 *   2. passive_credit_today === 1 → no-op, return "passive_credit_saved"
 *   3. grace_days_balance > 0     → decrement grace, write freeze, return "no_log_grace_used"
 *   4. else                       → reset streak to 0, return "no_log_streak_reset"
 */
export function runStreakCheckForUser(
  db: Database.Database,
  userId: string,
  todayDate: string
): StreakCheckOutcome {
  const user = getUserById(db, userId);
  if (!user) return "user_not_found";

  const log = getDayLog(db, userId, todayDate);

  // Rule 1: active log entry present
  if (log && log.logged_items > 0) return "logged_ok";

  // Rule 2: passive mood credit was tapped today
  if (log && log.passive_credit_today === 1) return "passive_credit_saved";

  // Rule 3: grace balance available
  if (user.grace_days_balance > 0) {
    updateUserStreak(db, userId, user.current_streak, user.grace_days_balance - 1);
    insertStreakFreeze(db, {
      id: randomUUID(),
      user_id: userId,
      date: todayDate,
      reason: "grace_day_used",
    });
    return "no_log_grace_used";
  }

  // Rule 4: streak broken
  updateUserStreak(db, userId, 0, 0);
  return "no_log_streak_reset";
}
```

- [ ] **Step 4: Run — verify passes**

```bash
npx jest tests/streaks/streak-checker.test.ts --no-coverage
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/streaks/streak-checker.ts tests/streaks/streak-checker.test.ts
git commit -m "feat: pure streak check business logic with grace + freeze"
```

---

## Task 4: Midnight Cron Scheduler (Timezone-Aware)

**Files:**
- Create: `src/streaks/midnight-cron.ts`
- Create: `tests/streaks/midnight-cron.test.ts`

**Interfaces:**
- Consumes:
  - `runStreakCheckForUser` from `src/streaks/streak-checker.ts`
  - `getAllTimezones`, `getUsersInTimezone` from `src/db/repositories.ts`
  - `initDb` from `src/db/schema.ts`
  - `formatInTimeZone` from `date-fns-tz`
- Produces:
  - `scheduleAllTimezoneJobs(db): CronJob[]` — returns array of active cron job handles (for teardown in tests)
  - `runMidnightCheckForTimezone(db, timezone: string, nowUtc: Date): void` — testable inner function

The cron strategy: call `getAllTimezones(db)` once at startup. For each unique timezone, schedule a cron job that fires at `0 0 * * *` **in that timezone** (node-cron supports IANA timezone via `{ timezone }` option). When the job fires, call `runMidnightCheckForTimezone`.

`runMidnightCheckForTimezone` converts `nowUtc` to `YYYY-MM-DD` in the given timezone, loads all users in that timezone, runs `runStreakCheckForUser` for each.

- [ ] **Step 1: Write failing tests**

```typescript
// tests/streaks/midnight-cron.test.ts
import { initDb } from "../../src/db/schema";
import { upsertUser, getUserById } from "../../src/db/repositories";
import { runMidnightCheckForTimezone } from "../../src/streaks/midnight-cron";
import type Database from "better-sqlite3";

describe("runMidnightCheckForTimezone", () => {
  let db: Database.Database;

  beforeEach(() => { db = initDb(":memory:"); });
  afterEach(() => { db.close(); });

  it("resets streak for user with 0 grace when no log", () => {
    upsertUser(db, {
      id: "u1", timezone: "Asia/Kolkata",
      current_streak: 5, grace_days_balance: 0,
    });
    // 2026-06-29 00:00 IST = 2026-06-28 18:30 UTC
    const nowUtc = new Date("2026-06-28T18:30:00.000Z");
    runMidnightCheckForTimezone(db, "Asia/Kolkata", nowUtc);
    expect(getUserById(db, "u1")!.current_streak).toBe(0);
  });

  it("uses grace day for user with balance > 0 when no log", () => {
    upsertUser(db, {
      id: "u1", timezone: "America/New_York",
      current_streak: 3, grace_days_balance: 1,
    });
    // 2026-06-29 00:00 EDT = 2026-06-29T04:00:00.000Z
    const nowUtc = new Date("2026-06-29T04:00:00.000Z");
    runMidnightCheckForTimezone(db, "America/New_York", nowUtc);
    const updated = getUserById(db, "u1")!;
    expect(updated.current_streak).toBe(3);       // preserved
    expect(updated.grace_days_balance).toBe(0);   // decremented
  });

  it("skips users in different timezones", () => {
    upsertUser(db, {
      id: "u1", timezone: "Asia/Kolkata",
      current_streak: 5, grace_days_balance: 0,
    });
    upsertUser(db, {
      id: "u2", timezone: "America/New_York",
      current_streak: 5, grace_days_balance: 0,
    });
    const nowUtc = new Date("2026-06-28T18:30:00.000Z");
    runMidnightCheckForTimezone(db, "Asia/Kolkata", nowUtc);
    // u1 processed (IST midnight), u2 untouched
    expect(getUserById(db, "u1")!.current_streak).toBe(0);
    expect(getUserById(db, "u2")!.current_streak).toBe(5);
  });

  it("processes multiple users in the same timezone", () => {
    upsertUser(db, { id: "u1", timezone: "UTC", current_streak: 4, grace_days_balance: 0 });
    upsertUser(db, { id: "u2", timezone: "UTC", current_streak: 6, grace_days_balance: 0 });
    runMidnightCheckForTimezone(db, "UTC", new Date("2026-06-29T00:00:00.000Z"));
    expect(getUserById(db, "u1")!.current_streak).toBe(0);
    expect(getUserById(db, "u2")!.current_streak).toBe(0);
  });
});
```

- [ ] **Step 2: Run — verify fails**

```bash
npx jest tests/streaks/midnight-cron.test.ts --no-coverage 2>&1 | head -10
```
Expected: `Cannot find module '../../src/streaks/midnight-cron'`

- [ ] **Step 3: Write src/streaks/midnight-cron.ts**

```typescript
import cron, { ScheduledTask } from "node-cron";
import { formatInTimeZone } from "date-fns-tz";
import type Database from "better-sqlite3";
import { getAllTimezones, getUsersInTimezone } from "../db/repositories";
import { runStreakCheckForUser } from "./streak-checker";

/**
 * Runs streak checks for every user belonging to `timezone`.
 * `nowUtc` is injected for testability (defaults to `new Date()` in production).
 */
export function runMidnightCheckForTimezone(
  db: Database.Database,
  timezone: string,
  nowUtc: Date = new Date()
): void {
  // Convert UTC instant to local date string in this timezone
  const localDate = formatInTimeZone(nowUtc, timezone, "yyyy-MM-dd");
  const users = getUsersInTimezone(db, timezone);
  for (const user of users) {
    runStreakCheckForUser(db, user.id, localDate);
  }
}

/**
 * Schedules one cron job per unique timezone found in the database.
 * Each job fires at midnight local time for that timezone.
 * Returns the array of ScheduledTask handles so callers can stop them.
 */
export function scheduleAllTimezoneJobs(db: Database.Database): ScheduledTask[] {
  const timezones = getAllTimezones(db);
  return timezones.map((tz) =>
    cron.schedule(
      "0 0 * * *",
      () => runMidnightCheckForTimezone(db, tz),
      { timezone: tz }
    )
  );
}
```

- [ ] **Step 4: Run — verify passes**

```bash
npx jest tests/streaks/midnight-cron.test.ts --no-coverage
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/streaks/midnight-cron.ts tests/streaks/midnight-cron.test.ts
git commit -m "feat: timezone-aware midnight cron scheduler"
```

---

## Task 5: Passive Mood Log Endpoint

**Files:**
- Create: `src/mood/mood-types.ts`
- Create: `src/mood/mood-handler.ts`
- Create: `tests/mood/mood-handler.test.ts`

**Interfaces:**
- Consumes:
  - `upsertDayLog`, `getDayLog`, `getUserById`, `randomUUID`, `DayLogRow` from `src/db/repositories.ts`
  - `formatInTimeZone` from `date-fns-tz`
- Produces:
  - Express `Router` exported as `moodRouter` — mounts `POST /mood`
  - Request body: `{ user_id: string, mood: "EATING_CLEAN" | "SOCIAL_COMFORT" | "RESTING_RECOVERY" }`
  - Success 200: `{ ok: true, date: string, passive_credit_applied: true }`
  - Error 400: `{ ok: false, error: string }`
  - Error 404: `{ ok: false, error: "user not found" }`

Behaviour: Validates body with zod. Loads user to get timezone. Converts `new Date()` to local `YYYY-MM-DD`. Upserts a `day_logs` row with `passive_credit_today = 1`, preserving existing `logged_items` if a row already exists (default `logged_items = 0`).

- [ ] **Step 1: Write mood-types.ts**

```typescript
// src/mood/mood-types.ts
import { z } from "zod";

export const MoodValues = ["EATING_CLEAN", "SOCIAL_COMFORT", "RESTING_RECOVERY"] as const;
export type Mood = (typeof MoodValues)[number];

export const MoodRequestSchema = z.object({
  user_id: z.string().min(1),
  mood: z.enum(MoodValues),
});

export type MoodRequest = z.infer<typeof MoodRequestSchema>;
```

- [ ] **Step 2: Write failing tests**

```typescript
// tests/mood/mood-handler.test.ts
import express from "express";
import request from "supertest";
import { initDb } from "../../src/db/schema";
import { upsertUser, getDayLog } from "../../src/db/repositories";
import { moodRouter } from "../../src/mood/mood-handler";
import type Database from "better-sqlite3";

// supertest is a dev dependency: npm install -D supertest @types/supertest
function buildApp(db: Database.Database) {
  const app = express();
  app.use(express.json());
  app.use("/", moodRouter(db));
  return app;
}

describe("POST /mood", () => {
  let db: Database.Database;

  beforeEach(() => {
    db = initDb(":memory:");
    upsertUser(db, { id: "u1", timezone: "UTC", current_streak: 3, grace_days_balance: 1 });
  });
  afterEach(() => { db.close(); });

  it("returns 200 and passive_credit_applied: true for valid payload", async () => {
    const res = await request(buildApp(db))
      .post("/mood")
      .send({ user_id: "u1", mood: "EATING_CLEAN" });
    expect(res.status).toBe(200);
    expect(res.body.ok).toBe(true);
    expect(res.body.passive_credit_applied).toBe(true);
    expect(typeof res.body.date).toBe("string");
  });

  it("sets passive_credit_today = 1 in day_logs", async () => {
    await request(buildApp(db))
      .post("/mood")
      .send({ user_id: "u1", mood: "SOCIAL_COMFORT" });
    // date will be today in UTC
    const today = new Date().toISOString().slice(0, 10);
    const log = getDayLog(db, "u1", today);
    expect(log?.passive_credit_today).toBe(1);
  });

  it("preserves existing logged_items when upserting passive credit", async () => {
    const today = new Date().toISOString().slice(0, 10);
    // Pre-populate with 3 logged items
    db.prepare(`
      INSERT INTO day_logs (id, user_id, date, logged_items, passive_credit_today)
      VALUES ('l1', 'u1', '${today}', 3, 0)
    `).run();
    await request(buildApp(db))
      .post("/mood")
      .send({ user_id: "u1", mood: "RESTING_RECOVERY" });
    expect(getDayLog(db, "u1", today)!.logged_items).toBe(3);
  });

  it("returns 404 for unknown user_id", async () => {
    const res = await request(buildApp(db))
      .post("/mood")
      .send({ user_id: "ghost", mood: "EATING_CLEAN" });
    expect(res.status).toBe(404);
    expect(res.body.ok).toBe(false);
  });

  it("returns 400 for invalid mood value", async () => {
    const res = await request(buildApp(db))
      .post("/mood")
      .send({ user_id: "u1", mood: "JUNK_FOOD" });
    expect(res.status).toBe(400);
    expect(res.body.ok).toBe(false);
  });

  it("returns 400 when user_id missing", async () => {
    const res = await request(buildApp(db))
      .post("/mood")
      .send({ mood: "EATING_CLEAN" });
    expect(res.status).toBe(400);
  });
});
```

- [ ] **Step 3: Install supertest**

```bash
npm install -D supertest @types/supertest
```

- [ ] **Step 4: Run — verify fails**

```bash
npx jest tests/mood/mood-handler.test.ts --no-coverage 2>&1 | head -10
```
Expected: `Cannot find module '../../src/mood/mood-handler'`

- [ ] **Step 5: Write src/mood/mood-handler.ts**

```typescript
import { Router, Request, Response } from "express";
import { formatInTimeZone } from "date-fns-tz";
import type Database from "better-sqlite3";
import { MoodRequestSchema } from "./mood-types";
import {
  getUserById, getDayLog, upsertDayLog, randomUUID,
} from "../db/repositories";

/**
 * Returns an Express Router with POST /mood mounted.
 * Accepts { user_id, mood } and sets passive_credit_today = 1
 * on today's day_log for the user's local timezone, preventing
 * a streak reset at the next midnight check.
 */
export function moodRouter(db: Database.Database): Router {
  const router = Router();

  router.post("/mood", (req: Request, res: Response) => {
    const parsed = MoodRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ ok: false, error: parsed.error.issues[0]?.message ?? "invalid body" });
      return;
    }

    const { user_id, mood } = parsed.data;
    const user = getUserById(db, user_id);
    if (!user) {
      res.status(404).json({ ok: false, error: "user not found" });
      return;
    }

    const localDate = formatInTimeZone(new Date(), user.timezone, "yyyy-MM-dd");
    const existing = getDayLog(db, user_id, localDate);

    upsertDayLog(db, {
      id: existing?.id ?? randomUUID(),
      user_id,
      date: localDate,
      logged_items: existing?.logged_items ?? 0,
      passive_credit_today: 1,
    });

    res.status(200).json({ ok: true, date: localDate, passive_credit_applied: true, mood });
  });

  return router;
}
```

- [ ] **Step 6: Run — verify passes**

```bash
npx jest tests/mood/mood-handler.test.ts --no-coverage
```
Expected: `6 passed`

- [ ] **Step 7: Commit**

```bash
git add src/mood/mood-types.ts src/mood/mood-handler.ts tests/mood/mood-handler.test.ts
git commit -m "feat: passive mood log endpoint with passive_credit_today flag"
```

---

## Task 6: Habit Consistency Scorer

**Files:**
- Create: `src/scoring/consistency-scorer.ts`
- Create: `tests/scoring/consistency-scorer.test.ts`

**Interfaces:**
- Consumes: `DayLogRow` from `src/db/repositories.ts`
- Produces:
  - `calculateHabitConsistencyScore(logs: DayLogRow[], windowDates: string[]): ConsistencyResult`
  - `interface ConsistencyResult { score: number; active_days: number; total_window_days: number; effective_denominator: number; }`

Algorithm (scaled scoring):
1. `total_window_days = windowDates.length` (always 7 for a standard week)
2. Determine which dates had any activity: `logged_items > 0 OR passive_credit_today === 1`
3. `active_days` = count of dates with activity
4. Empty days (no `DayLogRow` **and** no passive credit) are dropped from the denominator
5. `effective_denominator = active_days + days_with_log_but_zero_items`
   — days that had *some* engagement (opened app, tapped mood) but logged 0 food still count in denominator
   — days with no row at all are excluded
6. If `effective_denominator === 0`: `score = 0.0`
7. Else: `score = active_days / effective_denominator` (capped at 1.0)

Concretely: a user who logged 4 days and tapped passive mood on 1 day but left 2 days completely absent:
- `active_days = 5` (4 food + 1 passive)
- days_with_log_but_zero_items = 0
- `effective_denominator = 5`
- `score = 5/5 = 1.0` ← only scored on windows they engaged with

A user who logged 4 days, had 1 day with 0 items (opened app, logged nothing), and 2 absent days:
- `active_days = 4`
- `days_with_log_but_zero_items = 1`
- `effective_denominator = 5`
- `score = 4/5 = 0.8`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/scoring/consistency-scorer.test.ts
import {
  calculateHabitConsistencyScore,
} from "../../src/scoring/consistency-scorer";
import type { DayLogRow } from "../../src/db/repositories";

const WEEK = [
  "2026-06-23", "2026-06-24", "2026-06-25",
  "2026-06-26", "2026-06-27", "2026-06-28", "2026-06-29",
];

function makeLog(date: string, logged_items: number, passive: 0 | 1 = 0): DayLogRow {
  return { id: date, user_id: "u1", date, logged_items, passive_credit_today: passive };
}

describe("calculateHabitConsistencyScore", () => {
  it("returns 0 with no logs and no engagement", () => {
    const result = calculateHabitConsistencyScore([], WEEK);
    expect(result.score).toBe(0);
    expect(result.active_days).toBe(0);
    expect(result.effective_denominator).toBe(0);
  });

  it("scores 1.0 when all 7 days have food logged", () => {
    const logs = WEEK.map((d) => makeLog(d, 3));
    const result = calculateHabitConsistencyScore(logs, WEEK);
    expect(result.score).toBeCloseTo(1.0);
    expect(result.active_days).toBe(7);
    expect(result.effective_denominator).toBe(7);
  });

  it("drops absent days from denominator: 4 logged, 3 absent → score 1.0", () => {
    const logs = WEEK.slice(0, 4).map((d) => makeLog(d, 2));
    const result = calculateHabitConsistencyScore(logs, WEEK);
    expect(result.score).toBeCloseTo(1.0);
    expect(result.active_days).toBe(4);
    expect(result.effective_denominator).toBe(4);
    expect(result.total_window_days).toBe(7);
  });

  it("counts passive credit as active day", () => {
    const logs = [
      makeLog("2026-06-23", 3),
      makeLog("2026-06-24", 0, 1),  // passive only, no food
    ];
    const result = calculateHabitConsistencyScore(logs, WEEK);
    expect(result.score).toBeCloseTo(1.0);
    expect(result.active_days).toBe(2);
    expect(result.effective_denominator).toBe(2);
  });

  it("zero-item logged day counts in denominator but not as active", () => {
    const logs = [
      makeLog("2026-06-23", 3),
      makeLog("2026-06-24", 0, 0),  // opened app, logged nothing, no passive
    ];
    const result = calculateHabitConsistencyScore(logs, WEEK);
    expect(result.score).toBeCloseTo(0.5);   // 1 active / 2 effective
    expect(result.active_days).toBe(1);
    expect(result.effective_denominator).toBe(2);
  });

  it("mixed: 4 food, 1 passive, 1 zero-item open, 1 absent → score = 5/6", () => {
    const logs = [
      makeLog("2026-06-23", 2),
      makeLog("2026-06-24", 3),
      makeLog("2026-06-25", 1),
      makeLog("2026-06-26", 2),
      makeLog("2026-06-27", 0, 1), // passive
      makeLog("2026-06-28", 0, 0), // opened, nothing
      // 2026-06-29 absent — excluded from denominator
    ];
    const result = calculateHabitConsistencyScore(logs, WEEK);
    expect(result.score).toBeCloseTo(5 / 6, 5);
    expect(result.active_days).toBe(5);
    expect(result.effective_denominator).toBe(6);
  });

  it("score is capped at 1.0", () => {
    const logs = WEEK.map((d) => makeLog(d, 5, 1)); // both food and passive
    const result = calculateHabitConsistencyScore(logs, WEEK);
    expect(result.score).toBeLessThanOrEqual(1.0);
  });
});
```

- [ ] **Step 2: Run — verify fails**

```bash
npx jest tests/scoring/consistency-scorer.test.ts --no-coverage 2>&1 | head -10
```
Expected: `Cannot find module '../../src/scoring/consistency-scorer'`

- [ ] **Step 3: Write src/scoring/consistency-scorer.ts**

```typescript
import type { DayLogRow } from "../db/repositories";

export interface ConsistencyResult {
  /** Scaled score 0.0–1.0 (only scored on engaged windows) */
  score: number;
  /** Days with logged_items > 0 OR passive_credit_today === 1 */
  active_days: number;
  /** Length of the input windowDates array (always 7 for a full week) */
  total_window_days: number;
  /**
   * Denominator after dropping fully-absent days.
   * Includes active days + days with a row but 0 items and no passive credit
   * (user opened the app but logged nothing).
   */
  effective_denominator: number;
}

/**
 * Calculates a scaled habit consistency score for a single user over one week.
 *
 * Fully absent days (no DayLogRow at all for that date) are excluded from the
 * denominator so the user is only judged on windows they actually engaged with.
 * This keeps the score motivating rather than punishing for life events.
 *
 * @param logs        DayLogRow records for the user in this window (may be a subset of windowDates)
 * @param windowDates Array of YYYY-MM-DD strings representing the full window (typically 7)
 */
export function calculateHabitConsistencyScore(
  logs: DayLogRow[],
  windowDates: string[]
): ConsistencyResult {
  const total_window_days = windowDates.length;

  // Index logs by date for O(1) lookup
  const logByDate = new Map<string, DayLogRow>();
  for (const log of logs) {
    logByDate.set(log.date, log);
  }

  let active_days = 0;
  let effective_denominator = 0;

  for (const date of windowDates) {
    const log = logByDate.get(date);

    if (!log) {
      // Fully absent — drop from denominator entirely
      continue;
    }

    const isActive = log.logged_items > 0 || log.passive_credit_today === 1;

    // Any row (even 0-item open) counts in denominator
    effective_denominator += 1;

    if (isActive) {
      active_days += 1;
    }
  }

  const score =
    effective_denominator === 0
      ? 0.0
      : Math.min(1.0, active_days / effective_denominator);

  return { score, active_days, total_window_days, effective_denominator };
}
```

- [ ] **Step 4: Run — verify passes**

```bash
npx jest tests/scoring/consistency-scorer.test.ts --no-coverage
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/scoring/consistency-scorer.ts tests/scoring/consistency-scorer.test.ts
git commit -m "feat: scaled habit consistency scorer drops absent days from denominator"
```

---

## Task 7: Wire Express App + Server Entry Point

**Files:**
- Create: `src/app.ts`
- Create: `src/server.ts`
- Modify: (no existing files changed)

**Interfaces:**
- Consumes: `moodRouter` from `src/mood/mood-handler.ts`, `scheduleAllTimezoneJobs` from `src/streaks/midnight-cron.ts`, `initDb` from `src/db/schema.ts`
- Produces: `app` (Express app, for integration tests), `server.ts` (entry point calling `app.listen`)

- [ ] **Step 1: Write src/app.ts**

```typescript
import express from "express";
import type Database from "better-sqlite3";
import { moodRouter } from "./mood/mood-handler";

export function buildApp(db: Database.Database) {
  const app = express();
  app.use(express.json());
  app.use("/", moodRouter(db));

  app.get("/health", (_req, res) => {
    res.json({ ok: true });
  });

  return app;
}
```

- [ ] **Step 2: Write src/server.ts**

```typescript
import { initDb } from "./db/schema";
import { buildApp } from "./app";
import { scheduleAllTimezoneJobs } from "./streaks/midnight-cron";

const PORT = parseInt(process.env.PORT ?? "3000", 10);
const db   = initDb();
const app  = buildApp(db);

scheduleAllTimezoneJobs(db);

app.listen(PORT, () => {
  console.log(`Streak service running on :${PORT}`);
});
```

- [ ] **Step 3: Add start script to package.json**

In `package.json` scripts section add:
```json
"start": "ts-node src/server.ts",
"build": "tsc",
"test": "jest"
```

- [ ] **Step 4: Verify full test suite passes**

```bash
npx jest --no-coverage
```
Expected: all tests pass (22+ tests across 5 suites)

- [ ] **Step 5: Commit**

```bash
git add src/app.ts src/server.ts package.json
git commit -m "feat: wire express app and server entry point"
```

---

## Self-Review Checklist

**Spec coverage:**
| Requirement | Task |
|---|---|
| Midnight check per timezone | Task 4 (cron), Task 3 (checker logic) |
| grace_days_balance decrement | Task 3 — rule 3 |
| streak_freezes log entry | Task 3 — `insertStreakFreeze` call |
| current_streak reset to 0 | Task 3 — rule 4 |
| Passive mood endpoint (EATING_CLEAN / SOCIAL_COMFORT / RESTING_RECOVERY) | Task 5 |
| Passive credit prevents streak reset | Task 3 — rule 2; Task 5 sets the flag |
| HabitConsistencyScore | Task 6 |
| Drop empty days from denominator | Task 6 — absent days skipped |
| Score on active windows only | Task 6 — `effective_denominator` |

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency:** `DayLogRow`, `UserRow`, `StreakFreezeRow` defined once in `repositories.ts`, imported everywhere else. `runStreakCheckForUser` signature matches usage in `midnight-cron.ts`. `calculateHabitConsistencyScore` signature matches test imports.
