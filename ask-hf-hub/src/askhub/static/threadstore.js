// Threadstore: conversation threads that survive a reload.
//
// The interface used to keep its conversation in `const history = []`, which
// works and vanishes on reload. The server cannot help — /ask is stateless by
// design — so the browser owns the thread.
//
// Nothing is stored by reference. The corpus is rebuilt, re-extracted and
// re-emitted constantly; a reference resolves to whatever the corpus says now,
// which is precisely not what you were shown. Every turn is stored whole,
// schema_object and all, so reopening a thread shows what you saw whatever has
// happened to the repository since. That is affordable because JSON-LD
// compresses about tenfold — the repeated @context, @type and property names
// are nearly free — so a turn costs roughly 16 KB gzipped rather than 156 KB.
//
// Compression is per turn, not per thread, so appending never rewrites history.

const DB_NAME = "ask-hf-hub";
const DB_VERSION = 1;
const THREADS = "threads";
const TURNS = "turns";
const SCHEMA = 1;

function idb(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function gzip(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  if (typeof CompressionStream !== "function") return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function gunzip(bytes) {
  if (typeof DecompressionStream !== "function") {
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

// Time-sortable so the conversation list needs no secondary index, and random
// enough that two tabs starting a thread in the same millisecond do not collide.
function threadId() {
  const stamp = Date.now().toString(36).padStart(9, "0");
  const noise = Math.random().toString(36).slice(2, 8);
  return `t_${stamp}${noise}`;
}

// A lowercase digest of everything worth finding a conversation by: what was
// asked, how it was rewritten, and which models came back. Kept on the thread
// record rather than inside the compressed turns, so searching never has to
// inflate anything -- the alternative gunzips the whole history per keystroke.
//
// Capped, because a long conversation should not grow an unbounded index entry.
// The cap costs nothing in practice: threads are found by how they started and
// what they surfaced, and both are near the front.
const DIGEST_LIMIT = 4000;

function digest(existing, turn) {
  const parts = [
    existing || "",
    turn.question || "",
    turn.interpretedAs || "",
    (turn.results || []).map(r => r.name || "").join(" "),
  ];
  return parts.join(" ").replace(/\s+/g, " ").trim().toLowerCase().slice(0, DIGEST_LIMIT);
}

function title(question) {
  const clean = question.replace(/\s+/g, " ").trim();
  return clean.length > 72 ? `${clean.slice(0, 71)}…` : clean;
}

export class Threadstore {
  constructor(db) {
    this.db = db;
  }

  static async open() {
    if (!globalThis.indexedDB) return new Threadstore(null);
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(THREADS)) {
        db.createObjectStore(THREADS, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(TURNS)) {
        // Keyed on [threadId, seq] so "this thread's turns, in order" is a
        // range scan. Only the key fields live outside the compressed blob.
        db.createObjectStore(TURNS, { keyPath: ["threadId", "seq"] });
      }
    };
    try {
      return new Threadstore(await idb(request));
    } catch {
      // Private windows and blocked site data both land here. The interface
      // must still work; it just forgets.
      return new Threadstore(null);
    }
  }

  get available() {
    return this.db !== null;
  }

  #tx(store, mode) {
    return this.db.transaction(store, mode).objectStore(store);
  }

  async startThread({ scope, corpus }) {
    const now = Date.now();
    const thread = {
      id: threadId(), title: "", scope, corpus: corpus || null,
      createdAt: now, updatedAt: now, turnCount: 0, search: "", schema: SCHEMA,
    };
    if (this.db) await idb(this.#tx(THREADS, "readwrite").put(thread));
    return thread;
  }

  async appendTurn(thread, turn) {
    // Next sequence is one past the highest that exists, not turnCount + 1.
    // Deleting a middle turn leaves a gap, so a count is not a high-water mark:
    // with turns 1 and 3 surviving, turnCount is 2 and the next turn would land
    // on 3 and silently overwrite it.
    const seq = (await this.#lastSeq(thread.id)) + 1;
    thread.turnCount += 1;
    thread.updatedAt = Date.now();
    thread.search = digest(thread.search, turn);
    if (!thread.title) thread.title = title(turn.question);
    if (!this.db) return thread;
    const record = {
      threadId: thread.id,
      seq,
      askedAt: turn.askedAt || Date.now(),
      blob: await gzip(turn),
    };
    await idb(this.#tx(TURNS, "readwrite").put(record));
    await idb(this.#tx(THREADS, "readwrite").put(thread));
    thread.lastSeq = seq;
    return thread;
  }

  /** Highest sequence stored for a thread, or 0 when it has none. */
  async #lastSeq(threadId) {
    if (!this.db) return 0;
    const range = IDBKeyRange.bound([threadId, 0], [threadId, Infinity]);
    const keys = await idb(this.#tx(TURNS, "readonly").getAllKeys(range));
    return keys.reduce((high, key) => Math.max(high, key[1]), 0);
  }

  async threads() {
    if (!this.db) return [];
    const all = await idb(this.#tx(THREADS, "readonly").getAll());
    return all.filter(t => t.turnCount > 0).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  /** Does this thread match a search, by anything in it rather than its title? */
  static matches(thread, needle) {
    if (!needle) return true;
    const term = needle.toLowerCase();
    return (thread.title || "").toLowerCase().includes(term)
      || (thread.search || "").includes(term);
  }

  /** Build digests for threads stored before this index existed. */
  async backfill() {
    if (!this.db) return 0;
    const all = await idb(this.#tx(THREADS, "readonly").getAll());
    const stale = all.filter(t => t.turnCount > 0 && !t.search);
    for (const thread of stale) {
      let text = "";
      for (const turn of await this.turns(thread.id)) text = digest(text, turn);
      thread.search = text;
      await idb(this.#tx(THREADS, "readwrite").put(thread));
    }
    return stale.length;
  }

  async turns(threadId) {
    if (!this.db) return [];
    const range = IDBKeyRange.bound([threadId, 0], [threadId, Infinity]);
    const rows = await idb(this.#tx(TURNS, "readonly").getAll(range));
    return Promise.all(rows.sort((a, b) => a.seq - b.seq).map(async row => ({
      seq: row.seq, askedAt: row.askedAt, ...(await gunzip(row.blob)),
    })));
  }

  /** Drop one turn, leaving the rest of the conversation intact.
   *
   * Sequence numbers keep their gaps rather than being renumbered: they are
   * ordering keys, not positions, and rewriting them would mean rewriting every
   * later turn's key to fix something nothing reads. The digest and title are
   * rebuilt from what survives, because both were derived from the turn that
   * just left -- a thread whose first question is deleted should not keep being
   * named after it.
   *
   * Deleting the last turn deletes the thread: an empty conversation is not a
   * conversation, and `threads()` would hide it anyway, leaving an orphan.
   */
  async removeTurn(threadId, seq) {
    if (!this.db) return null;
    await idb(this.#tx(TURNS, "readwrite").delete([threadId, seq]));
    const thread = await idb(this.#tx(THREADS, "readonly").get(threadId));
    if (!thread) return null;

    const remaining = await this.turns(threadId);
    if (!remaining.length) {
      await this.remove(threadId);
      return null;
    }
    thread.turnCount = remaining.length;
    thread.title = title(remaining[0].question);
    thread.search = remaining.reduce((text, turn) => digest(text, turn), "");
    thread.updatedAt = Date.now();
    await idb(this.#tx(THREADS, "readwrite").put(thread));
    return thread;
  }

  async remove(threadId) {
    if (!this.db) return;
    await idb(this.#tx(THREADS, "readwrite").delete(threadId));
    const range = IDBKeyRange.bound([threadId, 0], [threadId, Infinity]);
    await idb(this.#tx(TURNS, "readwrite").delete(range));
  }

  async clear() {
    if (!this.db) return;
    await idb(this.#tx(TURNS, "readwrite").clear());
    await idb(this.#tx(THREADS, "readwrite").clear());
  }

  // Rough on-disk size, for a UI that wants to say what it is holding.
  async usage() {
    if (!this.db) return { threads: 0, turns: 0, bytes: 0 };
    const rows = await idb(this.#tx(TURNS, "readonly").getAll());
    const threads = await idb(this.#tx(THREADS, "readonly").getAll());
    return {
      threads: threads.length,
      turns: rows.length,
      bytes: rows.reduce((total, row) => total + (row.blob?.byteLength || 0), 0),
    };
  }
}
