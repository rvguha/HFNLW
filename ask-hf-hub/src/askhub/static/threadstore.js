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
      createdAt: now, updatedAt: now, turnCount: 0, schema: SCHEMA,
    };
    if (this.db) await idb(this.#tx(THREADS, "readwrite").put(thread));
    return thread;
  }

  async appendTurn(thread, turn) {
    thread.turnCount += 1;
    thread.updatedAt = Date.now();
    if (!thread.title) thread.title = title(turn.question);
    if (!this.db) return thread;
    const record = {
      threadId: thread.id,
      seq: thread.turnCount,
      askedAt: turn.askedAt || Date.now(),
      blob: await gzip(turn),
    };
    await idb(this.#tx(TURNS, "readwrite").put(record));
    await idb(this.#tx(THREADS, "readwrite").put(thread));
    return thread;
  }

  async threads() {
    if (!this.db) return [];
    const all = await idb(this.#tx(THREADS, "readonly").getAll());
    return all.filter(t => t.turnCount > 0).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async turns(threadId) {
    if (!this.db) return [];
    const range = IDBKeyRange.bound([threadId, 0], [threadId, Infinity]);
    const rows = await idb(this.#tx(TURNS, "readonly").getAll(range));
    return Promise.all(rows.sort((a, b) => a.seq - b.seq).map(async row => ({
      seq: row.seq, askedAt: row.askedAt, ...(await gunzip(row.blob)),
    })));
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
