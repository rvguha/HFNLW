// Minimal browser client for the server-sent-event search endpoint.
const $ = id => document.getElementById(id);
const pageParams = new URLSearchParams(location.search);
const askEndpoint = new URL(pageParams.get("ask") || "/ask", location.href);

// The catalog scope the tabs select. Sent as `site`; the server's aggregate
// scopes (sources.KINDS) filter the manifest section a collection sits in.
let scope = "models";

import { Threadstore } from "/threadstore.js?v=5";

// A conversation is the unit now, not a page load. `store` persists it,
// `thread` is the one being added to, and `turns` mirrors it in memory so a
// follow-up can be sent without inflating anything from disk.
let store = null;
let thread = null;
let turns = [];
let corpus = null;

const sampleQueries = {
  models: [
    "I've got a MacBook with 16GB and no cloud budget for this project. What can I realistically run locally without it swapping constantly?",
    "I need a model that genuinely knows molecular biology \u2014 trained on it, not one that just happens to cite a biology benchmark in a results table. That distinction matters for what we're doing.",
    "Our RAG pipeline retrieves 50 candidates and the ordering is poor \u2014 the right answer is often at rank 30. I gather a reranker is what I want here. What should I use?",
    "We're transcribing Vietnamese customer service calls. Note I want speech going to text, not the other way round \u2014 I keep finding TTS models when I search. What handles Vietnamese ASR?",
    "A colleague told me to use whisper large v3 for our transcription work but I can't remember where it lives. Can you point me at it?",
    "Our legal team is nervous about what we ship in a commercial product. For image models specifically \u2014 generation or classification \u2014 which ones have a licence that's clearly stated and unambiguously fine for commercial use? I don't want anything where the terms are vague.",
    "We feed entire contracts into the model, sometimes 100 pages. Short context windows mean chunking and losing cross-references. What has a genuinely long context window?",
    "I'm working with a team recording oral histories in Swahili, Yoruba and Hausa. Most ASR I've tried is hopeless on these. Is there anything that genuinely handles low-resource African languages?"
  ]
};

async function streamAsk(args, onEvent) {
  const response = await fetch(askEndpoint, {
    method: "POST",
    headers: { accept: "text/event-stream", "content-type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!response.ok) {
    let message = `Search HTTP ${response.status}`;
    try { message = (await response.json()).error || message; } catch { /* use status */ }
    throw new Error(message);
  }
  if (!response.body) throw new Error("Search response cannot be streamed");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function consume(final = false) {
    buffer = buffer.replaceAll("\r\n", "\n");
    const frames = buffer.split("\n\n");
    buffer = final ? "" : frames.pop();
    for (const frame of frames) {
      const data = frame.split("\n").filter(line => line.startsWith("data:"))
        .map(line => line.slice(5).trim()).join("\n");
      if (!data) continue;
      const message = JSON.parse(data);
      onEvent(message.message_type, message.content);
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    consume(done);
    if (done) break;
  }
}

function text(tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

function safeUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url : null;
  } catch {
    return null;
  }
}

function render(item) {
  const li = text("li", "", "result");
  const head = text("div", "", "head");
  const url = safeUrl(item.url);
  const title = text(url ? "a" : "span", item.name || "Untitled");
  if (url) {
    title.href = url;
    title.target = "_blank";
    title.rel = "noopener noreferrer";
  }
  head.append(title);
  li.append(head);
  if (item.site) li.append(text("div", item.site, "meta"));
  if (item.description) li.append(text("p", item.description, "description"));
  return li;
}

// One turn of the conversation: the question as asked, how the server
// understood it, and the results for it. Turns accumulate, because a follow-up
// only makes sense next to what it follows.
function startTurn(question) {
  const turn = text("section", "", "turn");
  const asked = text("div", question, "asked");
  const status = text("div", "Searching\u2026", "turn-status");
  const results = document.createElement("ul");
  results.className = "results";
  turn.append(asked, status, results);
  $("thread").append(turn);
  // "start", not "nearest": the turn is empty at this point, so a minimal
  // scroll is a no-op and the results then push the question below the fold.
  // Putting the question at the top of the viewport is also what a thread
  // wants -- you read down from what you just asked.
  turn.scrollIntoView({ behavior: "smooth", block: "start" });
  return {
    node: turn,
    results,
    setStatus: value => { status.textContent = value; },
    // The rewrite is what makes a follow-up work or fail, so it stays on screen
    // rather than flashing past in a status line.
    setInterpreted: value => {
      if (!value || value === question) return;
      const note = text("div", `interpreted as: ${value}`, "interpreted");
      asked.after(note);
    },
    addNotice: value => {
      let notice = turn.querySelector(".notice");
      if (!notice) {
        notice = text("div", "", "notice");
        status.after(notice);
      }
      notice.textContent += `${notice.textContent ? " " : ""}${value}`;
    },
    setAnswer: value => {
      const answer = text("div", value, "answer");
      status.after(answer);
    },
  };
}

$("form").addEventListener("submit", async event => {
  event.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;

  $("submit").disabled = true;
  $("query").value = "";              // ready for the follow-up
  $("samples").open = false;
  $("intro").hidden = true;
  hideUsage();

  const turn = startTurn(query);
  // Built up as the stream arrives; stored whole when the turn completes.
  const record = { question: query, askedAt: Date.now(), interpretedAs: null,
                   mode: $("mode").value, results: [], answer: null,
                   notices: [], usage: null };
  // What came before, captured before this turn joins the list -- a question is
  // not its own antecedent.
  const previous = turns.slice(-5).map(t => t.question);
  // Joined now rather than on completion. A follow-up asked while the previous
  // answer is still streaming would otherwise be sent with no context, and the
  // server would decontextualize it against nothing -- silently, because a
  // query with no antecedent is a legitimate query.
  turns.push(record);

  try {
    const args = { query, site: scope, mode: $("mode").value, previous_queries: previous };
    const provisional = new Set();
    let finalStarted = false;
    let finalCount = 0;

    await streamAsk(args, (type, content) => {
      if (type === "candidate" && !finalStarted) {
        for (const item of content || []) {
          const key = item.url || `${item.site}:${item.name}`;
          if (provisional.has(key)) continue;
          provisional.add(key);
          const node = render(item);
          node.classList.add("provisional");
          turn.results.append(node);
        }
        turn.setStatus(`Searching\u2026 ${provisional.size} possible result${provisional.size === 1 ? "" : "s"}`);
      } else if (type === "result") {
        if (!finalStarted) {
          finalStarted = true;
          turn.results.replaceChildren();
        }
        for (const item of content || []) {
          turn.results.append(render(item));
          record.results.push(item);
          finalCount += 1;
        }
      } else if (type === "nlws" && content?.answer) {
        turn.setAnswer(content.answer);
        record.answer = content.answer;
      } else if ((type === "intermediate_message" || type === "error") && content) {
        turn.addNotice(content);
        record.notices.push(content);
      } else if (type === "usage" && content) {
        renderUsage(content);
        record.usage = content;
      } else if (type === "decontextualized_query") {
        turn.setInterpreted(content);
        record.interpretedAs = content;
      } else if (type === "end-nlweb-response") {
        turn.setStatus(`${finalCount} result${finalCount === 1 ? "" : "s"}`);
      }
    });
    await remember(record);
  } catch (error) {
    turn.addNotice(error.message);
    turn.setStatus("Search failed");
  } finally {
    $("submit").disabled = false;
    $("query").focus();
  }
});

// Persist the completed turn, starting a thread on the first one so an
// abandoned empty conversation never appears in the list.
async function remember(record) {
  if (!store) return;
  if (!thread) thread = await store.startThread({ scope, corpus });
  await store.appendTurn(thread, record);
  $("chat-title").textContent = thread.title;
  await refreshConversations();
}

function newChat() {
  thread = null;
  turns = [];
  $("thread").replaceChildren();
  $("chat-title").textContent = "New chat";
  $("intro").hidden = false;
  $("samples").open = true;
  hideUsage();
  markActive(null);
  $("query").focus();
}

$("new-chat").addEventListener("click", newChat);

function renderUsage(usage) {
  const rows = $("usage-rows");
  rows.replaceChildren();
  const tokens = Number(usage.total_tokens || 0);
  const cost = Number(usage.cost || 0);
  const missing = Number(usage.unpriced_calls || 0);
  const summary = `${tokens.toLocaleString()} tokens \u00b7 $${cost.toFixed(6)} USD` +
    (missing ? ` \u00b7 ${missing} unpriced calls` : "");
  $("usage-total").textContent = summary;
  $("usage-open-total").textContent = `\u00b7 ${summary}`;
  for (const item of usage.models || []) {
    const tr = document.createElement("tr");
    const phase = Object.entries(item.phases || {}).map(([name, count]) => `${name} ${count}`).join(" \u00b7 ");
    const model = text("td", "");
    model.append(text("div", item.model || "unknown"), text("small", phase));
    tr.append(model);
    for (const value of [item.calls, item.prompt_tokens, item.completion_tokens, item.total_tokens]) {
      tr.append(text("td", Number(value || 0).toLocaleString()));
    }
    tr.append(text("td", `$${Number(item.cost || 0).toFixed(6)}${item.unpriced_calls ? "*" : ""}`));
    rows.append(tr);
  }
  $("usage-open").hidden = false;
}

function hideUsage() {
  $("usage-open").hidden = true;
  if ($("usage-dialog").open) $("usage-dialog").close();
}

function showSamples() {
  const container = $("sample-queries");
  container.replaceChildren();
  for (const query of sampleQueries[scope] || []) {
    const button = text("button", query, "sample-query");
    button.type = "button";
    button.addEventListener("click", () => {
      $("query").value = query;
      $("form").requestSubmit();
    });
    container.append(button);
  }
}

function selectScope(next) {
  scope = next;
  for (const tab of document.querySelectorAll(".tab[data-scope]")) {
    tab.setAttribute("aria-selected", String(tab.dataset.scope === next));
  }
  $("query").placeholder = "Ask what a model is documented to do\u2026";
  showSamples();
}

for (const tab of document.querySelectorAll(".tab[data-scope]")) {
  tab.addEventListener("click", () => selectScope(tab.dataset.scope));
}
selectScope(scope);

const usageDialog = $("usage-dialog");
$("usage-open").addEventListener("click", () => usageDialog.showModal());
$("usage-close").addEventListener("click", () => usageDialog.close());
usageDialog.addEventListener("click", event => {
  if (event.target === usageDialog) usageDialog.close();
});


// ---------------------------------------------------------------- sidebar

function when(ms) {
  const days = Math.floor((Date.now() - ms) / 86400000);
  if (days === 0) return new Date(ms).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return new Date(ms).toLocaleDateString([], { month: "short", day: "numeric" });
}

function markActive(id) {
  for (const node of document.querySelectorAll(".conversation")) {
    node.classList.toggle("active", node.dataset.id === id);
  }
}

async function refreshConversations() {
  if (!store?.available) return;
  const all = await store.threads();
  // Matches anything asked, rewritten, or returned in the thread -- not just
  // its title, which is only the first question.
  const needle = $("search").value.trim();
  const shown = needle ? all.filter(t => Threadstore.matches(t, needle)) : all;
  const list = $("conversations");
  list.replaceChildren();

  if (!shown.length) {
    list.append(text("p", needle ? "No conversations match." : "No conversations yet.", "empty"));
  }
  for (const item of shown) {
    const row = text("div", "", "conversation");
    row.dataset.id = item.id;
    row.setAttribute("role", "listitem");
    const open = text("button", item.title || "Untitled", "conversation-open");
    open.type = "button";
    open.addEventListener("click", () => openThread(item.id));
    const meta = text("div", `${item.turnCount} turn${item.turnCount === 1 ? "" : "s"} · ${when(item.updatedAt)}`, "conversation-meta");
    const remove = text("button", "×", "conversation-delete");
    remove.type = "button";
    remove.title = "Delete conversation";
    remove.addEventListener("click", async event => {
      event.stopPropagation();
      await store.remove(item.id);
      if (thread?.id === item.id) newChat();
      await refreshConversations();
    });
    row.append(open, meta, remove);
    list.append(row);
  }
  markActive(thread?.id ?? null);

  const usage = await store.usage();
  $("storage-usage").textContent = usage.turns
    ? `${usage.turns} turns · ${(usage.bytes / 1024).toFixed(0)} KB`
    : "";
}

// Reopening shows exactly what was shown, from the stored turn -- nothing is
// re-fetched, so nothing can come back different.
async function openThread(id) {
  const all = await store.threads();
  const found = all.find(t => t.id === id);
  if (!found) return;
  thread = found;
  turns = await store.turns(id);
  $("thread").replaceChildren();
  $("chat-title").textContent = found.title;
  $("intro").hidden = true;
  hideUsage();
  for (const record of turns) {
    const turn = startTurn(record.question);
    turn.setInterpreted(record.interpretedAs);
    for (const item of record.results || []) turn.results.append(render(item));
    if (record.answer) turn.setAnswer(record.answer);
    for (const notice of record.notices || []) turn.addNotice(notice);
    turn.setStatus(`${(record.results || []).length} result${(record.results || []).length === 1 ? "" : "s"}`);
  }
  const last = turns[turns.length - 1];
  if (last?.usage) renderUsage(last.usage);
  markActive(id);
  $("query").focus();
}

$("search").addEventListener("input", refreshConversations);

$("clear-all").addEventListener("click", async () => {
  if (!store?.available) return;
  if (!confirm("Delete every saved conversation? This cannot be undone.")) return;
  await store.clear();
  newChat();
  await refreshConversations();
});

$("toggle-sidebar").addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  $("toggle-sidebar").setAttribute("aria-expanded", String(!collapsed));
});

// Enter sends, Shift+Enter makes a newline -- the composer is a textarea so a
// long prompt can be written and read before it is sent.
$("query").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("form").requestSubmit();
  }
});

// Grow with the prompt, up to a point, so a paragraph-long question is visible
// as it is written rather than scrolling inside one line.
$("query").addEventListener("input", () => {
  const box = $("query");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 200)}px`;
});

async function boot() {
  try {
    const health = await fetch("/health").then(r => r.json());
    corpus = health.corpus || null;
    if (corpus?.items) {
      $("scope-chip").textContent = `${corpus.items.toLocaleString()} models`;
      $("scope-chip").title = corpus.snapshot_id
        ? `Corpus ${corpus.snapshot_id}, built ${corpus.built_at}`
        : "";
    }
  } catch { /* the chip is decoration; a failed probe must not stop the app */ }

  store = await Threadstore.open();
  if (store.available) await store.backfill();
  if (!store.available) {
    $("conversations").append(
      text("p", "This browser is not storing conversations, so they will not survive a reload.", "empty"));
    return;
  }
  await refreshConversations();
}

boot();
