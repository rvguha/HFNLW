// Minimal browser client for the server-sent-event search endpoint.
const $ = id => document.getElementById(id);
const pageParams = new URLSearchParams(location.search);
const askEndpoint = new URL(pageParams.get("ask") || "/ask", location.href);
const history = [];

// The catalog scope the tabs select. Sent as `site`; the server's aggregate
// scopes (sources.KINDS) filter the manifest section a collection sits in.
let scope = "models";

const sampleQueries = {
  models: [
    "A small model for biomedical named-entity recognition",
    "Models documented as trained on protein or genomic sequences",
    "Multilingual embedding models evaluated on non-English retrieval",
    "Speech recognition for low-resource African languages",
    "Image models with a clearly stated commercially usable license",
    "What can run locally on a laptop with 16 GB of memory?",
    "A legal domain model, not a chat model that mentions legal disclaimers",
    "Models fine-tuned on financial filings or market data",
    "Document question answering over scanned invoices and forms",
    "Models supporting Hindi, Tamil, or Telugu",
    "The original base model rather than a repackaged quantization",
    "Models that document their limitations and known risks"
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
  $("clear").hidden = false;
  hideUsage();

  const turn = startTurn(query);

  try {
    const args = { query, site: scope, previous_queries: history.slice(-5) };
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
          finalCount += 1;
        }
      } else if (type === "nlws" && content?.answer) {
        turn.setAnswer(content.answer);
      } else if ((type === "intermediate_message" || type === "error") && content) {
        turn.addNotice(content);
      } else if (type === "usage" && content) {
        renderUsage(content);
      } else if (type === "decontextualized_query") {
        turn.setInterpreted(content);
      } else if (type === "end-nlweb-response") {
        turn.setStatus(`${finalCount} result${finalCount === 1 ? "" : "s"}`);
      }
    });
    history.push(query);
  } catch (error) {
    turn.addNotice(error.message);
    turn.setStatus("Search failed");
  } finally {
    $("submit").disabled = false;
    $("query").focus();
  }
});

$("clear").addEventListener("click", () => {
  history.length = 0;
  $("thread").replaceChildren();
  $("clear").hidden = true;
  $("samples").open = true;
  hideUsage();
  $("query").focus();
});

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
