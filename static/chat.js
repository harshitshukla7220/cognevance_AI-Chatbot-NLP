const log = document.getElementById("log"), form = document.getElementById("form"),
      input = document.getElementById("input"), send = form.querySelector("button");

function add(text, who, meta) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  if (meta) { const m = document.createElement("div"); m.className = "meta"; m.textContent = meta; div.appendChild(m); }
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function ask(text) {
  add(text, "user");
  send.disabled = true;
  try {
    const res = await fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"},
                                      body: JSON.stringify({message: text})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    let meta = `intent: ${data.intent} · confidence: ${(data.confidence * 100).toFixed(0)}%`;
    if (data.source) meta += ` · source: ${data.source}`;
    add(data.response, "bot", meta);
  } catch (e) {
    add("Sorry, something went wrong. Please try again.", "bot");
  } finally { send.disabled = false; input.focus(); }
}

form.addEventListener("submit", e => { e.preventDefault(); const t = input.value.trim(); if (t) { input.value = ""; ask(t); } });
document.getElementById("chips").addEventListener("click", e => { if (e.target.tagName === "BUTTON") ask(e.target.textContent); });
document.getElementById("details").addEventListener("change", e => document.body.classList.toggle("show-details", e.target.checked));
add("Hi! I'm an NLP chatbot. Ask me about NLP, how I was built, trained and evaluated.", "bot");
