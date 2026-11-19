// Live dashboard: consumes the Server-Sent Events telemetry stream.
(() => {
  const $ = (id) => document.getElementById(id);
  const HISTORY_S = 30;
  const history = [];
  let paused = false;

  function fmt(v, digits = 1) {
    return v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(digits);
  }

  function renderDecision(d) {
    const card = $("decision");
    card.className = `card decision level-${d.level}`;
    $("level").textContent = d.level.toUpperCase();
    $("reason").textContent = d.reason;
    $("speed").textContent = fmt(d.speed_factor, 2);
    const fill = $("gauge-fill");
    fill.style.width = `${Math.round(d.speed_factor * 100)}%`;
    const colors = { safe: "--green", caution: "--yellow", warning: "--amber", critical: "--red" };
    fill.style.background = `var(${colors[d.level]})`;
  }

  function renderStages(timings) {
    const entries = Object.entries(timings).filter(([k]) => k !== "total");
    const max = Math.max(1, ...entries.map(([, v]) => v));
    $("stages").innerHTML = entries
      .map(([k, v]) => `<div class="stage"><span>${k}</span><div class="bar"><div style="width:${(100 * v) / max}%"></div></div><span class="v">${fmt(v, 2)} ms</span></div>`)
      .join("");
  }

  function renderTracks(tracks) {
    const rows = [...tracks]
      .sort((a, b) => (a.world?.distance ?? 1e9) - (b.world?.distance ?? 1e9))
      .slice(0, 12)
      .map((t) => `<tr><td>${t.id}</td><td>${t.class_name}</td><td>${t.world ? fmt(t.world.distance) + " m" : "–"}</td>` +
        `<td>${t.ttc !== undefined ? fmt(t.ttc) + " s" : "–"}</td><td><span class="risk risk-${t.risk}">${t.risk}</span></td></tr>`);
    $("tracks").innerHTML = rows.join("") || `<tr><td colspan="5" class="muted">no object</td></tr>`;
  }

  function addEvents(events) {
    const list = $("events");
    for (const e of events) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="t">${fmt(e.t, 1)} s</span><span class="lv-${e.level}">${e.message}</span>`;
      list.prepend(li);
    }
    while (list.children.length > 80) list.lastChild.remove();
  }

  function drawChart() {
    const canvas = $("chart");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;
    const t1 = history[history.length - 1].t, t0 = t1 - HISTORY_S;
    const x = (t) => ((t - t0) / HISTORY_S) * (w - 40) + 30;
    const css = getComputedStyle(document.documentElement);
    ctx.strokeStyle = css.getPropertyValue("--line");
    ctx.fillStyle = css.getPropertyValue("--muted");
    ctx.font = "11px sans-serif";
    const yMax = 40;
    const y = (v) => h - 18 - (Math.min(v, yMax) / yMax) * (h - 30);
    for (let v = 0; v <= yMax; v += 10) {
      ctx.beginPath(); ctx.moveTo(30, y(v)); ctx.lineTo(w - 10, y(v)); ctx.stroke();
      ctx.fillText(String(v), 4, y(v) + 4);
    }
    const series = [
      ["dist", "--blue", (v) => y(v)],
      ["ttc", "--amber", (v) => y(v * 4)],
      ["speed", "--green", (v) => y(v * yMax)],
    ];
    for (const [key, color, map] of series) {
      ctx.strokeStyle = css.getPropertyValue(color);
      ctx.lineWidth = 2;
      ctx.beginPath();
      let pen = false;
      for (const p of history) {
        const v = p[key];
        if (v === null || v === undefined) { pen = false; continue; }
        const px = x(p.t), py = map(v);
        if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    ctx.fillStyle = css.getPropertyValue("--muted");
    ctx.fillText("TTC axis ×4 · speed factor ×40", w - 190, 12);
  }

  function onMessage(msg) {
    const { state, stats } = JSON.parse(msg.data);
    if (!state || state.decision === undefined) return;
    renderDecision(state.decision);
    $("fps").textContent = fmt(state.fps);
    $("lat").textContent = stats && stats.p95 !== undefined ? fmt(stats.p95) : "–";
    $("ntracks").textContent = state.tracks.length;
    $("ego").textContent = state.ego_speed !== undefined ? fmt(state.ego_speed * 3.6, 0) : "–";
    $("frame").textContent = `frame ${state.frame}`;
    renderStages(state.timings_ms);
    renderTracks(state.tracks);
    addEvents(state.events);
    history.push({ t: state.t, dist: state.decision.min_distance, ttc: state.decision.min_ttc, speed: state.decision.speed_factor });
    while (history.length && history[0].t < state.t - HISTORY_S) history.shift();
    drawChart();
  }

  function connect() {
    const es = new EventSource("/api/telemetry");
    es.onopen = () => { $("conn").textContent = "live"; $("conn").className = "pill pill-on"; };
    es.onmessage = onMessage;
    es.onerror = () => {
      $("conn").textContent = "offline"; $("conn").className = "pill pill-off";
      es.close(); setTimeout(connect, 1500);
    };
  }

  async function refreshHealth() {
    try {
      const h = await (await fetch("/api/health")).json();
      $("uptime").textContent = `up ${Math.round(h.uptime_s)} s · ${h.frames} frames`;
    } catch (_) { /* server restarting */ }
  }

  $("pause").addEventListener("click", async () => {
    paused = !paused;
    await fetch("/api/control", { method: "POST", body: JSON.stringify({ paused }) });
    $("pause").textContent = paused ? "Resume" : "Pause";
  });

  window.addEventListener("resize", drawChart);
  setInterval(refreshHealth, 2000);
  connect();
})();
