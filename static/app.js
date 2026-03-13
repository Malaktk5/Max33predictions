// ── LOADING MESSAGES ─────────────────────────────────────────────────────────
const LOAD_MSGS = [
  "summoning max from the simulator...",
  "I did this instead of studying for my test...",
  "everyone pray Malak passes the year with no effort...",
  "maybe i should do a system that predicts norris being ass...",
  "asking the model nicely to say p1...",
  "threatening the api with your MOM...",
  "Max is unpredictable though so this is kinda useless...",
  "calculating the inevitable...",
  "weather check: will it rain? (doesn't matter, max wins anyway)",
];

// ── STATE ─────────────────────────────────────────────────────────────────────
let mcChart  = null;
let msgTick  = null;

// ── HELPERS ───────────────────────────────────────────────────────────────────
function posColor(pos) {
  if (pos <= 1) return 'var(--green)';
  if (pos <= 3) return 'var(--orange)';
  if (pos <= 6) return 'var(--text)';
  return 'var(--muted)';
}

// ── REFRESH ───────────────────────────────────────────────────────────────────
async function doRefresh() {
  const btn     = document.getElementById('ref-btn');
  const overlay = document.getElementById('overlay');
  const msgEl   = document.getElementById('load-msg');

  btn.disabled      = true;
  btn.textContent   = '... loading';
  overlay.style.display = 'flex';

  // Rotate loading messages
  let mi = 0;
  msgEl.textContent = LOAD_MSGS[0];
  msgTick = setInterval(() => {
    mi = (mi + 1) % LOAD_MSGS.length;
    msgEl.textContent = LOAD_MSGS[mi];
  }, 2400);

  try {
    const res  = await fetch('/api/refresh', { method: 'POST' });
    const data = await res.json();
    clearInterval(msgTick);
    data.error ? showError(data.error) : render(data);
  } catch (e) {
    clearInterval(msgTick);
    alert('network error: ' + e);
  }

  btn.disabled    = false;
  btn.textContent = '↺ Refresh';
  overlay.style.display = 'none';
}

// ── ERROR STATE ───────────────────────────────────────────────────────────────
function showError(err) {
  document.getElementById('main').innerHTML = `
    <div style="background:var(--bg2);border:1px solid rgba(255,95,0,.4);border-radius:6px;padding:28px 30px">
      <div style="color:var(--orange);font-family:var(--mono);font-size:8px;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px">
        ⚠ something exploded (not max's fault)
      </div>
      <pre style="font-family:var(--mono);font-size:11px;white-space:pre-wrap;color:var(--text)">${err}</pre>
      <div style="color:var(--muted);font-family:var(--mono);font-size:9px;margin-top:12px">
        check the terminal · blame the api · maybe blame perez
      </div>
    </div>`;
}

// ── RENDER ────────────────────────────────────────────────────────────────────
function render(d) {
  document.getElementById('last-updated').textContent = 'synced ' + d.last_updated;

  const c     = d.championship;
  const preds = d.predictions;
  const nxt   = preds.find(p => !p.completed);
  const done  = preds.filter(p => p.completed).length;

  // ── Spotlight card (next race or season complete)
  const spotlight = nxt
    ? `<div class="spotlight anim">
        <div>
          <div class="sp-label">Next Victim</div>
          <div class="sp-name">${nxt.name}</div>
          <div class="sp-meta">Round ${nxt.round} &nbsp;·&nbsp; ${nxt.date} &nbsp;·&nbsp; historical win rate: ${nxt.win_rate_hist}%</div>
        </div>
        <div class="sp-divider"></div>
        <div class="sp-stat">
          <div class="sp-stat-val">${nxt.win_prob}%</div>
          <div class="sp-stat-lbl">win prob</div>
        </div>
        <div class="sp-stat">
          <div class="sp-stat-val">${nxt.podium_prob}%</div>
          <div class="sp-stat-lbl">podium prob</div>
        </div>
        <div class="sp-stat">
          <div class="sp-stat-val">P${nxt.pred_pos}</div>
          <div class="sp-stat-lbl">predicted</div>
        </div>
      </div>`
    : `<div class="spotlight anim" style="border-left-color:var(--green)">
        <div>
          <div class="sp-label" style="color:var(--green)">Season Status</div>
          <div class="sp-name">SEASON COMPLETE</div>
          <div class="sp-meta">all 24 rounds done · how many did he win?</div>
        </div>
      </div>`;

  // ── Table rows
  const rows = preds.map(p => {
    const isNxt   = p === nxt;
    const posCell = (p.completed && p.actual)
      ? `<span class="apos">P${p.actual}</span>`
      : `<span class="ppos" style="color:${posColor(p.pred_pos)}">P${p.pred_pos}</span>`;
    const stat = p.completed
      ? `<span class="badge b-done">done ✓</span>`
      : isNxt
        ? `<span class="badge b-next">▶ next</span>`
        : `<span class="badge b-up">upcoming</span>`;

    return `
      <tr class="${p.completed ? 'done' : ''} ${isNxt ? 'nxt' : ''}">
        <td><span class="rnd">R${p.round}</span></td>
        <td class="race-nm">${p.name}</td>
        <td class="dt">${p.date}</td>
        <td>${posCell}</td>
        <td>
          <div class="barrow">
            <div class="barbg">
              <div class="barfill" style="width:${Math.min(100, p.win_prob * 2)}%;background:var(--orange)"></div>
            </div>
            <span class="barval" style="color:var(--orange)">${p.win_prob}%</span>
          </div>
        </td>
        <td>
          <div class="barrow">
            <div class="barbg">
              <div class="barfill" style="width:${p.podium_prob}%;background:#ff9030"></div>
            </div>
            <span class="barval" style="color:#ff9030">${p.podium_prob}%</span>
          </div>
        </td>
        <td class="dt">${p.win_rate_hist}%</td>
        <td>${stat}</td>
      </tr>`;
  }).join('');

  // ── Inject HTML
  document.getElementById('main').innerHTML = `
    ${spotlight}

    <div class="champ-grid anim">
      <div class="cm">
        <div class="cm-label">projected points</div>
        <div class="cm-value hi">${c.mean}</div>
        <div class="cm-sub">full season avg</div>
      </div>
      <div class="cm">
        <div class="cm-label">points banked</div>
        <div class="cm-value">${c.completed_pts}</div>
        <div class="cm-sub">${done} race${done !== 1 ? 's' : ''} done</div>
      </div>
      <div class="cm">
        <div class="cm-label">best case (P90)</div>
        <div class="cm-value">${c.p90}</div>
        <div class="cm-sub">top 10% of runs</div>
      </div>
      <div class="cm">
        <div class="cm-label">worst case (P10)</div>
        <div class="cm-value">${c.p10}</div>
        <div class="cm-sub">bottom 10% of runs</div>
      </div>
      <div class="cm">
        <div class="cm-label">races left</div>
        <div class="cm-value">${c.remaining_races}</div>
        <div class="cm-sub">of 24 total</div>
      </div>
    </div>

    <div class="chart-card anim">
      <div class="card-label">10,000 season simulations &nbsp;·&nbsp; championship points distribution</div>
      <canvas id="mcChart" height="72"></canvas>
    </div>

    <div class="races-card anim">
      <div class="races-header">
        <div class="card-label" style="margin:0">2026 Race Predictions &nbsp;·&nbsp; all 24 rounds</div>
        <span style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:2px">${done}/24 COMPLETED</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Race</th>
            <th>Date</th>
            <th>Pos</th>
            <th>Win %</th>
            <th>Podium %</th>
            <th>Hist Win %</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  // ── Monte Carlo chart
  drawChart(c);
}

// ── CHART ─────────────────────────────────────────────────────────────────────
function drawChart(c) {
  if (mcChart) { mcChart.destroy(); mcChart = null; }

  const ctx = document.getElementById('mcChart').getContext('2d');
  mcChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: c.hist_edges.slice(0, -1).map(v => Math.round(v)),
      datasets: [{
        data: c.hist,
        backgroundColor: 'rgba(255, 95, 0, .6)',
        borderColor:     'rgba(255, 95, 0, .85)',
        borderWidth:  1,
        borderRadius: 2,
      }]
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => `${ctx.raw} simulations` },
          bodyFont:  { family: 'IBM Plex Mono' },
          titleFont: { family: 'IBM Plex Mono' },
        }
      },
      scales: {
        x: {
          ticks: { color: '#445570', font: { family: 'IBM Plex Mono', size: 9 } },
          grid:  { color: '#1c2840' }
        },
        y: {
          ticks: { color: '#445570', font: { family: 'IBM Plex Mono', size: 9 } },
          grid:  { color: '#1c2840' }
        }
      }
    }
  });
}

// ── INIT ──────────────────────────────────────────────────────────────────────
window.onload = doRefresh;
