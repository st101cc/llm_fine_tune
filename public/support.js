function renderAgents() {
  const latest = Object.entries(appState.rescues).find((item) => item[1].status === "verified");
  const latestDep = latest ? findDependency(latest[0]) : dependencies[0];
  const actions = "<button class='button primary' data-action='rescue' data-dependency='" + latestDep.id + "'>" + icon("spark") + " Run a verification</button>";
  const directory = agents.map((agent) => {
    const capability = agent.id === "tests" ? "Runs focused test suites" : agent.id === "journey" ? "Replays real user paths" : agent.id === "security" ? "Checks production resolution" : "Changes code in an isolated branch";
    return "<article class='panel agent-directory-card'><div class='agent-directory-top'><span class='agent-symbol " + agent.color + "'>" + agent.icon + "</span>" + tag("blue", "independent") + "</div><h2>" + agent.name + "</h2><p>" + agent.role + "</p><div class='agent-capability'>" + capability + "</div></article>";
  }).join("");
  const latestStatus = latest ? tag("green", "verified") : tag("blue", "ready to run");
  const latestTitle = latest ? "All four agents passed" : "No current run";
  const latestCopy = latest ? "The migration passed testing, journey replay, and dependency security verification." : "Open a rescue plan to start the first agent-backed verification.";
  app.innerHTML = "<div class='page-wrap'>" + pageHeader("Independent checks", "Verification agents", "A fix is not called ready because one model says so. Each agent checks a different piece of evidence before a maintainer sees the PR.", actions) +
    "<section class='agent-explainer panel'><div><span class='page-kicker'><span class='kicker-line'></span>Why agents</span><h2>Separate eyes, separate evidence.</h2><p>Codex creates the change. These agents then independently test the code, replay user journeys, and re-check the resolved dependency graph.</p></div><div class='agent-explainer-rule'><span>" + icon("lock") + "</span><strong>No auto-merge</strong><p>A human maintainer still reviews the final PR.</p></div></section><div class='agent-directory'>" + directory + "</div><section class='panel latest-run'><div class='panel-header'><div class='panel-heading'><h2>Latest verification record</h2><p>Traceable evidence for " + latestDep.name + "</p></div>" + latestStatus + "</div><div class='latest-run-body'><div><strong>" + latestTitle + "</strong><p>" + latestCopy + "</p></div><button class='button' data-action='rescue' data-dependency='" + latestDep.id + "'>" + (latest ? "Open record" : "Start rescue") + " " + icon("arrow") + "</button></div></section></div>";
  bindCommonActions();
}

function renderActivity() {
  const rows = [
    ["Migration PR merged", "payments-api accepted the node-fetch upgrade. CI passed on all 48 checks.", "Today · 11:42"],
    ["Nightly scan completed", appState.lastScan.dependenciesChecked + " dependencies checked across acme/storefront, payments-api, and two internal packages.", appState.lastScan.completedAt],
    ["New advisory surfaced", "react-router 5.3.4 was flagged as the highest-priority rescue in acme/storefront.", "Yesterday · 18:32"],
    ["Rescue plan generated", "Codex mapped sharp native build impact and outlined a low-risk migration path.", "Yesterday · 16:20"],
    ["Repository connected", getLastConnectedRepository() ? getLastConnectedRepository() + " was added with read-only scan permissions." : "acme/marketing-site was added with read-only scan permissions.", "Mon · 09:12"]
  ];
  const timeline = rows.map((row) => "<div class='timeline-item'><h3>" + row[0] + "</h3><p>" + row[1] + "</p><span>" + row[2] + "</span></div>").join("");
  app.innerHTML = "<div class='page-wrap'>" + pageHeader("Workspace history", "Activity", "A transparent record of every scan, recommendation, and rescue run across your repositories.", "") + "<section class='panel timeline-panel'><div class='timeline'>" + timeline + "</div></section></div>";
}

function repositoryCard(name, health, description, tracked, attention, lastScan) {
  return "<section class='panel repo-card'><div class='repo-card-header'><strong>" + escapeHtml(name) + "</strong><span>" + health + "</span></div><p>" + description + "</p><div class='repo-metrics'><div class='repo-metric'><strong>" + tracked + "</strong><span>tracked</span></div><div class='repo-metric'><strong>" + attention + "</strong><span>attention</span></div><div class='repo-metric'><strong>" + lastScan + "</strong><span>last scan</span></div></div><button class='row-action repo-action' data-scan-repository='" + escapeHtml(name) + "'>" + icon("scan") + " Run scan <span class='arrow'>" + icon("arrow") + "</span></button></section>";
}

function renderRepositories() {
  const builtInRepositories = ["acme/storefront", "acme/payments-api", "acme/marketing-site"];
  const connected = getConnectedRepositories()
    .filter((repository) => !builtInRepositories.includes(repository))
    .map((repository) => repositoryCard(repository, "Connected", "Added to this workspace. Run a baseline scan or continue monitoring it.", "—", "—", "ready"))
    .join("");
  const cards = repositoryCard("acme/storefront", "Healthy 86%", "Customer-facing commerce experience. Last scan found 8 dependencies to review.", "72", "8", "4m") +
    repositoryCard("acme/payments-api", "Healthy 94%", "Payment orchestration service. Its latest rescue run shipped yesterday.", "41", "2", "8m") +
    repositoryCard("acme/marketing-site", "New", "Recently connected. Codex will complete the first baseline scan shortly.", "—", "—", "12m") + connected;
  app.innerHTML = "<div class='page-wrap'>" + pageHeader("Connected surface area", "Repositories", "Keep every codebase in view, with permissions and health signals you can explain to your team.", "<button class='button primary' data-action='connect'>+ Connect repo</button>") + "<div class='repo-grid'>" + cards + "</div></div>";
  document.querySelectorAll("[data-scan-repository]").forEach((button) => button.addEventListener("click", () => {
    runtime.scan = { repository: button.dataset.scanRepository };
    go("scan");
  }));
  bindCommonActions();
}

function renderSettings() {
  const settingRow = (key, title, description) => "<div class='setting-row'><div><strong>" + title + "</strong><span>" + description + "</span></div><button class='toggle " + (appState.settings[key] ? "on" : "") + "' data-setting='" + key + "' aria-label='" + title + "'></button></div>";
  app.innerHTML = "<div class='page-wrap'>" + pageHeader("Workspace controls", "Settings", "Make the rescue workflow fit your team safety bar.", "") + "<div class='settings-grid'><section class='panel settings-card'><h2>Rescue policy</h2><p>These defaults apply to every Codex migration run in this workspace.</p>" + settingRow("review", "Require human review", "Always open a pull request before merging") + settingRow("tests", "Write regression tests", "Add coverage before touching production code") + settingRow("scan", "Auto-scan new repositories", "Start a baseline scan when connected") + "</section><section class='panel settings-card'><h2>Notifications</h2><p>Choose which moments should reach your engineering channel.</p>" + settingRow("highRisk", "High-risk dependency found", "Send a notification immediately") + settingRow("prReady", "PR ready for review", "Notify the repository owners") + settingRow("digest", "Weekly health digest", "Send every Monday at 09:00") + "</section></div></div>";
  document.querySelectorAll("[data-setting]").forEach((toggle) => toggle.addEventListener("click", () => {
    const key = toggle.dataset.setting;
    appState.settings[key] = !appState.settings[key];
    persistState();
    toggle.classList.toggle("on", appState.settings[key]);
    showToast(appState.settings[key] ? "Setting enabled" : "Setting paused", "Your workspace preference was updated.");
  }));
}

function bindCommonActions() {
  document.querySelectorAll("[data-action='connect']").forEach((button) => button.addEventListener("click", () => {
    runtime.connect = null;
    go("connect");
  }));
  document.querySelectorAll("[data-action='scan']").forEach((button) => button.addEventListener("click", () => go("scan")));
  document.querySelectorAll("[data-action='rescue']").forEach((button) => button.addEventListener("click", () => go("rescue/" + button.dataset.dependency)));
  document.querySelectorAll("[data-action='diff']").forEach((button) => button.addEventListener("click", () => go("dependency/" + button.dataset.dependency + "/diff")));
  bindDependencyLinks();
}

function showToast(title, message) {
  const region = document.querySelector("#toast-region");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = "<span class='toast-icon'>" + icon("spark") + "</span><div><strong>" + escapeHtml(title) + "</strong><span>" + escapeHtml(message) + "</span></div>";
  region.append(toast);
  setTimeout(() => toast.remove(), 4200);
}

app.addEventListener("click", (event) => {
  const backButton = event.target.closest("[data-action='back']");
  if (!backButton || backButton.disabled) return;
  go(logicalBackRoute());
});

window.addEventListener("hashchange", render);
render();
