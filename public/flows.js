function verificationResult(agentId) {
  return ({
    migration: "3 source files updated and lockfile resolved.",
    tests: "42/42 focused tests passed.",
    journey: "4 critical storefront journeys completed.",
    security: "No vulnerable production resolution remains."
  })[agentId];
}

function verificationAgentCard(agent, index, step, verified) {
  const passed = verified || index < step;
  const current = !verified && index === step;
  const status = passed ? "Passed" : current ? "Running" : "Queued";
  const statusClass = passed ? "passed" : current ? "running" : "queued";
  const result = passed ? verificationResult(agent.id) : current ? "Checking the isolated branch…" : "Starts after the prior check finishes.";
  return "<article class='panel agent-run-card " + statusClass + "'><div class='agent-run-top'><span class='agent-symbol " + agent.color + "'>" + agent.icon + "</span><span class='agent-status " + statusClass + "'>" + (passed ? icon("check") : current ? icon("clock") : "•") + " " + status + "</span></div><h3>" + agent.name + "</h3><p>" + agent.role + "</p><div class='agent-result'>" + result + "</div></article>";
}

function renderRescueRun(id) {
  const dep = findDependency(id);
  const stored = appState.rescues[dep.id];
  const verified = stored && stored.status === "verified";
  const running = runtime.rescue && runtime.rescue.id === dep.id && runtime.rescue.running;
  const step = verified ? agents.length : running ? runtime.rescue.step : 0;
  const percent = verified ? 100 : Math.round((step / agents.length) * 100);
  const headerActions = verified ? "<button class='button success-button' data-action='diff' data-dependency='" + dep.id + "'>View verified diff</button>" : "";
  const heroButton = verified
    ? "<a class='button primary' href='#/activity'>View activity " + icon("arrow") + "</a>"
    : "<button class='button primary' id='start-verification' " + (running ? "disabled" : "") + ">" + (running ? "Verification in progress…" : "Start rescue + verification") + "</button>";
  const heroTitle = verified ? "Verification complete" : running ? "Agents are checking the fix" : "Ready to create and verify a fix";
  const heroCopy = verified
    ? "The update cleared the advisory, passed focused coverage, and held up across critical user journeys."
    : running
      ? "Each agent works independently so a green code change is not mistaken for a safe release."
      : "Start a controlled Codex run in an isolated branch. It repairs the code, writes tests, and invites the agent team to check the result.";
  const evidenceTone = verified ? tag("green", "all checks passed") : tag("blue", running ? "collecting evidence" : "awaiting run");
  app.innerHTML = "<div class='page-wrap'><a class='back-link' href='#/dependency/" + dep.id + "'>" + icon("back") + " Back to rescue plan</a>" + pageHeader("Verification run", dep.name + " rescue", verified ? "Every independent check passed. The migration is ready for a maintainer to review." : "The fix only becomes review-ready after independent agents validate it from different angles.", headerActions) +
    "<section class='panel rescue-hero'><div class='rescue-hero-copy'><span class='agent-orb'>" + (running ? icon("clock") : verified ? icon("check") : icon("spark")) + "</span><div><h2>" + heroTitle + "</h2><p>" + heroCopy + "</p></div></div>" + heroButton + "</section>" +
    "<div class='verification-progress'><div><span>Run progress</span><strong>" + percent + "%</strong></div><div class='progress-track'><span style='width:" + percent + "%'></span></div></div>" +
    "<section class='agent-run-grid'>" + agents.map((agent, index) => verificationAgentCard(agent, index, step, verified)).join("") + "</section>" +
    "<section class='panel evidence-panel'><div class='panel-header'><div class='panel-heading'><h2>Evidence collected</h2><p>What a maintainer sees before merging</p></div>" + evidenceTone + "</div><div class='evidence-list'><div><span class='evidence-icon'>" + icon("check") + "</span><div><strong>Focused regression coverage</strong><p>" + (verified ? "42 tests passed, including 12 tests added for the migration." : "The test agent will run focused unit and integration checks.") + "</p></div></div><div><span class='evidence-icon'>" + icon("scan") + "</span><div><strong>Production dependency graph</strong><p>" + (verified ? "No vulnerable react-router 5.x resolution remains in the production bundle." : "The security agent will inspect the resolved production graph.") + "</p></div></div><div><span class='evidence-icon'>" + icon("merge") + "</span><div><strong>Critical journey replay</strong><p>" + (verified ? "Browse, add-to-cart, checkout, and direct deep links all completed successfully." : "The journey agent will replay routes affected by the migration.") + "</p></div></div></div></section></div>";
  document.querySelector("#start-verification")?.addEventListener("click", () => startVerification(dep));
  bindCommonActions();
}

function startVerification(dep) {
  if (runtime.rescue && runtime.rescue.running) return;
  runtime.rescue = { id: dep.id, step: 0, running: true };
  appState.rescues[dep.id] = { status: "running", startedAt: "just now" };
  persistState();
  render();
  const advance = () => {
    if (!runtime.rescue || runtime.rescue.id !== dep.id) return;
    runtime.rescue.step += 1;
    if (runtime.rescue.step >= agents.length) {
      runtime.rescue.running = false;
      appState.rescues[dep.id] = { status: "verified", completedAt: "just now" };
      persistState();
      showToast("Verification complete", "All independent checks passed for " + dep.name + ".");
    }
    const parts = routeParts();
    if (parts[0] === "rescue" && parts[1] === dep.id) render();
    if (runtime.rescue && runtime.rescue.running) setTimeout(advance, 700);
  };
  setTimeout(advance, 650);
}

function scanForm(repositories, selectedRepository) {
  const options = repositories.map((repo) => "<option value='" + escapeHtml(repo) + "'" + (repo === selectedRepository ? " selected" : "") + ">" + escapeHtml(repo) + "</option>").join("");
  return "<section class='panel scan-form'><div class='form-intro'><span class='form-symbol'>" + icon("scan") + "</span><div><h2>Choose what to scan</h2><p>A scan is read-only. It looks at package manifests, lockfiles, and the dependency graph.</p></div></div><label class='field-label' for='scan-repository'>Repository</label><select id='scan-repository' class='form-select'>" + options + "</select><div class='scan-scope'><span>" + icon("check") + " package manifests</span><span>" + icon("check") + " lockfile resolutions</span><span>" + icon("check") + " advisory exposure</span><span>" + icon("check") + " blast radius</span></div><button class='button primary' id='start-scan'>" + icon("scan") + " Start dependency scan</button></section>";
}

function scanResults(repository) {
  return "<section class='scan-results'><div class='scan-result-head'><span class='complete-orb'>" + icon("check") + "</span><div><h2>" + escapeHtml(repository) + " is scanned</h2><p>184 dependencies checked · 12 need attention · 3 high priority</p></div></div><div class='scan-result-grid'><article><strong>3</strong><span>high priority</span><p>Security advisories or unsupported versions with production exposure.</p></article><article><strong>6</strong><span>safe upgrades</span><p>Codex found a compatible migration path with low estimated effort.</p></article><article><strong>86%</strong><span>healthy coverage</span><p>Dependencies on supported versions with no known advisory exposure.</p></article></div><div class='scan-results-actions'><button class='button' data-action='scan-again'>Scan another repository</button><button class='button primary' data-action='review-findings'>Review rescue queue " + icon("arrow") + "</button></div></section>";
}

function renderScan() {
  const scan = runtime.scan || {};
  const scanning = Boolean(scan.running);
  const completed = Boolean(scan.completed);
  const selected = scan.repository || "acme/storefront";
  const repositories = Array.from(new Set(["acme/storefront", "acme/payments-api", "acme/marketing-site"].concat(getConnectedRepositories())));
  let body;
  if (scanning) {
    const progressText = scan.progress < 38 ? "Reading manifests and lockfiles…" : scan.progress < 75 ? "Mapping transitive dependencies and advisories…" : "Ranking upgrade paths by risk and effort…";
    body = "<section class='panel scan-running'><div class='scan-orbit'>" + icon("scan") + "</div><div><span class='page-kicker'><span class='kicker-line'></span>In progress</span><h2>Inspecting " + escapeHtml(selected) + "</h2><p>" + progressText + "</p><div class='scan-progress'><span style='width:" + scan.progress + "%'></span></div><strong>" + scan.progress + "% complete</strong></div></section>";
  } else if (completed) {
    body = scanResults(selected);
  } else {
    body = scanForm(repositories, selected);
  }
  const title = scanning ? "Scanning repository…" : completed ? "Scan complete" : "Scan a repository";
  const subtitle = scanning ? "Codex is reading the manifest, lockfile, and dependency graph to identify what needs attention." : completed ? escapeHtml(selected) + " is up to date. The rescue queue has been refreshed with the findings below." : "Choose a connected repository and run a focused, read-only dependency scan.";
  app.innerHTML = "<div class='page-wrap'><a class='back-link' href='#/repositories'>" + icon("back") + " Back to repositories</a>" + pageHeader("Repository scan", title, subtitle, "") + body + "</div>";
  document.querySelector("#start-scan")?.addEventListener("click", () => startScan(document.querySelector("#scan-repository").value));
  document.querySelector("[data-action='review-findings']")?.addEventListener("click", () => go("dependencies"));
  document.querySelector("[data-action='scan-again']")?.addEventListener("click", () => { runtime.scan = null; renderScan(); });
}

function startScan(repository) {
  runtime.scan = { running: true, completed: false, progress: 8, repository };
  render();
  const steps = [26, 49, 74, 100];
  const advance = () => {
    if (!runtime.scan || !runtime.scan.running) return;
    const next = steps.shift();
    runtime.scan.progress = next;
    if (next === 100) {
      runtime.scan.running = false;
      runtime.scan.completed = true;
      appState.lastScan = { repository, completedAt: "just now", dependenciesChecked: 184, needsAttention: 12 };
      persistState();
      showToast("Scan complete", "Found 12 dependencies worth reviewing in " + repository + ".");
    }
    if (routeParts()[0] === "scan") render();
    if (runtime.scan && runtime.scan.running) setTimeout(advance, 520);
  };
  setTimeout(advance, 400);
}

function normalizeGitHubRepository(value) {
  const input = value.trim().replace(/\/+$/, "").replace(/\.git$/, "");
  if (/^[\w.-]+\/[\w.-]+$/.test(input)) return input;
  try {
    const url = new URL(input);
    if (!["github.com", "www.github.com"].includes(url.hostname.toLowerCase())) return null;
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length !== 2 || !parts.every((part) => /^[\w.-]+$/.test(part))) return null;
    return parts[0] + "/" + parts[1].replace(/\.git$/, "");
  } catch {
    return null;
  }
}

function renderConnect() {
  const connecting = runtime.connect && runtime.connect.running;
  const connected = runtime.connect && runtime.connect.completed;
  const title = connected ? "Repository connected" : connecting ? "Connecting repository…" : "Connect a repository";
  const subtitle = connected ? "It is now part of your workspace and ready for its first read-only scan." : "Add another repository to monitor dependency risk. This demo stores connections only in this browser.";
  let body;
  if (connecting) {
    const currentStep = runtime.connect.step;
    const message = currentStep === 1 ? "Checking repository access…" : currentStep === 2 ? "Registering the dependency scan…" : "Applying your workspace rescue policy…";
    body = "<section class='panel connect-progress'><span class='scan-orbit'>" + icon("repo") + "</span><div><h2>Setting up read-only monitoring</h2><p>" + message + "</p><div class='scan-progress'><span style='width:" + (currentStep * 33) + "%'></span></div></div></section>";
  } else if (connected) {
    body = connectSuccess();
  } else {
    body = connectForm();
  }
  app.innerHTML = "<div class='page-wrap'><a class='back-link' href='#/repositories'>" + icon("back") + " Back to repositories</a>" + pageHeader("Repository onboarding", title, subtitle, "") + body + "</div>";
  document.querySelector("#connect-repository")?.addEventListener("click", () => {
    const input = document.querySelector("#repository-name");
    const repository = normalizeGitHubRepository(input.value);
    if (!repository) {
      document.querySelector("#repository-error").textContent = "Enter owner/repository or a GitHub URL such as https://github.com/acme/inventory-service.";
      input.focus();
      return;
    }
    startConnect(repository);
  });
  document.querySelector("[data-action='scan-connected']")?.addEventListener("click", () => {
    runtime.scan = { repository: runtime.connect?.repository || getLastConnectedRepository() };
    go("scan");
  });
  document.querySelector("[data-action='connect-another']")?.addEventListener("click", () => {
    runtime.connect = null;
    renderConnect();
  });
}

function connectForm() {
  return "<section class='panel connect-form'><div class='connection-steps'><div class='connection-step active'><span>1</span><div><strong>Paste GitHub repository</strong><p>Use a full GitHub URL or owner/repository.</p></div></div><div class='connection-step'><span>2</span><div><strong>Grant read-only access</strong><p>A real integration would request GitHub access here.</p></div></div><div class='connection-step'><span>3</span><div><strong>Start baseline scan</strong><p>A sandboxed runner would install and test the project.</p></div></div></div><div class='connect-fields'><label class='field-label' for='repository-name'>GitHub repository</label><input class='form-select' id='repository-name' type='url' value='https://github.com/acme/inventory-service' placeholder='https://github.com/owner/repository' autocomplete='off' /><div id='repository-error' class='field-error' aria-live='polite'></div><div class='permission-note'><span>" + icon("lock") + "</span><div><strong>Prototype behavior</strong><p>The URL is validated and added to this browser only. This version does not clone, install, execute, or test GitHub code yet.</p></div></div><button class='button primary' id='connect-repository'>Add repository to demo " + icon("arrow") + "</button></div></section>";
}

function connectSuccess() {
  const repository = runtime.connect?.repository || getLastConnectedRepository();
  return "<section class='panel connect-success'><span class='complete-orb'>" + icon("check") + "</span><h2>" + escapeHtml(repository) + " is connected</h2><p>Added to the local demo. No GitHub code was fetched, installed, executed, or tested.</p><div class='success-metrics'><div><strong>Read-only</strong><span>planned access level</span></div><div><strong>Daily</strong><span>planned scans</span></div><div><strong>Human review</strong><span>required for PRs</span></div></div><div class='scan-results-actions'><button class='button' data-action='connect-another'>+ Add another repository</button><a class='button' href='#/repositories'>View repositories</a><button class='button primary' data-action='scan-connected'>" + icon("scan") + " Simulate repository scan</button></div></section>";
}

function startConnect(repository) {
  runtime.connect = { running: true, completed: false, step: 1, repository };
  render();
  const advance = () => {
    if (!runtime.connect || !runtime.connect.running) return;
    runtime.connect.step += 1;
    if (runtime.connect.step > 3) {
      runtime.connect = { running: false, completed: true, step: 3, repository };
      appState.connectedRepository = repository;
      appState.connectedRepositories = Array.from(new Set(getConnectedRepositories().concat(repository)));
      persistState();
      showToast("Repository connected", repository + " is ready for a baseline scan.");
    }
    if (routeParts()[0] === "connect") render();
    if (runtime.connect && runtime.connect.running) setTimeout(advance, 550);
  };
  setTimeout(advance, 500);
}
