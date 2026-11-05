const $ = (id) => document.getElementById(id);
const fmtBytes = (value) => value == null ? "-" : `${(Number(value) / 1024 ** 3).toFixed(2)} GiB`;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...($("apiToken")?.value ? { "X-MiniCoder-Token": $("apiToken").value } : {}),
      ...(options.headers || {}),
    },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

function addCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
  row.appendChild(cell);
}

function render(status, workers, providerData) {
  const metric = status.latest_metric || {};
  const badge = $("status");
  badge.textContent = status.status || "unknown";
  badge.className = `badge ${["running", "finished", "idle"].includes(status.status) ? status.status : "idle"}`;
  $("step").textContent = metric.global_step ?? metric.step ?? status.latest_event?.global_step ?? "-";
  $("loss").textContent = metric.training_loss ?? metric.loss ?? metric.last_train_loss ?? "-";
  $("tps").textContent = metric.tokens_per_second ?? metric.tokens_per_s ?? "-";
  $("pid").textContent = status.pid ?? "-";
  $("device").textContent = [metric.provider_name, metric.gpu_name].filter(Boolean).join(" / ") || "-";
  $("vram").textContent = `${fmtBytes(metric.gpu_memory_used_bytes ?? metric.gpu_peak_allocated_bytes)} / ${fmtBytes(metric.gpu_memory_total_bytes ?? metric.gpu_vram_total_bytes)}`;
  $("ram").textContent = `${fmtBytes(metric.cpu_ram_used_bytes)} / ${fmtBytes(metric.cpu_ram_total_bytes)}`;
  $("gpuUtil").textContent = metric.gpu_utilization_percent == null ? "-" : `${metric.gpu_utilization_percent}%`;
  $("learningRate").textContent = metric.learning_rate ?? metric.lr ?? "-";
  $("tokensSeen").textContent = metric.tokens_seen ?? "-";
  $("eta").textContent = metric.estimated_remaining_seconds ?? "-";
  $("checkpointStatus").textContent = status.checkpoint_status || "-";
  $("checkpointPath").textContent = `Checkpoint path: ${status.checkpoint_path || "-"}`;
  $("metrics").textContent = (status.recent_metrics || []).slice(-12).map((item) => JSON.stringify(item)).join("\n") || "No metrics yet";

  const body = $("workers");
  body.replaceChildren();
  const entries = Object.values(workers || {});
  for (const worker of entries) {
    const row = document.createElement("tr");
    for (const key of ["worker_id", "provider", "gpu_name"]) addCell(row, worker[key]);
    addCell(row, `${worker.vram_available ?? 0}/${worker.vram_total ?? 0}`);
    addCell(row, worker.network_id);
    addCell(row, worker.distributed_backend);
    addCell(row, worker.status);
    body.appendChild(row);
  }
  if (!entries.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.textContent = "No registered workers";
    row.appendChild(cell);
    body.appendChild(row);
  }
  renderProviders(providerData);
}

function renderProviders(data) {
  const body = $("providers");
  body.replaceChildren();
  const reports = data?.provider_reports || {};
  for (const [name, report] of Object.entries(reports)) {
    const row = document.createElement("tr");
    const observed = report.runtime_resources || {};
    const devices = report.gpu?.torch?.devices || report.gpu?.nvidia_smi?.devices || [];
    addCell(row, name);
    addCell(row, report.runtime_detected ? "Detected here" : "Not detected here");
    addCell(row, observed.gpu_name || devices.map((item) => item.name || "GPU").join(", ") || "Not observed");
    addCell(row, fmtBytes(observed.system_ram_total_bytes));
    addCell(row, fmtBytes(observed.storage_free_bytes));
    addCell(row, [observed.cuda_runtime || "CUDA ?", observed.torch_version || "PyTorch ?"].join(" / "));
    const submission = report.capabilities?.submit_cloud_job;
    addCell(row, submission === false || submission === "unsupported" ? "No remote submission" : "Not verified");
    body.appendChild(row);
  }
}

async function refresh() {
  try {
    const [status, workers, providerData] = await Promise.all([api("/status"), api("/workers"), api("/providers")]);
    render(status, workers, providerData);
  } catch (error) {
    $("message").textContent = error.message;
  }
}

async function control(path, body) {
  try {
    const result = await api(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });
    $("message").textContent = JSON.stringify(result, null, 2);
    await refresh();
  } catch (error) {
    $("message").textContent = error.message;
  }
}

$("start").onclick = () => {
  const rate = $("costRate").value.trim();
  const budget = $("costBudget").value.trim();
  control("/train/start", {
    config: $("config").value,
    train_config: $("trainConfig").value || null,
    run_dir: $("runDir").value,
    gpu_cost_rate_usd_per_hour: rate ? Number(rate) : null,
    max_budget_usd: budget ? Number(budget) : null,
    confirm_potential_cost: $("confirmCost").checked,
  });
};
$("pause").onclick = () => control("/train/pause");
$("resume").onclick = () => control("/train/resume");
$("checkpoint").onclick = () => control("/train/checkpoint");
$("stop").onclick = () => control("/train/stop");
$("restore").onclick = () => control("/train/restore");
$("export").onclick = () => control("/train/export", { pack_bitnet: false });
$("kagglePrepare").onclick = () => {
  try {
    const metadata = JSON.parse($("kaggleMetadata").value);
    control("/providers/kaggle/prepare", { kernel_dir: $("kaggleDir").value, metadata });
  } catch (error) { $("message").textContent = `Invalid metadata JSON: ${error.message}`; }
};
$("kaggleSubmit").onclick = () => control("/providers/kaggle/submit", {
  kernel_dir: $("kaggleDir").value,
  confirm_submit: $("confirmKaggle").checked,
});
$("kaggleStatus").onclick = () => control("/providers/kaggle/status", {
  kernel_id: $("kaggleId").value,
});
$("kaggleOutput").onclick = () => control("/providers/kaggle/output", {
  kernel_id: $("kaggleId").value,
  destination_dir: $("kaggleOutputDir").value,
});
refresh();
setInterval(refresh, 2000);
