const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");
let eventSource = null;
let latestSnapshot = null;
let currentSpeed = "5";
let running = false;
let selectedUnit = null;
let isConnecting = false;
let reconnectTimeout = null;
let currentRole = "tech"; // tech (실무자) 또는 supervisor (상급자)

function statusLabel(status) {
  if (status === "danger") return "위험";
  if (status === "inspect") return "점검";
  if (status === "maintained") return "정비";
  if (status === "under_maintenance") return "정비중";
  if (status === "pending_supervisor" || status === "pending") return "결재대기";
  return "건강";
}

async function loadInitialState() {
  const response = await fetch("/api/state", { cache: "no-store" });
  const payload = await response.json();
  $("fleetSize").textContent = payload.fleet_size;
  renderSnapshot(payload.initial);
}

function renderSnapshot(snapshot) {
  latestSnapshot = snapshot;
  $("timeLabel").textContent = snapshot.stream_time;
  renderFleet(snapshot.engines, snapshot.touched_units || []);
  renderCost(snapshot.cost);
  renderSummary(snapshot.summary);
  renderRisks(snapshot.top_risks);
  renderCriticalQueue(snapshot.top_risks);
  renderEngineDetail();
  renderWorkOrders(snapshot.work_orders || []);
  renderLog(snapshot.log);
}

function renderFleet(engines, touchedUnits) {
  const touched = new Set(touchedUnits.map((unit) => Number(unit)));
  $("fleetGrid").innerHTML = engines.map((engine) => {
    let statusClass = engine.status;
    if (engine.under_maintenance) {
      statusClass = "under_maintenance";
    } else if (engine.pending_supervisor) {
      statusClass = "pending";
    }
    const rul = Math.round(engine.rul);
    const pulse = touched.has(Number(engine.unit)) ? "pulse" : "";
    const activeClass = Number(engine.unit) === Number(selectedUnit) ? "selected" : "";
    return `
      <button class="engine ${statusClass} ${pulse} ${activeClass}" data-unit="${engine.unit}" title="Unit ${engine.unit} | cycle ${engine.stream_cycle} | ${statusLabel(statusClass)}">
        <span class="id">#${engine.unit}</span>
        <span class="rul">${rul}</span>
      </button>
    `;
  }).join("");
  document.querySelectorAll(".engine").forEach((button) => {
    button.addEventListener("click", () => {
      selectedUnit = Number(button.dataset.unit);
      document.querySelectorAll(".engine").forEach((btn) => btn.classList.remove("selected"));
      button.classList.add("selected");
      renderEngineDetail();
    });
  });
}

function renderCost(cost) {
  if ($("protectedHeader")) {
    $("protectedHeader").textContent = `${cost.protected_failures}회`;
  }
  if ($("agentCost")) {
    $("agentCost").textContent = `$${fmt.format(cost.agent)}`;
    $("baselineCost").textContent = `$${fmt.format(cost.baseline)}`;
    $("protected").textContent = `${cost.protected_failures}회`;
    $("missed").textContent = `${cost.missed_failures}회`;
    const maxCost = Math.max(cost.agent, cost.baseline, 1);
    $("agentBar").style.width = `${Math.max(3, (cost.agent / maxCost) * 100)}%`;
    $("baselineBar").style.width = `${Math.max(3, (cost.baseline / maxCost) * 100)}%`;
  }
}

function renderSummary(summary) {
  $("healthyCount").textContent = summary.healthy;
  $("inspectCount").textContent = summary.inspect;
  $("dangerCount").textContent = summary.danger;
  $("maintainedCount").textContent = summary.maintained;
}

function renderRisks(risks) {
  $("riskRows").innerHTML = risks.map((risk) => `
    <tr>
      <td>#${risk.unit}</td>
      <td>${Math.round(risk.rul)}</td>
      <td>${statusLabel(risk.status)}</td>
      <td>${risk.pred_uncertainty.toFixed(1)}</td>
    </tr>
  `).join("");
}

function renderCriticalQueue(risks) {
  $("criticalQueue").innerHTML = risks.map((risk) => {
    const action = risk.status === "danger" ? "즉시 정비 권고" : "점검 예약 권고";
    return `
      <button class="queue-item ${risk.status}" data-unit="${risk.unit}">
        <span>#${risk.unit}</span>
        <b>RUL ${Math.round(risk.rul)}</b>
        <em>${action}</em>
      </button>
    `;
  }).join("");
  document.querySelectorAll(".queue-item").forEach((button) => {
    button.addEventListener("click", () => {
      selectedUnit = Number(button.dataset.unit);
      renderEngineDetail();
    });
  });
  if (!selectedUnit && risks.length) selectedUnit = Number(risks[0].unit);
}

async function fetchDiagnostics(unit) {
  try {
    const response = await fetch(`/api/diagnose?unit=${unit}`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.error("진단 호출 에러:", err);
    return null;
  }
}

async function renderEngineDetail() {
  if (!latestSnapshot) return;
  const engines = latestSnapshot.engines || [];
  const risks = latestSnapshot.top_risks || [];
  const engine = engines.find((item) => Number(item.unit) === Number(selectedUnit));
  const risk = risks.find((item) => Number(item.unit) === Number(selectedUnit));
  if (!engine) {
    $("engineDetail").className = "engine-detail empty";
    $("engineDetail").textContent = "위험 엔진을 선택하세요.";
    $("diagnosticsDetail").style.display = "none";
    return;
  }
  const rul = Math.round(engine.rul);
  const uncertainty = Number(engine.pred_uncertainty).toFixed(1);
  
  // 상태별 상세 지침 텍스트 매핑
  let recommendation = "정상 모니터링";
  let statusStr = engine.status;
  if (engine.under_maintenance) {
    recommendation = "정비 작업 지시 집행 중 (3틱 대기)";
    statusStr = "under_maintenance";
  } else if (engine.pending_supervisor) {
    recommendation = "실무자 상신 완료 → 상급자 최종 결재 대기";
    statusStr = "pending_supervisor";
  } else if (engine.status === "danger") {
    recommendation = "즉시 정비 1차 상신 권고";
  } else if (engine.status === "inspect") {
    recommendation = "점검 상신 또는 보류 후 모니터링";
  }

  // 2단계 결재 권한 버튼 노출 제어 (실무자 vs 상급자 모드 완벽 분리)
  const btnRequest = $("btnRequest");
  const btnFinalApprove = $("btnFinalApprove");
  const btnDefer = $("btnDefer");
  const btnReject = $("btnReject");

  if (currentRole === "tech") {
    // 실무자 모드: 1차 상신 및 보류만 표출 (이미 상신 완료했거나 정비중인 엔진은 버튼 숨김)
    if (engine.pending_supervisor || engine.under_maintenance || engine.status === "maintained") {
      btnRequest.style.display = "none";
      btnDefer.style.display = "none";
    } else {
      btnRequest.style.display = "block";
      btnDefer.style.display = "block";
    }
    btnFinalApprove.style.display = "none";
    btnReject.style.display = "none";
  } else if (currentRole === "supervisor") {
    // 상급자 모드: 최종 승인 및 반려만 표출 (1차 상신(pending) 상태인 엔진만 결재 처리 가능)
    if (engine.pending_supervisor) {
      btnFinalApprove.style.display = "block";
      btnReject.style.display = "block";
    } else {
      btnFinalApprove.style.display = "none";
      btnReject.style.display = "none";
    }
    btnRequest.style.display = "none";
    btnDefer.style.display = "none";
  }

  const reason = [
    `예측 RUL ${rul}`,
    `상태 ${statusLabel(statusStr)}`,
    `예측 불확실성 ${uncertainty}`,
    risk ? `위험점수 ${Number(risk.risk_score).toFixed(1)}` : "상위 위험 큐 외"
  ].join(" · ");
  
  $("engineDetail").className = "engine-detail";
  $("engineDetail").innerHTML = `
    <div class="detail-title">Engine #${engine.unit}</div>
    <div class="detail-rul">${rul}</div>
    <div class="detail-status ${statusStr}">${recommendation}</div>
    <p>${reason}</p>
    <p>권장 근거: RUL 임계값, 예측 불확실성, 인간 피드백 가중치를 종합 계산했습니다.</p>
  `;

  // 위험 또는 점검 대상이거나 결재 대기 중일 때 Z-score 진단 표시
  if (engine.status === "danger" || engine.status === "inspect" || engine.pending_supervisor) {
    const diagData = await fetchDiagnostics(engine.unit);
    if (diagData && diagData.diagnose.anomalies.length > 0) {
      $("diagnosticsDetail").style.display = "block";
      
      // z-score 절대값을 바탕으로 게이지 바 생성
      $("anomalyBars").innerHTML = diagData.diagnose.anomalies.map((anom) => {
        const absZ = Math.abs(anom.z_score);
        const percent = Math.min(100, Math.max(6, (absZ / 4.0) * 100));
        const color = anom.z_score > 0 ? "var(--red)" : "var(--blue)";
        const direction = anom.z_score > 0 ? "상승" : "하락";
        return `
          <div class="anomaly-bar-row">
            <div class="anomaly-bar-info">
              <span>센서 ${anom.sensor} (${direction})</span>
              <span>편차: ${anom.z_score > 0 ? '+' : ''}${anom.z_score.toFixed(1)}σ</span>
            </div>
            <div class="anomaly-bar-bg">
              <div class="anomaly-bar-fill" style="width: ${percent}%; background-color: ${color};"></div>
            </div>
          </div>
        `;
      }).join("");

      // 가이드라인 바인딩
      $("recommendationList").innerHTML = diagData.recommend.checklist.map((item) => `
        <li>
          <strong>[${item.part}]</strong> ${item.action} 
          <span style="color: var(--muted); font-size: 13px;">(예상: ${item.hours}시간)</span>
        </li>
      `).join("");
    } else {
      $("diagnosticsDetail").style.display = "none";
    }
  } else {
    $("diagnosticsDetail").style.display = "none";
  }
}

function renderWorkOrders(orders) {
  if (!orders.length) {
    $("workOrders").innerHTML = `<div class="empty-line">아직 발행된 작업지시가 없습니다.</div>`;
    return;
  }
  $("workOrders").innerHTML = orders.map((order) => {
    const reportBtn = order.report_md ? `
      <button class="view-report-btn" data-order-id="${order.id}">📋 정비 완료 보고서 보기</button>
    ` : "";
    return `
      <div class="work-order ${order.decision}">
        <b>${order.id}</b>
        <span>[${order.time}] Engine #${order.unit} · ${order.decision} · ${order.status}</span>
        <em>${order.reason}</em>
        ${reportBtn}
      </div>
    `;
  }).join("");

  // 보고서 보기 버튼 리스너
  document.querySelectorAll(".view-report-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const orderId = btn.dataset.orderId;
      const order = orders.find((o) => o.id === orderId);
      if (order && order.report_md) {
        $("modalBody").innerHTML = parseMarkdown(order.report_md);
        $("reportModal").style.display = "block";
      }
    });
  });
}

function renderLog(logs) {
  $("agentLog").innerHTML = logs.map((entry) => `
    <div class="log-entry ${entry.agent}">
      <span>[${entry.time}]</span> ${entry.message}
    </div>
  `).join("");
}

function startStream() {
  stopStream(false);
  eventSource = new EventSource(`/api/events?speed=${currentSpeed}`);
  running = true;
  $("toggleStream").textContent = "일시정지";
  $("liveBadge").textContent = `LIVE x${currentSpeed}`;
  $("liveBadge").className = "live-badge live";
  eventSource.addEventListener("meta", (event) => {
    const meta = JSON.parse(event.data);
    $("fleetSize").textContent = meta.fleet_size;
  });
  eventSource.addEventListener("snapshot", (event) => {
    renderSnapshot(JSON.parse(event.data));
  });
  eventSource.onerror = () => {
    stopStream(false);
    $("agentLog").innerHTML = `<div class="log-entry crisis_detector">실시간 스트림 연결이 끊겼습니다. 서버 상태를 확인하세요.</div>`;
  };
}

function stopStream(updateUi = true) {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  running = false;
  if (updateUi) {
    $("toggleStream").textContent = "실시간 스트림 시작";
    $("liveBadge").textContent = "PAUSED";
    $("liveBadge").className = "live-badge paused";
  }
}

$("toggleStream").addEventListener("click", () => {
  if (running) {
    stopStream();
  } else {
    startStream();
  }
});

document.querySelectorAll(".speed-btn").forEach((button) => {
  button.addEventListener("click", () => {
    currentSpeed = button.dataset.speed;
    document.querySelectorAll(".speed-btn").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    if (running) startStream();
  });
});

document.querySelectorAll(".decision-buttons button").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!selectedUnit) return;
    const decision = button.dataset.decision;
    const reason = $("decisionReason").value;
    const response = await fetch("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ unit: selectedUnit, decision, reason }),
    });
    if (!response.ok) {
      $("agentLog").innerHTML = `<div class="log-entry crisis_detector">조치 요청 실패: ${response.status}</div>` + $("agentLog").innerHTML;
      return;
    }
    $("decisionReason").value = "";
    renderSnapshot(await response.json());
  });
});

loadInitialState().catch((error) => {
  $("agentLog").innerHTML = `<div class="log-entry crisis_detector">상태 파일을 불러오지 못했습니다: ${error.message}</div>`;
});

// 마크다운 초안 파서
function parseMarkdown(md) {
  if (!md) return "보고서 내용이 존재하지 않습니다.";
  let html = md;
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');
  html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
  html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

// 모달 제어 핸들러
$("closeModal").addEventListener("click", () => {
  $("reportModal").style.display = "none";
});

window.addEventListener("click", (event) => {
  if (event.target === $("reportModal")) {
    $("reportModal").style.display = "none";
  }
});

// 우측 사이드바 2대 메인 업무 목적 탭 전환 제어
(function initTabSelector() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTabId = tab.dataset.tab;
      
      // 모든 탭 버튼 active 클래스 해제 및 클릭된 탭에 추가
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      // 모든 tab-pane 비활성화 및 타겟 pane 활성화
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) targetPane.classList.add("active");
    });
  });
})();

// 상단 권한 모드 스위치 리스너 바인딩
(function initRoleSelector() {
  const roleButtons = document.querySelectorAll(".role-btn");
  roleButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      roleButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentRole = btn.dataset.role;

      // 역할 선택에 따라 유용하게 기본 탭 자동 스위칭 연동
      if (currentRole === "tech") {
        const tabBtn = document.querySelector('.tab-btn[data-tab="tab-control"]');
        if (tabBtn) tabBtn.click();
      } else if (currentRole === "supervisor") {
        const tabBtn = document.querySelector('.tab-btn[data-tab="tab-log"]');
        if (tabBtn) tabBtn.click();
      }
      
      // 즉각 정보 패널 의사결정 모듈 다시 그리기
      renderEngineDetail();
    });
  });
})();
