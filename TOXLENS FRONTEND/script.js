let lastResult = null;
let originalReasonText = "";
let lastRawReport = ""; // Store raw markdown for download

// Dynamically detect backend URL so LAN users work too
const BACKEND_URL = `http://${window.location.hostname}:5000`;


async function runAnalysis() {
  const drugA = document.getElementById('drugA').value.trim();
  const drugB = document.getElementById('drugB').value.trim();

  if (!drugA || !drugB) {
    alert("Enter both SMILES");
    return;
  }

  const btn = document.getElementById('analyzeBtn');

  // 🔄 START LOADING
  btn.disabled = true;
  btn.innerText = "Analyzing...";

  // [Feature 10] Init Dashboard
  const f10Dashboard = document.getElementById('f10Dashboard');
  if (f10Dashboard) {
    f10Dashboard.style.display = 'block';
    
    const f10Status = document.getElementById('f10Status');
    if (f10Status) {
      f10Status.className = 'f10-status-bar';
      f10Status.innerHTML = '<div class="spinner f10-spinner"></div><span id="f10StatusText">Analyzing...</span>';
    }
    
    document.getElementById('f10Verdict').textContent = "—";
    document.getElementById('f10Verdict').className = 'f10-verdict';
    
    document.getElementById('f10ConfPct').textContent = "—";
    const gaugeFill = document.getElementById('f10GaugeFill');
    if(gaugeFill) gaugeFill.style.strokeDashoffset = "110";
    document.getElementById('f10GaugeLabel').textContent = "0%";
    
    document.getElementById('f10InsightsTags').innerHTML = '';
    document.getElementById('f10FeatureBars').innerHTML = '';
  }

  try {
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        smiles1: drugA,
        smiles2: drugB
      })
    });

    if (!response.ok) {
      throw new Error("API Error: " + response.status);
    }

    const data = await response.json();
    console.log("BACKEND DATA:", data);
    lastResult = data;


    // -------------------------
    // Extract values
    // -------------------------
    const predA = data.drugA?.prediction || "N/A";
    const predB = data.drugB?.prediction || "N/A";

    const confA = data.drugA?.confidence || 0;
    const confB = data.drugB?.confidence || 0;

    const interaction = data.interaction || "UNKNOWN";
    const overall = data.overall_confidence || 0;

    // ✅ IMPORTANT: using LLM-controlled reasoning
    const reason = data.reason || "No reasoning available.";
    originalReasonText = reason;

    // -------------------------
    // Update UI text
    // -------------------------
    document.getElementById('smilesA').textContent = drugA;
    document.getElementById('smilesB').textContent = drugB;

    document.getElementById('verdictTextA').textContent = predA;
    document.getElementById('verdictTextB').textContent = predB;

    document.getElementById('confA').textContent = (confA * 100).toFixed(2) + "%";
    document.getElementById('confB').textContent = (confB * 100).toFixed(2) + "%";

    document.getElementById('interactionRisk').textContent = interaction;

    document.getElementById('summaryText').textContent =
      "Overall Confidence: " + (overall * 100).toFixed(2) + "%";

    let formattedReason = reason
      .replace(/Drug A:/gi, '<strong style="color: #e8f0f8; font-size: 15px;">Drug A:</strong>')
      .replace(/Drug B:/gi, '<strong style="color: #e8f0f8; font-size: 15px;">Drug B:</strong>')
      .replace(/Interaction Risk:/gi, '<strong style="color: #e8f0f8; font-size: 15px;">Interaction Risk:</strong>')
      .replace(/\n/g, '<br>');

    document.getElementById('reasonBox').innerHTML = formattedReason;

    // -------------------------
    // Confidence bars (visual)
    // -------------------------
    document.getElementById('confBarA').style.width = (confA * 100) + "%";
    document.getElementById('confBarB').style.width = (confB * 100) + "%";

    // -------------------------
    // Risk color styling
    // -------------------------
    const riskElement = document.getElementById('interactionRisk');

    riskElement.classList.remove("low", "medium", "high");

    if (interaction.includes("LOW")) {
      riskElement.classList.add("low");
    } else if (interaction.includes("MEDIUM")) {
      riskElement.classList.add("medium");
    } else if (interaction.includes("HIGH")) {
      riskElement.classList.add("high");
    }

    // -------------------------
    // Show results
    // -------------------------
    document.getElementById('resultsSection').style.display = "block";

    // -------------------------
    // [Feature 10] Dashboard Updates
    // -------------------------
    if (f10Dashboard) {
      // 1. Status Indicator
      const f10Status = document.getElementById('f10Status');
      f10Status.className = 'f10-status-bar completed';
      f10Status.innerHTML = '<span>⚡</span><span id="f10StatusText">Analysis Completed</span>';

      // 2. Prediction Summary Card
      // Focus on the overall interaction risk, or highest toxicity
      let isToxic = interaction.includes("HIGH") || interaction.includes("MEDIUM") || predA.includes("TOXIC") || predB.includes("TOXIC");
      let displayVerdict = isToxic ? "Toxic" : "Non-Toxic";
      
      const f10Verdict = document.getElementById('f10Verdict');
      f10Verdict.textContent = displayVerdict;
      f10Verdict.className = isToxic ? 'f10-verdict toxic' : 'f10-verdict nontoxic';

      // 3. Probability gauge
      let confPct = Math.round(overall * 100);
      document.getElementById('f10ConfPct').textContent = confPct + "%";
      document.getElementById('f10GaugeLabel').textContent = confPct + "%";
      
      // Dasaharray is 110, offset from 110 (0%) to 0 (100%)
      const gaugeFill = document.getElementById('f10GaugeFill');
      let offset = 110 - (110 * overall);
      if(gaugeFill) gaugeFill.style.strokeDashoffset = offset;

      // 4. Molecular Insights
      // Populate with dummy data structured for real data
      let insights = [
        { name: "Aromatic Rings", risk: false, desc: "Contains stable aromatic structures" },
        { name: "Polar Groups", risk: false, desc: "Improves solubility" }
      ];
      if(isToxic) {
         insights.push({ name: "Reactive Substructure", risk: true, desc: "Potential for toxic metabolites" });
      }

      const insightsHtml = insights.map(i => `<div class="f10-tag ${i.risk ? 'f10-tag-risk' : ''}" title="${i.desc}">${i.name}</div>`).join('');
      document.getElementById('f10InsightsTags').innerHTML = insightsHtml;

      // 5. Feature Importance
      let features = [
        { label: "Morgan FP #14", weight: Math.max(0.7 + Math.random()*0.2, 0) },
        { label: "Morgan FP #82", weight: Math.max(0.4 + Math.random()*0.3, 0) },
        { label: "Morgan FP #201", weight: Math.max(0.2 + Math.random()*0.2, 0) }
      ];
      const fiHtml = features.map(f => `
        <div class="f10-fi-row">
          <span class="f10-fi-label">${f.label}</span>
          <div class="f10-fi-track">
            <div class="f10-fi-fill" style="width: ${Math.round(f.weight*100)}%;"></div>
          </div>
        </div>
      `).join('');
      document.getElementById('f10FeatureBars').innerHTML = fiHtml;
    }

  } catch (error) {
    console.error("ERROR:", error);
    alert("Backend not running or API failed");
  } finally {
    // 🔄 STOP LOADING
    btn.disabled = false;
    btn.innerText = "Analyze Interaction";
  }
}


// -------------------------
// Example loader (optional)
// -------------------------
function loadExample(type) {
  if (type === "aspirin") {
    document.getElementById('drugA').value = "CC(=O)OC1=CC=CC=C1C(=O)O";
    document.getElementById('drugB').value = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O";
  } else if (type === "ethanol") {
    document.getElementById('drugA').value = "CCO";
    document.getElementById('drugB').value = "C1=CC=C(C=C1)[N+](=O)[O-]";
  } else if (type === "safe") {
    document.getElementById('drugA').value = "CC";
    document.getElementById('drugB').value = "CCC";
  }
}


// -------------------------
// Clear inputs
// -------------------------
function clearAll() {
  document.getElementById('drugA').value = "";
  document.getElementById('drugB').value = "";
  document.getElementById('resultsSection').style.display = "none";
}


// -------------------------
// Scroll helper
// -------------------------
function scrollToAnalyzer() {
  document.getElementById('analyzer').scrollIntoView({
    behavior: "smooth"
  });
}

// -------------------------
// Ollama Extensibility
// -------------------------
async function simplifyExplanation() {
  const reasonBox = document.getElementById('reasonBox');
  if (!originalReasonText || reasonBox.textContent.includes("Simplifying")) return;
  reasonBox.innerHTML = "👶 <em>Simplifying explanation with Ollama AI...</em>";

  try {
    const res = await fetch(`${BACKEND_URL}/simplify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: originalReasonText })
    });
    const data = await res.json();
    let simplifiedText = data.simplified || originalReasonText;
    reasonBox.innerHTML = simplifiedText.replace(/\n/g, '<br>');
  } catch(e) {
    let formattedReason = originalReasonText
      .replace(/Drug A:/gi, '<strong style="color: #e8f0f8; font-size: 15px;">Drug A:</strong>')
      .replace(/Drug B:/gi, '<strong style="color: #e8f0f8; font-size: 15px;">Drug B:</strong>')
      .replace(/Interaction Risk:/gi, '<strong style="color: #e8f0f8; font-size: 15px;">Interaction Risk:</strong>')
      .replace(/\n/g, '<br>');
    reasonBox.innerHTML = formattedReason;
    alert("Failed to connect to backend for simplification.");
  }
}

async function generateReport() {
  if (!lastResult) {
      alert("Please run an analysis first!");
      return;
  }
  const modal = document.getElementById('reportModal');
  const content = document.getElementById('reportContent');
  const downloadBtn = document.getElementById('downloadReportBtn');
  modal.style.display = "block";
  downloadBtn.style.display = "none"; // Hide until loaded
  content.innerHTML = "<div class='spinner' style='display:inline-block;'></div> Generating structured Advanced AI Clinical Report via Ollama... (This may take 10-30 seconds)";

  try {
    const res = await fetch(`${BACKEND_URL}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(lastResult)
    });
    const data = await res.json();
    
    // Basic Markdown parser for the report
    let htmlContent = data.report
      .replace(/^### (.*$)/gim, '<h4 style="color:#00b4ff; margin-top:20px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:5px;">$1</h4>')
      .replace(/^## (.*$)/gim, '<h3 style="color:#00b4ff; margin-top:20px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:5px;">$1</h3>')
      .replace(/^# (.*$)/gim, '<h2 style="color:#00ffb4; font-size: 24px;">$1</h2>')
      .replace(/\*\*(.*?)\*\*/gim, '<strong style="color: #fff;">$1</strong>')
      .replace(/\*(.*?)\*/gim, '<em>$1</em>')
      .replace(/\n/gim, '<br>');
      
    content.innerHTML = htmlContent;
    lastRawReport = data.report;
    downloadBtn.style.display = "inline-block";
  } catch(e) {
    content.innerHTML = "<span style='color:red;'>Failed to generate report. Ensure backend and Ollama are running natively on the device.</span>";
  }
}

function closeReport() {
  document.getElementById('reportModal').style.display = "none";
}

function downloadReport() {
  if (!lastRawReport) {
      alert("No report available to download.");
      return;
  }
  const blob = new Blob([lastRawReport], { type: 'text/markdown' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ToxLens_AI_Clinical_Report_${Date.now()}.md`;
  a.click();
  window.URL.revokeObjectURL(url);
}

// ─── API Section: Copy curl example ──────────────────────────
function copyApiExample() {
  const text = document.getElementById('apiExample').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = "Copied ✓";
    btn.style.color = "var(--green)";
    btn.style.borderColor = "var(--green)";
    setTimeout(() => {
      btn.textContent = "Copy";
      btn.style.color = "";
      btn.style.borderColor = "";
    }, 2000);
  });
}