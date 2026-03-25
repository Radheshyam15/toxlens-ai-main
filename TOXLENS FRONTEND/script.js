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

  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
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

    document.getElementById('reasonBox').textContent = reason;

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