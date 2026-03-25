import ollama

def generate_reasoning(result):

    struct = result["structured_reasoning"]
    
    shap_a = [s for s in struct['drugA'].get('shap_insights', []) if not str(s).startswith("SHAP Error")]
    shap_b = [s for s in struct['drugB'].get('shap_insights', []) if not str(s).startswith("SHAP Error")]

    # Label which data source is driving the explanation
    shap_driven_a = len(shap_a) > 0
    shap_driven_b = len(shap_b) > 0

    prompt = f"""
You are an expert molecular toxicologist writing a unique, data-driven interpretation report.

STRICT RULES:
- Use ONLY the exact data provided below — do not add or invent groups, mechanisms, SHAP features, or atoms
- Your explanation MUST directly reference the specific group names, mechanism names, and SHAP substructures/atoms listed
- Do NOT write generic sentences like "this compound may cause toxicity" — cite the ACTUAL values
- If SHAP data is provided, you MUST name the specific substructure or atom and explain WHY it drives toxicity
- If SHAP data is empty, base the explanation only on the groups and mechanisms
- DO NOT include any notes, preambles, disclaimers, asterisks, or extra sections
- Output ONLY the three sections: Drug A, Drug B, Interaction Risk

DATA:
Drug A functional groups detected: {struct['drugA']['groups']}
Drug A toxicity mechanisms: {struct['drugA']['mechanisms']}
Drug A SHAP-identified key substructures/atoms: {shap_a if shap_driven_a else "None available"}

Drug B functional groups detected: {struct['drugB']['groups']}
Drug B toxicity mechanisms: {struct['drugB']['mechanisms']}
Drug B SHAP-identified key substructures/atoms: {shap_b if shap_driven_b else "None available"}

Interaction classification: {result['interaction']}

OUTPUT FORMAT (exactly this, no extras):

Drug A:
[Write 2-3 sentences. Name the exact groups from the list. If SHAP data exists, name the specific substructure or atom and explain its biological significance. State the mechanism by exact name.]

Drug B:
[Write 2-3 sentences. Name the exact groups from the list. If SHAP data exists, name the specific substructure or atom and explain its biological significance. State the mechanism by exact name.]

Interaction Risk:
[Write 2-3 sentences. Explain based on the interaction classification and how the mechanisms of Drug A and Drug B specifically interact or compound each other.]
"""
    try:
        response = ollama.generate(
            model="phi3",
            prompt=prompt,
            options={"temperature": 0.3}
        )

        raw = response.get("response", "").strip()

        # Extract only the 3 allowed sections — discard anything else (hallucinated sections, notes, etc.)
        import re
        sections = {}
        pattern = re.compile(
            r'(Drug A:|Drug B:|Interaction Risk:)(.*?)(?=Drug A:|Drug B:|Interaction Risk:|$)',
            re.DOTALL | re.IGNORECASE
        )
        for match in pattern.finditer(raw):
            key = match.group(1).strip().rstrip(":")
            body = match.group(2).strip()
            # Normalize key
            key_norm = key.lower().replace(" ", "_")
            sections[key_norm] = body

        parts = []
        if "drug_a" in sections:
            parts.append(f"Drug A:\n{sections['drug_a']}")
        if "drug_b" in sections:
            parts.append(f"Drug B:\n{sections['drug_b']}")
        if "interaction_risk" in sections:
            parts.append(f"Interaction Risk:\n{sections['interaction_risk']}")

        return "\n\n".join(parts).strip() if parts else raw

    except Exception as e:
        print(f"Ollama error in reasoning generation: {e}")
        # Fallback to rule-based Model Interpretability if LLM is unavailable
        explanation = "🕵️‍♂️ Model Interpretability Report:\n\n"
        
        # Drug A Explanation
        if struct["drugA"]["groups"]:
            groups_a = ", ".join(struct["drugA"]["groups"])
            mechs_a = ", ".join(struct["drugA"]["mechanisms"])
            explanation += f"🔹 Drug A contains [{groups_a}] structures, which are linked to: {mechs_a}.\n"
        else:
            explanation += "🔹 Drug A: No major toxic functional groups detected.\n"
            
        # Drug B Explanation
        if struct["drugB"]["groups"]:
            groups_b = ", ".join(struct["drugB"]["groups"])
            mechs_b = ", ".join(struct["drugB"]["mechanisms"])
            explanation += f"🔹 Drug B contains [{groups_b}] structures, which are linked to: {mechs_b}.\n"
        else:
            explanation += "🔹 Drug B: No major toxic functional groups detected.\n"
            
        # Interaction
        explanation += f"\n⚠️ Conclusion: {struct.get('interaction_reason', 'Review chemical mechanisms for overlap.')}"
        
        return explanation

def simplify_text(text):
    prompt = f"Explain this complex scientific toxicity explanation in very simple terms that an average 10-year-old or non-scientist patient can easily understand. Do not use complex jargon.\n\nText: {text}\n\nSimple explanation:"
    try:
        response = ollama.generate(
            model="phi3",
            prompt=prompt,
            options={"temperature": 0.3}
        )
        return response.get("response", "").strip()
    except Exception as e:
        print(f"Ollama error in simplification: {e}")
        return "Explanation simplification unavailable. Ensure Ollama is running locally."

def generate_clinical_report(result):
    struct = result.get("structured_reasoning", {})
    A_pred = result.get('drugA', {}).get('prediction')
    B_pred = result.get('drugB', {}).get('prediction')
    interaction = result.get('interaction')
    
    A_smiles = result.get('drugA', {}).get('smiles', 'Unknown')
    B_smiles = result.get('drugB', {}).get('smiles', 'Unknown')
    
    prompt = f"""
You are a Lead Clinical Pharmacologist and Toxicologist.
Generate a Comprehensive, Unique, and Distinct Clinical Toxicity Report for the following drug interaction prediction.
CRITICAL: Ensure this report varies significantly in phrasing, structure, and detail from any previous report you have written. Do not use the exact same templated sentences every time. Be creative, specific, and scientifically accurate.

DATA:
Drug A (SMILES: {A_smiles}) predicted: {A_pred}
Drug A mechanism: {struct.get('drugA', {}).get('mechanisms')}
Drug A key sub-structures (SHAP): {struct.get('drugA', {}).get('shap_insights', [])}

Drug B (SMILES: {B_smiles}) predicted: {B_pred}
Drug B mechanism: {struct.get('drugB', {}).get('mechanisms')}
Drug B key sub-structures (SHAP): {struct.get('drugB', {}).get('shap_insights', [])}

Interaction Risk: {interaction}

FORMAT REQUIRED (Use Markdown):
## 1. Executive Summary
Brief but unique summary of the danger and findings. Highlight the specific SMILES structures evaluated.
## 2. Chemical Profile & SHAP Importances
How the specific Substructures and Atoms identified by SHAP (mapped from the mathematical fingerprints) translate into biological danger.
## 3. Mechanism of Toxicity
Biological and molecular pathways specifically linked to these drugs.
## 4. Clinical Implications & Recommendations
What this means for a patient taking both drugs. Give tailored advice.

Ensure it reads as a highly professional medical report. Do NOT include generic conversational filler.
"""
    try:
        response = ollama.generate(
            model="phi3",
            prompt=prompt,
            options={"temperature": 0.8}
        )
        return response.get("response", "").strip()
    except Exception as e:
        print(f"Ollama error in report generation: {e}")
        return "Clinical report generation unavailable. Ensure Ollama is running locally."