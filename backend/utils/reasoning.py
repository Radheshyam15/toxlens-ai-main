import requests

def generate_reasoning(result):

    struct = result["structured_reasoning"]

    prompt = f"""
You are a scientific formatter.

STRICT RULES (MANDATORY):
- ONLY use the exact functional groups and mechanisms provided
- DO NOT add any new groups (e.g., amine if not listed)
- DO NOT infer or assume anything
- DO NOT expand beyond given data
- If unsure, DO NOT mention it

DATA:
Drug A groups: {struct['drugA']['groups']}
Mechanisms: {struct['drugA']['mechanisms']}

Drug B groups: {struct['drugB']['groups']}
Mechanisms: {struct['drugB']['mechanisms']}

Interaction: {result['interaction']}

Task:
Write a precise, professional explanation in 2-3 sentences using ONLY the above data.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=60
        )

        return response.json().get("response", "").strip()

    except:
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