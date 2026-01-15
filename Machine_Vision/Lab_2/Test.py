import os, base64, re
from google import genai

def img_to_b64(path: str):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return b64, mime

def pick(ans: str):
    m = re.search(r"\b(Yes|No|Uncertain)\b", ans, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Uncertain"

def extract(report: str):
    # best-effort simple parsing
    a = pick(re.search(r"Image A.*?(Yes|No|Uncertain)", report, re.I | re.S).group(0)) if re.search(r"Image A.*?(Yes|No|Uncertain)", report, re.I | re.S) else "Uncertain"
    b = pick(re.search(r"Image B.*?(Yes|No|Uncertain)", report, re.I | re.S).group(0)) if re.search(r"Image B.*?(Yes|No|Uncertain)", report, re.I | re.S) else "Uncertain"
    sev = re.search(r"(Which image.*?)(\n|$)", report, re.I)
    return a, b, sev.group(1).strip() if sev else None

def compare_mri(api_key: str, imageA: str, imageB: str):
    client = genai.Client(api_key=api_key)
    a_b64, a_mime = img_to_b64(imageA)
    b_b64, b_mime = img_to_b64(imageB)

    prompt = """
1. Tumor Presence:
   - Does Image A show a brain tumor? (Yes / No / Uncertain)
   - Does Image B show a brain tumor? (Yes / No / Uncertain)
2. If tumor present:
   - Location (relative)
   - Size (relative)
   - Shape and intensity
3. Simple Treatment Plan
4. Comparison:
   - Which image shows more severe abnormality?

IMPORTANT:
- Answer in clear numbered sections 1-4.
- Keep it medical-style but simple.
- Add a short disclaimer: "Educational use only."
""".strip()

    resp = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": a_mime, "data": a_b64}},
                {"inline_data": {"mime_type": b_mime, "data": b_b64}},
            ]
        }]
    )

    report = resp.text if hasattr(resp, "text") else str(resp)
    a_ans, b_ans, sev = extract(report)

    print("\n========== STRUCTURED OUTPUT ==========")
    print(f"Image A Tumor? {a_ans}")
    print(f"Image B Tumor? {b_ans}")
    if sev: print(f"Severity: {sev}")
    print("======================================\n")

    print("========== FULL AI REPORT ==========")
    print(report)
    print("====================================")

if __name__ == "__main__":
    API_KEY = "AIzaSyCnOoXZjvVx1NKEGgB_R_sISz-dHphnUSM"
    compare_mri(API_KEY, "NoTumour.jpg", "Tumour.jpg")
