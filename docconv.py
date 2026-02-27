import base64
from mistralai import Mistral

with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

b64 = base64.b64encode(pdf_bytes).decode()
client = Mistral(api_key="97ZQlsV45YrDusgZRwjArWGbh3nerFPb")

resp = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{b64}"
    },
    include_image_base64=True,
)

for page in resp.pages:
    print(page.markdown)

