import base64
import requests
from pathlib import Path

img_path = Path(
    r"C:\Users\dylan\.cursor\projects\c-Users-dylan-Desktop-FML-CODE-PR3-Backend\assets"
    r"\c__Users_dylan_AppData_Roaming_Cursor_User_workspaceStorage_4b19a1b16d0f8f8b9701b1ddc20cfb05_images_image-0a1bff43-33f4-494e-9ef4-e78e1edcbe60.png"
)
b64 = base64.b64encode(img_path.read_bytes()).decode()
r = requests.post(
    "http://localhost:8000/process-look-base64",
    json={"image": f"data:image/png;base64,{b64}"},
    timeout=120,
)
r.raise_for_status()
item = r.json()["items"][0]
print(
    f"category={item.get('categoryGroup')} color={item.get('color')} "
    f"fabric={item.get('fabricGroup')} bottomLength={item.get('bottomLength')}"
)
