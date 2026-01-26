# ML
# FindMyLook - ML Service 🚀

שירות ה-ML (Machine Learning) של פרויקט FindMyLook. 
[cite_start]השירות מבוסס Python ו-FastAPI ותפקידו לנתח תמונות "לוק" המועלות על ידי המשתמש, לזהות פריטי לבוש (Detection) ולהפיק וקטורים (Embeddings) לחיפוש ויזואלי בבסיס הנתונים[cite: 23, 38].

## 🛠 טכנולוגיות בשימוש
- [cite_start]**FastAPI**: שרת ה-API של השירות.
- [cite_start]**YOLOv8 (Ultralytics)**: מודל לזיהוי אובייקטים (Object Detection)[cite: 127].
- [cite_start]**CLIP / Sentence-Transformers**: מודל להפקת ייצוגים ויזואליים (Image Embeddings)[cite: 129].
- **Pillow (PIL)**: ספריית עיבוד תמונות לגזירה ועיבוד מקדים.

## 🚀 הוראות הקמה (Setup)

לפני תחילת העבודה, ודאי שמותקן אצלך Python 3.8 ומעלה.

### 1. שכפול המאגר (Clone)
```bash
git clone <repository_url>
cd ML
# יצירת הסביבה
python -m venv venv
# הפעלה (Windows - Git Bash)
source venv/Scripts/activate
# הפעלה (Windows - PowerShell)
.\venv\Scripts\activate
pip install -r requirements.txt

python main.py הרצה

📂 מבנה הפרויקט המומלץ

main.py: נקודת הכניסה לשירות והגדרת ה-Endpoints.
models/: תיקייה לאחסון קבצי ה-Weights של המודלים (קבצי .pt).
requirements.txt: רשימת הספריות להתקנה.
venv/: סביבה וירטואלית (נמצא ב-.gitignore).
