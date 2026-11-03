# NATEQ — Multi-Book RAG

النسخة دي بتخلي الـRAG **عام لأي كتاب** بدل ربطه بكتاب صلاح الدين. ممكن تضيف كتاب PDF أو JSON مجهز، وتختار الكتاب وقت السؤال من غير تعديل كود الـRetrieval أو الـPrompt.

**الكود والـComments والـDocstrings كلها بالإنجليزي.** الشرح هنا بالعربي.

## 1. الإعداد لأول مرة

من داخل فولدر المشروع في VS Code أو PowerShell:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

افتح `.env` وحط `GROQ_API_KEY`. الموديل `BAAI/bge-m3` والـCross-Encoder هيتحملوا أول مرة فقط لو مش موجودين عندك. Groq بيشتغل بالـAPI، مش بيتنزل على جهازك.

## 2. قائمة الكتب

```powershell
python run.py --list-books
```

نسخة صلاح الدين موجودة جاهزة داخل `data/books/salah_al_din` وتشمل PDF الأصلي وملف JSON المستخرج منه سابقًا. **مش محتاج تعيد استخراج PDF صلاح الدين**.

## 3. إضافة كتاب جديد

مثال بعد ما تكون حصلت بنفسك على نسخة PDF من كتاب «الرحيق المختوم»:

```powershell
python run.py --add-book "D:\Books\al_raheeq.pdf" --book al_raheeq --title "الرحيق المختوم"
```

الملف بيتنسخ تلقائيًا إلى `data/books/al_raheeq/book.pdf`. أول سؤال عليه بيعمل استخراجًا للنص (لو PDF نصي) وبيبني فهرس مستقل للكتاب. الاستدعاءات التالية بتعيد استخدام الفهرس المحفوظ.

لو عندك JSON مستخرج وجاهز بشكل سليم، ممكن تضيفه بدل PDF أو معاه:

```powershell
python run.py --prepared-json "D:\Books\raheeq_extracted.json" --book al_raheeq --title "الرحيق المختوم"
```

**تنبيه:** تسجيل الكتاب بيحصل مرة واحدة. لو `al_raheeq` مسجل بالفعل، ما تضيفوش بنفس المعرّف. كما إن استخراج PDF النصي مش OCR؛ الصور الممسوحة ضوئيًا والخطوط غير المعتادة ممكن تحتاج استخراج نص موثوق يدويًا أو عن طريق OCR منفصل.

## 4. اختيار الكتاب عند السؤال

```powershell
python run.py --book salah_al_din --question "أين انتقل صلاح الدين مع والده؟"
python run.py --book al_raheeq --question "ما أبرز أحداث الهجرة؟"
```

ولتغيير الكتاب الافتراضي لو مش عايز تكتب `--book` في كل مرة:

```powershell
python run.py --set-default-book al_raheeq
python run.py --question "ما أبرز أحداث الهجرة؟"
```

## 5. Jupyter / FastAPI

افتح `NATEQ_Quickstart.ipynb` أو استخدم الكود ده:

```python
from src.application.book_manager import BookManager

manager = BookManager(data_dir="data")
print(manager.list_books())

result = manager.ask("أين ولد صلاح الدين؟", book_id="salah_al_din")
print(result["answer_with_sources"])

# To switch books, change only book_id.
result = manager.ask("ما أبرز أحداث الهجرة؟", book_id="al_raheeq")
print(result["answer_with_sources"])
```

الـResponse بتحتفظ بنفس الحقول المستخدمة في الـFrontend: `answer`, `status`, `citations`, `sources`, `cited_sources`, `answer_with_sources`, `citation_audit`, `timings`. وضفنا `book_id` و`book_title` عشان الواجهة تبيّن الكتاب الحالي.

## 6. Architecture

`BookRegistry → Book Pages → Parent-Child Chunking → Chroma + Arabic BM25 → Cross-Encoder → Focused Parent → Groq + LangChain Prompt → Structural Citation Validation → Answer + Sources`

- الـParents: 2000 حرف، overlap: 200.
- الـChildren: 400 حرف، overlap: 80.
- Dense Top 20 + Arabic BM25 حتى 5 Candidates إضافية.
- Fusion: 35% Retrieval و65% Cross-Encoder.
- Focused Parent Context: 4500 حرف من خمسة Parents كحد أقصى.
- **كل كتاب له Chroma Index وArabic BM25 وParents وChildren مستقلين تمامًا.** الفهارس موجودة في `indexes/<book_id>/<version>/`.
- الـEmbedding والـCross-Encoder وGroq Client ممكن نعيد استخدامهم بين الكتب في نفس Python Process بدل تحميلهم من جديد.
- مصدر الإجابة بيعرض اسم الكتاب ورقم الصفحة والنص المستشهد به، ومفيش مصادر من كتب أخرى داخل نفس السؤال.

**مهم:** رقم الصفحة هو رقم صفحة ملف الـPDF فعليًا، وقد يختلف عن الرقم المطبوع أسفل صفحات الكتاب. ونجاح Structural Citation Validation معناه إن الـReference ID صالح، مش ضمان دلالي إن المصدر يثبت كل ادعاء.

## 7. الاختبارات

```powershell
python -m pytest -q
```

الاختبارات بتغطي إعدادات الـRAG، وأمان IDs الكتب، وتبديل الكتب والفهارس بدون خلط المصادر، وجودة استخراج العربية من صفحة فعلية، وشكل الاستشهادات. لكن تشغيل Groq وBGE-M3 وChroma **End-to-End** محتاج تجربة على جهازك بعد تثبيت المكتبات والموديلات.
