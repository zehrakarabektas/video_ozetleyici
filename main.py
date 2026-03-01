import os
import json
import tempfile
import time
import random
import asyncio
import yt_dlp
from typing import List, Dict, Optional, Any, Callable, Tuple

import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from openai import OpenAI
from anthropic import Anthropic


# ── Ortam değişkenlerini yükle ─────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY .env dosyasında tanımlı değil!")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY .env dosyasında tanımlı değil!")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ── FastAPI uygulaması ─────────────────────────────────────
app = FastAPI(title="Video/Ses Özetleme API")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper modelini yükle (bir kere, global) 
whisper_model = whisper.load_model("base")
GLOBAL_PARALLEL_REQUESTS = int(os.getenv("GLOBAL_PARALLEL_REQUESTS", "2"))
PARALLEL_REQUEST_SEM = asyncio.Semaphore(GLOBAL_PARALLEL_REQUESTS)

# Retry ayarları
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))
BASE_BACKOFF = float(os.getenv("LLM_BASE_BACKOFF", "0.8"))  # saniye
MAX_BACKOFF = float(os.getenv("LLM_MAX_BACKOFF", "12.0"))


# Yardımcı fonksiyonlar
def download_audio_from_url(url: str) -> str:
    """
    Verilen URL'den sesi indirir ve geçici dosya yolunu döner.
    """
    temp_dir = tempfile.gettempdir()
    temp_base_name = f"download_{int(time.time())}_{random.randint(1000, 9999)}"
    output_template = os.path.join(temp_dir, temp_base_name)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template + '.%(ext)s', # Çıktı şablonu
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # yt-dlp işlemi bitince dosya .mp3 uzantısı alacak
        final_path = output_template + ".mp3"
        
        if not os.path.exists(final_path):
            raise RuntimeError("Dosya indirildi ama bulunamadı.")
            
        return final_path
    except Exception as e:
        raise RuntimeError(f"Video indirme hatası: {str(e)}")
    
def _is_rate_limit_error(e: Exception) -> bool:
    sc = getattr(e, "status_code", None) or getattr(e, "status", None)
    if sc == 429:
        return True
    msg = str(e).lower()
    return ("rate limit" in msg) or ("too many requests" in msg) or ("429" in msg)


def _is_transient_error(e: Exception) -> bool:
    # Ağ kopması, timeout, 5xx, geçici hatalar
    sc = getattr(e, "status_code", None) or getattr(e, "status", None)
    if isinstance(sc, int) and 500 <= sc <= 599:
        return True
    msg = str(e).lower()
    transient_keywords = ["timeout", "timed out", "temporarily", "connection", "connect", "overloaded", "unavailable"]
    return any(k in msg for k in transient_keywords)


def call_with_retry(fn: Callable[[], Any], label: str = "llm") -> Any:
    """
    Blocking fonksiyonlar için güvenli retry.
    Bu fonksiyon thread içinde çalıştırılacak.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e) or _is_transient_error(e):
                backoff = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, 0.35)
                sleep_s = backoff + jitter
                print(f"[WARN] {label} attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {sleep_s:.2f}s")
                time.sleep(sleep_s)
                continue
            raise
    raise RuntimeError(f"{label} failed after {MAX_RETRIES} retries: {last_err}")


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Model bazen JSON dışında metin ekleyebilir.
    Önce direkt parse, olmazsa ilk { ... } bloğunu ayıkla.
    """
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}


def transcribe_audio(path: str) -> Dict[str, Any]:
    """
    Verilen ses/video dosyasını Whisper ile metne çevirir.
    Dili otomatik algılar, metni, dili ve segment zamanlarını döner.
    """
    result = whisper_model.transcribe(path, language=None)  # auto language detect
    text = result.get("text", "")
    detected_language = result.get("language", "unknown")
    segments = result.get("segments", [])
    return {"text": text, "language": detected_language, "segments": segments}

#cumle bitmeden bolebiliyordu  chunk_text_with_overlap bu fonksiyonda o sorun giderildi.
def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """
    Zaman bilgisi yoksa uzun metni yaklaşık max_chars karakterlik parçalara böler.
    """
    chunks: List[str] = []
    current: List[str] = []

    sentences = text.split(". ")
    for sentence in sentences:
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence = sentence + "."

        current_len = sum(len(s) for s in current)
        if current_len + len(sentence) <= max_chars:
            current.append(sentence)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [sentence]

    if current:
        chunks.append(" ".join(current))

    return chunks

def chunk_text_with_overlap(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """
    Metni bölerken önceki parçadan 'overlap' kadar karakteri 
    bağlam kopmasın diye içeri dahil edildi.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + max_chars
        
        # Eğer sona gelmediysek, son boşluktan (kelime bitiminden) bölelim
        if end < text_len:
            # max_chars civarındaki son boşluğu bul
            split_point = text.rfind(" ", start, end)
            if split_point != -1:
                end = split_point
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Bir sonraki parça için 'overlap' kadar geri git
        start = end - overlap 
        
        # Sonsuz döngü koruması 
        if start >= end:
            start = end

    return chunks

def summarize_chunk(
    text: str,
    target_lang: str = "tr",
    source_lang: Optional[str] = None,
    mode: str = "chunk",
    provider: str = "openai",  # "openai" | "anthropic"
) -> str:
    """
    LLM ile özet üretir.
    mode="chunk": 1–2 paragraf
    mode="final": 3–4 paragraf (bütünlüklü)
    """
    lang_names = {"tr": "Türkçe", "en": "İngilizce"}
    target_name = lang_names.get(target_lang, target_lang)
    source_info = f"Kaynak metnin dili: {source_lang}. " if source_lang else ""

    base_rules = """
- Çıktıyı paragraf halinde yaz; madde işareti, tire veya numara kullanma.
- Anlam bütünlüğünü koru, gereksiz tekrar yapma.
- Cümleleri mutlaka tam bitmiş şekilde yaz (asla yarım bırakma).
""".strip()

    if mode == "chunk":
        extra_rules = """
- Özet 1–2 paragraf olsun.
- Sadece bu bölümün ana fikrini ve kritik noktalarını anlat.
- Paragrafları "\\n\\n" ile ayır.
""".strip()
        max_tokens = 500
    else:
        extra_rules = """
- Özet tam olarak 3 veya 4 paragraf olsun.
- Her paragraf en az 3–4 cümle içersin.
- Cümleler asla yarım kalmasın, mutlaka nokta ile bitsin.
- İçeriğin tamamının akışını, ana fikirlerini ve önemli noktalarını bütünlüklü şekilde özetle.
- Kendini tekrar etme; organize ve akıcı yaz.
""".strip()
        max_tokens = 1500

    user_prompt = f"""
{source_info}Aşağıdaki metni {target_name} dilinde özetle.

Kurallar:
{base_rules}

{extra_rules}

Metin:
{text}
""".strip()

    system_prompt = "Sen video özetleme konusunda uzman bir asistansın."

    if provider == "openai":
        def _call():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_completion_tokens=max_tokens,
            )
        resp = call_with_retry(_call, label="openai:summarize_chunk")
        return resp.choices[0].message.content.strip()

    elif provider == "anthropic":
        def _call():
            return anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=max_tokens,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        msg = call_with_retry(_call, label="anthropic:summarize_chunk")
        return "".join([b.text for b in msg.content if hasattr(b, "text")]).strip()

    else:
        raise ValueError("provider openai | anthropic olmalı")


def format_timestamp(seconds: float) -> str:
    total_seconds = int(round(seconds))
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m:02d}:{s:02d}"


def chunk_segments_with_timestamps(
    segments: List[Dict[str, Any]], max_chars: int = 1500
) -> List[Dict[str, Any]]:
    """
    Whisper segment listesini yaklaşık max_chars karakterlik chunk'lara böler.
    """
    chunks: List[Dict[str, Any]] = []
    current_text: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_len = 0

    for seg in segments:
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue

        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))

        if current_start is None:
            current_start = seg_start

        prospective_len = current_len + len(seg_text) + 1

        if prospective_len <= max_chars or current_len == 0:
            current_text.append(seg_text)
            current_len = prospective_len
            current_end = seg_end
        else:
            chunks.append(
                {"text": " ".join(current_text).strip(), "start": current_start, "end": current_end}
            )
            current_text = [seg_text]
            current_start = seg_start
            current_end = seg_end
            current_len = len(seg_text) + 1

    if current_text:
        chunks.append(
            {
                "text": " ".join(current_text).strip(),
                "start": current_start if current_start is not None else 0.0,
                "end": current_end if current_end is not None else 0.0,
            }
        )

    return chunks

def generate_insights(
    base_text: str,
    target_lang: str = "tr",
    source_lang: Optional[str] = None,
    provider: str = "openai",
) -> Dict[str, Any]:
    """
    Tek çağrıda yapılandırılmış içgörü üretir.
    Haiku ve GPT-4o uyumlu agresif prompt içerir.
    """
    
    target_name_en = "Turkish" if target_lang == "tr" else "English"
    source_info = f"The source text language code is '{source_lang}'." if source_lang else ""

    #  PROMPT GÜNCELLEMESİ 
    # nicel verileri cıkarması saglandı bazen onemsiz diye geciyordu ısrarcı ol komutu eklendi . halisinasyon için de rulesa if le baslayan komut eklendi
    prompt = f"""
You are an expert content analyst. Your task is to analyze the provided text and extract structured insights in valid JSON format.

{source_info}

IMPORTANT RULES:
- The **keys** of the JSON must be in English.
- The **values** MUST be written in **{target_name_en}**.
- Output ONLY valid JSON.
- If the text provided is mostly incomprehensible, noise, or too short to extract insights, return empty arrays for all fields and state 'Yetersiz İçerik' in the topic_sentence. 

EXTRACTION INSTRUCTIONS:

1. **topic_sentence**: 
   - Write a single, comprehensive introductory paragraph (1-2 sentences) summarizing the main theme.

2. **section_summaries**:
   - Divide the content into 4 to 8 logical sections.
   - `title`: A descriptive title (in {target_name_en}).
   - `content`: 2-3 detailed paragraphs (in {target_name_en}).

3. **timeline**:
   - Extract ALL chronological markers.
   - Look for: Years (1999), Dates, Centuries (20. yy), Periods, Relative times.
   - `period`: The time expression.
   - `event`: What happened (in {target_name_en}).
   - If absolutely no time reference is found, return [].

4. **quantitative_data**:
   - **BE AGGRESSIVE:** Extract ANY numerical fact found in the text. Do not skip small numbers.
   - **STRICT RULE:** BUT NEVER invent or infer information. Only extract numbers and dates that are explicitly stated in the text.
   - Look for: Percentages (%), Years used as counts, Currencies ($/TL), Counts (5 people), Durations (10 minutes), Rankings (1st place).
   - `metric`: What is the number about? (Write in {target_name_en}).
   - `value`: The number itself with its unit (e.g. "%80", "850 Yılı", "3. Kez").
   - If absolutely no numbers exist, return [].

OUTPUT JSON SCHEMA:
{{
  "topic_sentence": "String...",
  "section_summaries": [
    {{"title": "String...", "content": "String..."}}
  ],
  "timeline": [
    {{"period": "String...", "event": "String..."}}
  ],
  "quantitative_data": [
    {{"metric": "String...", "value": "String..."}}
  ]
}}

TEXT TO ANALYZE:
{base_text}
""".strip()

    system_prompt = "You are a helpful AI assistant that outputs strict JSON."

    if provider == "openai":
        def _call():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini", 
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, 
                max_completion_tokens=2000,
            )
        resp = call_with_retry(_call, label="openai:generate_insights")
        raw = resp.choices[0].message.content.strip()
        data = safe_json_loads(raw)

    elif provider == "anthropic":
        def _call():
            return anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022", 
                max_tokens=2000,
                temperature=0.1, 
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        msg = call_with_retry(_call, label="anthropic:generate_insights")
        raw = "".join([b.text for b in msg.content if hasattr(b, "text")]).strip()
        data = safe_json_loads(raw)

    else:
        raise ValueError("provider openai | anthropic olmalı")

    return {
        "topic_sentence": data.get("topic_sentence", "") or "",
        "section_summaries": data.get("section_summaries", []) or [],
        "timeline": data.get("timeline", []) or [],
        "quantitative_data": data.get("quantitative_data", []) or [],
    }

#eski turkce promtlu kod 
def generate_insights1(
    base_text: str,
    target_lang: str = "tr",
    source_lang: Optional[str] = None,
    provider: str = "openai",
) -> Dict[str, Any]:
    """
    Tek çağrıda şunları üretir:
    - topic_sentence (1–2 cümle)
    - section_summaries (4–8 başlık, her biri 2–3 paragraf)
    - timeline (kronolojik tablo)
    - quantitative_data (nicel veri tablosu)
    """
    lang_names = {"tr": "Türkçe", "en": "İngilizce"}
    target_name = lang_names.get(target_lang, target_lang)
    source_info = f"Kaynak metnin dili ISO kodu ile: {source_lang}. " if source_lang else ""

    prompt = f"""
{source_info}Aşağıdaki metin, bir video/ders/podcast içeriğinden alınmış transkript/özet parçalarının birleşimidir.
Görevin: SADECE bu metne dayanarak, {target_name} dilinde GEÇERLİ JSON üretmek.

ÇOK ÖNEMLİ KURALLAR:
- Kesinlikle uydurma bilgi, tarih, sayı, isim ekleme.
- Metinde açıkça yoksa timeline/quantitative_data içine koyma; boş liste [] döndür.
- Çıktı sadece JSON olsun. JSON dışında tek karakter yazma.

1) topic_sentence:
- İçeriğin ana temasını anlatan 1–2 cümlelik kısa paragraf.
- Genel ama net olsun.

2) section_summaries:
- 4–8 adet başlık üret (title).
- Başlıklar ve içerikleri verilen metinden çıkar.
- Başlıklar birbirini tekrar etmesin, farklı temaları kapsasın.
- Her başlık için content:
  - Tam olarak 2 veya 3 paragraf.
  - Madde işareti/numara yok.
  - Paragrafları "\\n\\n" ile ayır.
  - Gereksiz tekrar yapma.

3) timeline:
- Metinde tarih/dönem/yıl/önce-sonra gibi kronolojik işaretler VARSA çıkar.
- 4–10 satır üret.
- period: "12 Mart 2020", "05.06.1998",  "Mart 2020", "850","1450'ler", "20. yüzyıl", "günümüz", "10.000 yıl önce" gibi.
- event: 1–2 cümle.
- Metinde kronoloji yoksa [].

4) quantitative_data:
- Metinde açıkça geçen sayısal verileri çıkar (%, oran, kat, adet, süre, sıcaklık, para vb.).
- 3–10 satır üret.
- metric: kısa isim, value: metindeki değer (mümkünse aynı formatla).
- Metinde nicel veri yoksa [].

JSON ŞEMASI (aynen):
{{
  "topic_sentence": "....",
  "section_summaries": [
    {{"title": "....", "content": "Paragraf1\\n\\nParagraf2"}}
  ],
  "timeline": [
    {{"period": "....", "event": "...."}}
  ],
  "quantitative_data": [
    {{"metric": "....", "value": "...."}}
  ]
}}

Metin:
{base_text}
""".strip()

    system_prompt = "Sen video özetlerinden yapılandırılmış çıktı üreten uzman bir asistansın."

    if provider == "openai":
        def _call():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=1400,
            )
        resp = call_with_retry(_call, label="openai:generate_insights")
        raw = resp.choices[0].message.content.strip()
        data = safe_json_loads(raw)

    elif provider == "anthropic":
        def _call():
            return anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1400,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        msg = call_with_retry(_call, label="anthropic:generate_insights")
        raw = "".join([b.text for b in msg.content if hasattr(b, "text")]).strip()
        data = safe_json_loads(raw)

    else:
        raise ValueError("provider openai | anthropic olmalı")

    topic_sentence = data.get("topic_sentence", "") or ""
    section_summaries = data.get("section_summaries", []) or []
    timeline = data.get("timeline", []) or []
    quantitative_data = data.get("quantitative_data", []) or []

    if not isinstance(section_summaries, list):
        section_summaries = []
    if not isinstance(timeline, list):
        timeline = []
    if not isinstance(quantitative_data, list):
        quantitative_data = []

    return {
        "topic_sentence": topic_sentence,
        "section_summaries": section_summaries,
        "timeline": timeline,
        "quantitative_data": quantitative_data,
    }


def summarize_long_text(
    full_text: str,
    summary_lang: str,
    source_lang: Optional[str] = None,
    segments: Optional[List[Dict]] = None,
    provider: str = "openai",
) -> Dict[str, Any]:
    """
    Uzun metni chunk'lara böler, her birini özetler ve
    final özet + topic + başlık özetleri + timeline + nicel tablo üretir.
    """

    #  Metni,segmentleri böl
    if segments:
        raw_chunks = chunk_segments_with_timestamps(segments, max_chars=1500)
        chunks_text_only = [c["text"] for c in raw_chunks]
    else:
        #chunks_text_only = chunk_text(full_text, max_chars=1500)
        chunks_text_only = chunk_text_with_overlap(full_text, max_chars=1500, overlap=200)
        raw_chunks = [{"text": t, "start": None, "end": None} for t in chunks_text_only]

    #  Chunk özetleri
    chunk_summaries: List[Dict[str, Any]] = []
    chunk_summary_texts: List[str] = []

    for idx, ch in enumerate(raw_chunks, start=1):
        print(f"[INFO] ({provider}) Chunk {idx}/{len(raw_chunks)} özetleniyor...")
        s = summarize_chunk(
            ch["text"],
            target_lang=summary_lang,
            source_lang=source_lang,
            mode="chunk",
            provider=provider,
        )
        chunk_summary_texts.append(s)

        start = ch.get("start")
        end = ch.get("end")
        chunk_summaries.append(
            {
                "start": start,
                "end": end,
                "start_str": format_timestamp(start) if start is not None else None,
                "end_str": format_timestamp(end) if end is not None else None,
                "summary": s,
            }
        )

    joined_summaries = "\n\n".join(chunk_summary_texts)

    # Final özet
    final_summary = summarize_chunk(
        joined_summaries,
        target_lang=summary_lang,
        source_lang=source_lang,
        mode="final",
        provider=provider,
    )

    # Yapılandırılmış veri
    insights = generate_insights(
        joined_summaries,
        target_lang=summary_lang,
        source_lang=source_lang,
        provider=provider,
    )

    topic_sentence = insights["topic_sentence"]

    #  topic_sentence boşsa final özetten türet
    if not topic_sentence and final_summary:
        sentences = [s.strip() for s in final_summary.split(".") if s.strip()]
        if sentences:
            topic_sentence = ". ".join(sentences[:2])
            if not topic_sentence.endswith("."):
                topic_sentence += "."

    return {
        "chunks": chunks_text_only,
        "chunk_summaries": chunk_summaries,
        "final_summary": final_summary,
        "topic_sentence": topic_sentence,
        "section_summaries": insights["section_summaries"],
        "timeline": insights["timeline"],
        "quantitative_data": insights["quantitative_data"],
    }


async def save_upload_to_temp(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        return tmp.name


#API endpoint'leri

@app.get("/api-info")
async def api_info():
    return {
        "message": "Video/Ses Özetleme API'ye hoş geldiniz.",
        "usage": "POST /process ile 'file' alanında video/ses dosyası gönderin.",
        "parallel_requests_limit": GLOBAL_PARALLEL_REQUESTS,
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
async def web_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process_file(
    file: Optional[UploadFile] = File(None),  
    media_url: Optional[str] = Form(None),    
    summary_lang: str = Form("tr"),
):
    # Giriş Kontrolü/ Ya dosya ya da link olmalı
    if not file and not media_url:
        raise HTTPException(status_code=400, detail="Lütfen bir dosya yükleyin veya bir link girin.")

    summary_lang = (summary_lang or "tr").lower()
    if summary_lang not in {"tr", "en"}:
        summary_lang = "tr"

    temp_path = None
    filename_for_ui = "Bilinmiyor"

    try:
        # Kaynak Belirleme (Dosya mı Link mi?)
        if file:
            filename_for_ui = file.filename
            temp_path = await save_upload_to_temp(file)
        elif media_url:
            filename_for_ui = media_url
            temp_path = await asyncio.to_thread(download_audio_from_url, media_url)

        # Whisper Transkripsiyon 
        transcription = transcribe_audio(temp_path)
        transcript = transcription["text"]
        detected_language = transcription["language"]
        segments = transcription.get("segments", [])

        # LLM Özetleme 
        async with PARALLEL_REQUEST_SEM:
            openai_task = asyncio.to_thread(
                summarize_long_text,
                transcript,
                summary_lang=summary_lang,
                source_lang=detected_language,
                segments=segments,
                provider="openai",
            )
            anthropic_task = asyncio.to_thread(
                summarize_long_text,
                transcript,
                summary_lang=summary_lang,
                source_lang=detected_language,
                segments=segments,
                provider="anthropic",
            )

            summary_openai, summary_anthropic = await asyncio.gather(openai_task, anthropic_task)

    except Exception as e:
        print("[ERROR] /process failed:", repr(e))
        raise HTTPException(status_code=500, detail=f"İşleme sırasında hata oluştu: {str(e)}")
    
    finally:
        # Geçici dosyayı sil
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # Cevap Döndürme
    return {
        "filename": filename_for_ui,
        "transcript": transcript,
        "detected_language": detected_language,
        "summary_language": summary_lang,

        "openai": {
            "chunk_summaries": summary_openai["chunk_summaries"],
            "final_summary": summary_openai["final_summary"],
            "topic_sentence": summary_openai["topic_sentence"],
            "section_summaries": summary_openai["section_summaries"],
            "timeline": summary_openai["timeline"],
            "quantitative_data": summary_openai["quantitative_data"],
        },
        "anthropic": {
            "chunk_summaries": summary_anthropic["chunk_summaries"],
            "final_summary": summary_anthropic["final_summary"],
            "topic_sentence": summary_anthropic["topic_sentence"],
            "section_summaries": summary_anthropic["section_summaries"],
            "timeline": summary_anthropic["timeline"],
            "quantitative_data": summary_anthropic["quantitative_data"],
        },

        # Frontend uyumluluğu için (Fallback)
        "chunk_summaries": summary_openai["chunk_summaries"],
        "final_summary": summary_openai["final_summary"],
        "topic_sentence": summary_openai["topic_sentence"],
        "section_summaries": summary_openai["section_summaries"],
        "timeline": summary_openai["timeline"],
        "quantitative_data": summary_openai["quantitative_data"],
    }
