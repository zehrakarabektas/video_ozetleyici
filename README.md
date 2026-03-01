## 🎥 AI Video & Audio Summarizer (FastAPI)
Bu proje; video ve ses dosyalarını (veya YouTube linklerini) Whisper kullanarak metne dönüştüren, ardından OpenAI (GPT-4o-mini) ve Anthropic (Claude 3.5 Haiku) modelleriyle eşzamanlı olarak analiz eden tam kapsamlı bir web uygulamasıdır. Sadece özet çıkarmakla kalmaz; içerikteki önemli tarihleri, sayısal verileri ve bölüm başlıklarını yapılandırılmış bir şekilde sunar.

## 🚀 Özellikler
* **Çoklu Kaynak Desteği:** Yerel dosya yükleme veya doğrudan video URL'si (YouTube vb.) üzerinden işlem yapabilme.
* **Whisper Transkripsiyon:** OpenAI'ın Whisper modeli ile yüksek doğrulukta ses-metin dönüşümü ve otomatik dil algılama.
* **Karşılaştırmalı Özetleme:** Aynı içerik için hem OpenAI hem de Anthropic modellerinden paralel sonuç alma.
* **Yapılandırılmış İçgörüler:**
   * **Topic Sentence:** İçeriğin ana temasını belirten tek cümlelik giriş.
   * **Section Summaries:** İçeriği mantıksal bölümlere ayırarak detaylı özetleme.
   * **Timeline:** Kronolojik olayların ve tarihlerin takibi.
   * **Quantitative Data:** Metindeki sayısal verilerin, oranların ve istatistiklerin ayıklanması.
* **Akıllı Metin Bölme:** Uzun içerikleri anlam kaybı yaşamadan (overlap tekniğiyle) parçalara ayırarak işleme.
* **Hata Yönetimi:** API limitlerine karşı otomatik "Retry" (yeniden deneme) ve "Backoff" mekanizması.

## 🛠 Teknik Yöntemler
* **Backend:** FastAPI (Python)
* **Transkripsiyon:** OpenAI Whisper (Base Model)
* **LLM API:** OpenAI SDK, Anthropic SDK
* **Video İşleme:** yt-dlp, FFmpeg
* **Asenkron Yapı:** Asyncio & Semaphores (Paralel istek kontrolü için)
* **Frontend:** HTML5, CSS3 (Glassmorphism), Bootstrap 5, Vanilla JS, Jinja2 Templates

## ⚙️ Uygulama Akışı
* **Giriş:** Kullanıcı bir dosya yükler veya video linki yapıştırır.
* **İşleme:** Sunucu, medya dosyasını indirir ve Whisper ile transkriptini oluşturur.
* **Bölümleme:** Uzun metinler "Overlap" (örtüşme) yöntemiyle parçalanarak bağlam kaybı önlenir.
* **Analiz:** Hazırlanan chunklar OpenAI ve Anthropic API'lerine paralel olarak gönderilir.
* **Sunum:** Gelen yapılandırılmış JSON verisi, frontend'de Bootstrap tabları ve akordeon yapısıyla kullanıcıya sunulur.

## 📦 Hızlı Kurulum
 **1.** Sisteminizde FFmpeg kurulu olduğundan emin olun.
 **2.** Bağımlılıkları yükleyin:  pip install -r requirements.txt
 **3.** .env dosyasını oluşturun:
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
 **4.** Başlatın: uvicorn main:app --reload

## 🖼️ Uygulama Görünümü
<img width="1694" height="876" alt="Ekran görüntüsü 2026-03-01 154350" src="https://github.com/user-attachments/assets/da08e4f5-2439-4dba-9dcb-5527040156f8" />

## 🤝 Proje Ekibi
* **Esengül TURAN** 
* **Zehra KARABEKTAŞ**
  
