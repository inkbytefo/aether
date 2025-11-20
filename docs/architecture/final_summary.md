# AETHER-1: Nihai Mimari Özeti (V 1.0)
## Autonomous Episodic Thinking & Heuristic Reasoning Core

**AETHER-1**, mevcut LLM (Büyük Dil Modeli) paradigmasının **statik hafıza** ve **muhakeme eksikliği** kısıtlamalarını aşmayı hedefleyen, **hibrit (nöro-sembolik)** bir AGI araştırma mimarisidir.

### 1. Proje Vizyonu ve Felsefesi
Modelin hedefi, sadece bir sonraki **kelimeyi** tahmin etmek yerine, bir sonraki **durumu** (Next State Prediction) tahmin eden ve **deneyimden sürekli öğrenen** (Plasticity) bir bilişsel çekirdek inşa etmektir. Cevap hızından ziyade **mantık tutarlılığına** odaklanılmıştır.

### 2. Üç Ana Mimari Sütun (The Core Pillars)

AETHER-1, üç uzmanlaşmış modülün dinamik entegrasyonu üzerine kurulmuştur:

#### A. Omurga: Verimli Sürekli Akış (SSM Backbone)
* **Teknoloji:** **Mamba (Selective State Space Model)** mimarisi kullanılır.
* **Amaç:** Klasik Transformer'ların $O(N^2)$ karmaşıklığını, **doğrusal ($O(N)$)** karmaşıklığa indirgeyerek sınırsız (infinite) bağlam (context) işleme potansiyeli yaratmak.
* **Sonuç:** Model, binlerce sayfalık metni tek bir "Durum Vektöründe" (State Vector) sıkıştırıp saklayabilir, yani konuşmayı sıfırlama ihtiyacı ortadan kalkar.

#### B. Hafıza: Dinamik ve Plastik Öğrenme (Fast Weights)
* **Teknoloji:** **Hebbian Learning (Fast Weights)** entegrasyonu.
* **Amaç:** Modelin, temel dil bilgisini (Slow Weights) korurken, kullanıcı ile etkileşim sırasında **anlık olarak yeni bilgi öğrenmesini** sağlamak.
* **Mekanizma:** Her SSM bloğuna, o anki girdi ve çıktıya bağlı olarak dinamik olarak güncellenen ek bir "Hızlı Ağırlık Matrisi" ($A_{fast}$) eklenir. Bu, modelin bir daha eğitilmeye ihtiyaç duymadan yeni bilgileri **nöral düzeyde** kaydetmesini sağlar.

#### C. Bilişsel Çekirdek: System 2 Muhakemesi
* **Teknoloji:** **Tree-of-Thoughts (ToT)** arama algoritması.
* **Amaç:** Zor ve mantık gerektiren sorular geldiğinde, modelin "refleks" (System 1) cevabını durdurup, **çözüm yollarını simüle etmeye** zorlamak.
* **İşleyiş:** Model çıktı üretmeden önce arka planda 3-5 olası mantık adımını dener, bu adımları puanlayan bir **Değerlendirici (Evaluator)** tarafından en yüksek puanlı yolu seçer ve sadece nihai sonucu dışarıya aktarır.

### 3. Bilişsel Modülün Çalışma Prensibi (The Reasoning Loop)

AETHER-1'in düşünme yeteneği, eğitim ve çıkarım (inference) aşamalarında özel bir veri manipülasyonu ile sağlanır:

* **Thought Marker Injection (Düşünce İşaretçileri):** Eğitim verisi, mantık zincirini gösteren özel etiketlerle (`<think>...</think>`) zenginleştirilmiştir. Bu, modelin "mantık yürütme sürecini" bir çıktı olarak görmesini ve taklit etmesini sağlar.
* **Döngüsel Çalışma:** Çıkarım sırasında, `reasoning.py` içindeki algoritmamız, modelin SSM durumunu koruyarak, düşünce ağacında dal budaklanmasını (branching) sağlar. Hata riski yüksek olan yollar **budanır**, sadece mantıksal olarak tutarlı yollar devam eder.

### 4. Eğitim ve Test Stratejisi
* **Müfredat (Curriculum Learning):** Model, Dil (TinyStories) $\rightarrow$ Mantık (Kod ve Matematik) $\rightarrow$ Hibrit Entegrasyon (Plastik Hafıza Aktivasyonu) sıralamasıyla eğitilir.
* **Başarı Metriği:** Başarı, Perplexity (PPL) düşüklüğünden ziyade, **ARC (Abstraction and Reasoning Corpus) Testi** gibi soyut mantık problemlerini çözme yeteneği ile ölçülecektir.

### 5. Mevcut Ekosistemle Farkı
| Özellik | Mevcut LLM'ler (GPT-4, Llama) | AETHER-1 |
| :--- | :--- | :--- |
| **Hafıza Yönetimi** | KV-Cache, pahalı, sınırlı. | SSM State, sürekli, verimli. |
| **Öğrenme** | Statik (Eğitim biter). | **Dinamik (Konuşurken öğrenir).** |
| **Muhakeme** | System 1 (Refleksif). | **System 2 (Tefekkür/Planlama).** |
| **Halüsinasyon** | Yüksek riskli. | Düşünme döngüsü sayesinde **çok düşük riskli.** |
