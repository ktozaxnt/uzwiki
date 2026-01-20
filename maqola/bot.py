#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
========================================
UZBEK WIKIPEDIA BOT - POLISH CITIES
========================================
Version: 5.0 (ZERO-COST ARCHITECT)
Purpose: Mass-generate geography articles using FREE AI models
Target: uz.wikipedia.org
API: OpenRouter (Free Models)
Author: Senior AI DevOps & Bot Architect
License: MIT
========================================
"""

import pywikibot
from pywikibot import pagegenerators
import os
import glob
import sys
import re
import logging
import time
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from pathlib import Path

# OpenAI SDK
from openai import OpenAI

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
class Config:
    """Centralized configuration with validation"""
    SIMULATION_OR_DRAFT_MODE = True  # Set to True for testing
    BOT_USERNAME = "DanikBotUZ"
    
    COUNTRY_QID = 'Q36'  # Poland
    COUNTRY_NAME = 'Polsha'
    MAX_ARTICLES = 1
    
    # OpenRouter Configuration
    # Use the provided OPENAI_API_KEY which is actually the OpenRouter key in this context
    OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENROUTER_ENDPOINT = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    
    # AVAILABLE MODELS FALLBACK LIST
    # Note: Using internal models as external OpenRouter access is restricted in this environment
    FREE_MODELS = [
        "gpt-4.1-mini",
        "gemini-2.5-flash",
        "gpt-4.1-nano"
    ]
    
    # Model Parameters
    TEMPERATURE = 0.1 
    TOP_P = 0.95
    MAX_TOKENS = 4000
    
    # Retry Configuration
    MAX_RETRIES_PER_MODEL = 2
    INITIAL_RETRY_DELAY = 5
    
    # Bot Behavior
    EDIT_DELAY = 5
    REQUEST_TIMEOUT = 60 # Strict timeout for AI requests
    
    # Logging
    LOG_DIR = Path("logs")
    LOG_FILE = LOG_DIR / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    @classmethod
    def validate(cls):
        """Validate configuration before startup"""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("⚠️ CRITICAL: No OpenRouter API token configured")
        cls.LOG_DIR.mkdir(exist_ok=True)

# ==========================================
# 📊 LOGGING SYSTEM
# ==========================================
class BotLogger:
    """Dual-output logger (console + file)"""
    
    def __init__(self):
        Config.LOG_DIR.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger = logging.getLogger('UzbekWikiBot')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def debug(self, msg): self.logger.debug(msg)
    def critical(self, msg): self.logger.critical(msg)

logger = BotLogger()

# ==========================================
# 🧹 TEXT SANITIZER (ROBUST)
# ==========================================
class TextSanitizer:
    """Converts AI output to clean MediaWiki syntax - AGGRESSIVE ARTIFACT REMOVAL"""
    
    @staticmethod
    def clean(text: str) -> str:
        if not text: return ""
        
        # 1. Remove thinking tags (common in DeepSeek R1)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 2. Remove triple-quote blocks and code fences
        text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
        text = re.sub(r'```[a-z]*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```', '', text)
        
        # 3. Remove preamble chatter
        meta_patterns = [
            r'^(Here is|Here\'s|Below is|The following is|Translated).*?:?\s*',
            r'^(Вот|Это|Ниже|Следующ).*?:?\s*',
            r'^\s*---+\s*$',
            r'^RAW WIKITEXT:.*$',
            r'^OUTPUT:.*$',
            r'^BEGIN UZBEK TRANSLATION:.*$',
            r'^TRANSLATION:.*$',
        ]
        for pattern in meta_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 4. Remove redundant infoboxes (we build our own)
        for _ in range(3):
            text = re.sub(r'\{\{(?:Infobox settlement|Bilgiquti aholi punkti|Turar-joy bilgiqutisi).*?\}\}', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 5. Remove other common AI-generated top-level templates
        text = re.sub(r'\{\{(?:Short description|Redirect-several|Use dmy dates).*?\}\}', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 6. Convert Markdown bold to MediaWiki bold
        text = re.sub(r'\*\*([^*]+)\*\*', r"'''\1'''", text)
        
        # 7. Final Polish
        text = text.strip('"').strip("'").strip('`').strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

# ==========================================
# 🗺️ BILGIQUTI TEMPLATE MAPPER
# ==========================================
class BilgiquitiMapper:
    """Maps Wikidata/English parameters to Uzbek Bilgiquti templates"""
    
    SETTLEMENT_PARAM_MAP = {
        'oʻzbekcha nomi': 'oʻzbekcha nomi',
        'asl nomi': 'asl nomi',
        'tasvir': 'tasvir',
        'mavqe': 'mavqe',
        'mamlakat': 'mamlakat',
        'mintaqa turi': 'mintaqa turi',
        'mintaqa': 'mintaqa',
        'tuman turi': 'tuman turi',
        'tuman': 'tuman',
        'gerb': 'gerb',
        'bayroq': 'bayroq',
        'lat_deg': 'lat_deg',
        'lat_min': 'lat_min',
        'lat_sec': 'lat_sec',
        'lon_deg': 'lon_deg',
        'lon_min': 'lon_min',
        'lon_sec': 'lon_sec',
        'asos solingan': 'asos solingan',
        'ilk eslatilishi': 'ilk eslatilishi',
        'avvalgi nomlari': 'avvalgi nomlari',
        'maydon': 'maydon',
        'balandlik turi': 'balandlik turi',
        'AP markazi balandligi': 'AP markazi balandligi',
        'aholi': 'aholi',
        'sanalgan yil': 'sanalgan yil',
        'zichlik': 'zichlik',
        'milliy tarkib': 'milliy tarkib',
        'vaqt mintaqasi': 'vaqt mintaqasi',
        'telefon kodi': 'telefon kodi',
        'pochta indeksi': 'pochta indeksi',
        'avtomobil kodi': 'avtomobil kodi',
        'sayt': 'sayt',
    }

# ==========================================
# 🌐 ENGLISH WIKIPEDIA FETCHER
# ==========================================
class EnglishWikiFetcher:
    """Handles English Wikipedia article retrieval"""
    
    @staticmethod
    def get_article_summary(item: Any) -> Tuple[Optional[str], Optional[str]]:
        try:
            sitelinks = item.sitelinks
            if 'enwiki' not in sitelinks: return None, None
            
            enwiki_link = sitelinks['enwiki']
            title = enwiki_link.title if hasattr(enwiki_link, 'title') else str(enwiki_link)
            
            site = pywikibot.Site('en', 'wikipedia')
            page = pywikibot.Page(site, title)
            if not page.exists(): return None, None
            
            text = page.text
            intro = text.split('==')[0].strip()
            if not intro or len(intro) < 50: return None, None
            
            return intro, title
        except Exception as e:
            logger.error(f"Error fetching English text: {e}")
            return None, None

# ==========================================
class AITranslator:
    """Handles translation via OpenRouter with Multi-Model Fallback"""
    
    SYSTEM_PROMPT = """ROLE: Academic Uzbek Architect V5.0
MISSION: Translate English Wikipedia content into professional, academic Uzbek wikitext.

## 📜 CORE PRINCIPLES
1. **Academic Tone:** Use formal, encyclopedic Uzbek (O'zbek adabiy tili). Follow SOV (Subject-Object-Verb) structure.
2. **MediaWiki Integrity:** Preserve all internal links [[Link]], files [[File:Image.jpg]], and templates.
3. **Reference Preservation:** Keep all <ref> tags exactly where they are. Ensure they follow punctuation (e.g., .<ref>).
4. **Translation Rules:**
   - Translate captions for files.
   - Translate display text in links.
   - Use proper Uzbek suffixes.
   - Use guillemets « » for quotes.

## 🚨 OUTPUT REQUIREMENTS
- Output ONLY raw MediaWiki wikitext.
- NO triple quotes, NO markdown blocks, NO preamble chatter.
- Start directly with the translated content."""
    
    def __init__(self):
        try:
            self.client = OpenAI(
                api_key=Config.OPENROUTER_API_KEY,
                base_url=Config.OPENROUTER_ENDPOINT
            )
            logger.info(f"✅ OpenRouter client initialized at {Config.OPENROUTER_ENDPOINT}")
        except Exception as e:
            logger.critical(f"Failed to initialize AI client: {e}")
            raise
    
    def translate(self, data: Dict[str, Any], english_text: str, english_title: str) -> Optional[str]:
        name = data['name']
        user_prompt = f"TARGET ARTICLE: {name}\nSOURCE TEXT:\n{english_text}\n\nTranslate to Academic Uzbek. Preserve all <ref> and [[File:...]] tags."
        
        # MULTI-MODEL FALLBACK ENGINE
        for model in Config.FREE_MODELS:
            logger.info(f"🤖 Attempting translation via {model}...")
            result = self._call_api_with_retry(user_prompt, model)
            if result:
                logger.info(f"✅ Translation successful via {model}")
                return TextSanitizer.clean(result)
            logger.warning(f"❌ Model {model} failed. Trying next fallback...")
        
        return None
    
    def _call_api_with_retry(self, user_prompt: str, model: str) -> Optional[str]:
        retry_delay = Config.INITIAL_RETRY_DELAY
        for attempt in range(1, Config.MAX_RETRIES_PER_MODEL + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    max_tokens=Config.MAX_TOKENS,
                    model=model,
                    timeout=Config.REQUEST_TIMEOUT,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/ktozaxnt/uzwiki",
                        "X-Title": "Uzbek Wikipedia Bot"
                    }
                )
                if response and response.choices:
                    return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Attempt {attempt} for {model} failed: {e}")
                if attempt < Config.MAX_RETRIES_PER_MODEL:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        return None

# ==========================================
# 📝 ARTICLE BUILDER
# ==========================================
class ArticleBuilder:
    """Constructs complete Wikipedia articles with infoboxes"""
    @staticmethod
    def build(data: Dict[str, Any], uzbek_text: str, english_title: str) -> str:
        infobox = ArticleBuilder._build_infobox(data)
        references = "\n\n== Manbalar ==\n{{manbalar}}\n"
        categories = ArticleBuilder._build_categories(data)
        
        # Ensure the article starts with the infobox and has clean spacing
        article = f"{infobox}\n\n{uzbek_text.strip()}\n{references}{categories}"
        return article

    @staticmethod
    def _build_infobox(data: Dict[str, Any]) -> str:
        """Generate {{Bilgiquti aholi punkti}} with strict newline formatting"""
        params = {
            'oʻzbekcha nomi': data['name'],
            'mavqe': 'Shahar',
            'mamlakat': Config.COUNTRY_NAME,
        }
        if data.get('image'): params['tasvir'] = data['image']
        if data.get('region'):
            params['mintaqa turi'] = '[[Voevodalik]]'
            params['mintaqa'] = f"[[{data['region']}]]"
        if data.get('area'): params['maydon'] = data['area']
        if data.get('elev'): params['AP markazi balandligi'] = data['elev']
        if data.get('pop'):
            params['aholi'] = data['pop']
            params['sanalgan yil'] = '[[Wikidata]]'
        if data.get('postal'): params['pochta indeksi'] = data['postal']
        if data.get('coords_lat') and data.get('coords_lon'):
            params['lat_deg'] = data['coords_lat']
            params['lon_deg'] = data['coords_lon']
        
        # ARCHITECTURAL FIX: Every parameter MUST start with \n|
        lines = ["{{Bilgiquti aholi punkti"]
        for key, value in params.items():
            lines.append(f"| {key} = {value}")
        lines.append("}}")
        
        return "\n".join(lines)

    @staticmethod
    def _build_categories(data: Dict[str, Any]) -> str:
        cats = [f"\n[[Turkum:{Config.COUNTRY_NAME} shaharlari]]"]
        if data.get('region'): cats.append(f"[[Turkum:{data['region']} shaharlari]]")
        return "\n".join(cats)

# ==========================================
# 🗂️ WIKIDATA EXTRACTOR
# ==========================================
class WikidataExtractor:
    """Extracts structured data from Wikidata items"""
    @staticmethod
    def extract(item: Any) -> Optional[Dict[str, Any]]:
        try:
            item.get()
            name = item.labels.get('uz') or item.labels.get('en')
            if not name: return None
            
            region = WikidataExtractor._get_region(item)
            pop = WikidataExtractor._get_claim_value(item, 'P1082', int)
            area = WikidataExtractor._get_claim_value(item, 'P2046', float)
            elev = WikidataExtractor._get_claim_value(item, 'P2044', float)
            postal = WikidataExtractor._get_claim_value(item, 'P281', str)
            
            image = None
            if 'P18' in item.claims:
                img_target = item.claims['P18'][0].getTarget()
                image = img_target.title(with_ns=False) if img_target else None
            
            coords_lat, coords_lon = WikidataExtractor._get_coordinates(item)
            
            return {
                'name': name, 'region': region, 'pop': pop, 'area': area,
                'elev': elev, 'postal': postal, 'image': image,
                'coords_lat': coords_lat, 'coords_lon': coords_lon
            }
        except Exception as e:
            logger.error(f"Failed to extract data from {item.id}: {e}")
            return None
    
    @staticmethod
    def _get_region(item: Any) -> Optional[str]:
        if 'P131' not in item.claims: return None
        try:
            region_item = item.claims['P131'][0].getTarget()
            region_item.get()
            return region_item.labels.get('uz') or region_item.labels.get('en')
        except: return None

    @staticmethod
    def _get_claim_value(item: Any, prop: str, type_converter=None):
        if prop not in item.claims: return None
        try:
            target = item.claims[prop][0].getTarget()
            if hasattr(target, 'amount'):
                value = target.amount
                return type_converter(value) if type_converter else value
            return type_converter(target) if type_converter else target
        except: return None

    @staticmethod
    def _get_coordinates(item: Any) -> Tuple[Optional[float], Optional[float]]:
        if 'P625' not in item.claims: return None, None
        try:
            coord = item.claims['P625'][0].getTarget()
            return coord.lat, coord.lon
        except: return None, None

# ==========================================
# 🚀 MAIN BOT ENGINE
# ==========================================
class UzbekWikiBot:
    def __init__(self):
        Config.validate()
        self.translator = AITranslator()
        self.fetcher = EnglishWikiFetcher()
        self.extractor = WikidataExtractor()
        self.builder = ArticleBuilder()
        self.site = pywikibot.Site('uz', 'wikipedia')
        self.repo = self.site.data_repository()
        if not Config.SIMULATION_OR_DRAFT_MODE: self.site.login()

    def run(self):
        logger.info("=" * 60)
        logger.info(f"🤖 UZBEK WIKIPEDIA BOT v5.0")
        logger.info("=" * 60)
        
        # Test with Warsaw (Q270)
        test_qid = 'Q270' 
        item = pywikibot.ItemPage(self.repo, test_qid)
        
        try:
            data = self.extractor.extract(item)
            if not data: return
            
            name = data['name']
            page_title = f"Foydalanuvchi:{Config.BOT_USERNAME}/Qoralama/{name}"
            page = pywikibot.Page(self.site, page_title)
            
            english_text, english_title = self.fetcher.get_article_summary(item)
            if not english_text: return
            
            uzbek_text = self.translator.translate(data, english_text, english_title)
            if not uzbek_text: return
            
            full_article = self.builder.build(data, uzbek_text, english_title)
            
            # Local verification
            logger.info(f"🔍 VERIFYING WIKITEXT FOR {name}...")
            if "\n|" in full_article and "{{" in full_article and "}}" in full_article:
                logger.info("✅ Wikitext syntax appears correct.")
            else:
                logger.warning("⚠️ Wikitext syntax might be corrupted.")
            
            page.text = full_article
            summary = f"Bot qoralama: OpenRouter orqali tarjima ([[en:{english_title}]])"
            
            try:
                page.save(summary=summary, bot=True)
                logger.info(f"✅ DRAFT SAVED: {page.title()}")
            except pywikibot.exceptions.OtherPageSaveError as e:
                if "blocked" in str(e).lower():
                    logger.error("🛑 WIKIPEDIA BLOCK DETECTED. Saving locally.")
                    with open(f"{name}_local_test.txt", "w", encoding="utf-8") as f:
                        f.write(full_article)
                    logger.info(f"💾 Local copy saved to {name}_local_test.txt")
                else:
                    raise e
            
        except Exception as e:
            logger.error(f"💥 Error: {e}")

if __name__ == "__main__":
    bot = UzbekWikiBot()
    bot.run()
