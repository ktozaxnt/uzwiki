#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
========================================
UZBEK WIKIPEDIA BOT - POLISH CITIES
========================================
Version: 3.0 (ACADEMIC UZBEK ARCHITECT)
Purpose: Mass-generate geography articles from Wikidata + English Wikipedia
Target: uz.wikipedia.org
API: GitHub Models (models.github.ai) via Azure AI Inference SDK
Author: Senior Python Architect
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
from typing import Optional, Tuple, Dict, Any
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
    
    # GitHub Models Configuration
    GITHUB_TOKEN = os.environ.get("OPENAI_API_KEY")
    GITHUB_ENDPOINT = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    AI_MODEL = "gpt-4.1-mini"
    
    # Model Parameters
    TEMPERATURE = 0.2 # Lower temperature for more consistent academic output
    TOP_P = 0.95
    MAX_TOKENS = 4000
    
    # Retry Configuration
    MAX_RETRIES = 12
    INITIAL_RETRY_DELAY = 30
    MAX_RETRY_DELAY = 120
    RATE_LIMIT_DELAY = 60
    
    # Bot Behavior
    EDIT_DELAY = 50
    REQUEST_TIMEOUT = 180
    
    # Logging
    LOG_DIR = Path("logs")
    LOG_FILE = LOG_DIR / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    @classmethod
    def validate(cls):
        """Validate configuration before startup"""
        if not cls.GITHUB_TOKEN:
            raise ValueError("⚠️ CRITICAL: No API token configured")
        if not cls.AI_MODEL:
            raise ValueError("⚠️ CRITICAL: No AI model configured")
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
# 🗺️ BILGIQUTI TEMPLATE MAPPER
# ==========================================
class BilgiquitiMapper:
    """Maps English Infobox parameters to Uzbek Bilgiquti templates"""
    
    # Standard parameter mappings for settlement infoboxes
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
    
    @classmethod
    def get_template_name(cls, entity_type: str) -> str:
        """Determine appropriate Uzbek template based on entity type"""
        return "Bilgiquti aholi punkti"
    
    @classmethod
    def map_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map English parameter names to Uzbek equivalents"""
        mapped = {}
        for key, value in params.items():
            uz_key = cls.SETTLEMENT_PARAM_MAP.get(key, key)
            mapped[uz_key] = value
        return mapped

# ==========================================
# 🌐 ENGLISH WIKIPEDIA FETCHER
# ==========================================
class EnglishWikiFetcher:
    """Handles English Wikipedia article retrieval with strict validation"""
    
    @staticmethod
    def get_article_summary(item: Any) -> Tuple[Optional[str], Optional[str]]:
        """Fetch English Wikipedia summary (intro section only)"""
        try:
            sitelinks = item.sitelinks
            
            if 'enwiki' not in sitelinks:
                logger.debug(f"No enwiki sitelink for {item.id}")
                return None, None
            
            enwiki_link = sitelinks['enwiki']
            title = enwiki_link.title if hasattr(enwiki_link, 'title') else str(enwiki_link)
            
            site = pywikibot.Site('en', 'wikipedia')
            page = pywikibot.Page(site, title)
            
            if not page.exists():
                logger.warning(f"English article '{title}' does not exist")
                return None, None
            
            # Get full text to preserve refs and files
            text = page.text
            
            # Extract intro section (before first header)
            intro = text.split('==')[0].strip()
            
            if not intro or len(intro) < 50:
                logger.warning(f"English article '{title}' has no substantial intro")
                return None, None
            
            logger.debug(f"✅ Fetched English intro for '{title}' ({len(intro)} chars)")
            return intro, title
            
        except pywikibot.exceptions.NoPageError:
            logger.error(f"English page not found for {item.id}")
            return None, None
        except Exception as e:
            logger.error(f"Error fetching English text: {type(e).__name__}: {e}")
            return None, None

# ==========================================
# 🧹 TEXT SANITIZER (UPGRADED)
# ==========================================
class TextSanitizer:
    """Converts AI output to clean MediaWiki syntax - AGGRESSIVE MODE"""
    
    @staticmethod
    def clean(text: str) -> str:
        """Multi-stage cleaning pipeline with code delimiter removal"""
        if not text:
            return ""
        
        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove triple-quote blocks
        text = re.sub(r'^""".*?"""', '', text, flags=re.DOTALL | re.MULTILINE)
        text = re.sub(r"^'''.*?'''", '', text, flags=re.DOTALL | re.MULTILINE)
        
        # Remove Markdown code fences
        text = re.sub(r'^```[a-z]*\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```', '', text, flags=re.MULTILINE)
        
        # Remove leading/trailing standalone quotes
        text = text.strip('"').strip("'").strip('`')
        
        # Remove meta commentary prefixes
        meta_patterns = [
            r'^(Here is|Here\'s|Below is|The following is).*?:?\s*',
            r'^(Вот|Это|Ниже|Следующ).*?:?\s*',
            r'^\s*---+\s*$',
            r'^RAW WIKITEXT:.*$',
            r'^OUTPUT:.*$',
            r'^BEGIN UZBEK TRANSLATION:.*$',
            r'^EXECUTE:.*$',
            r'^TRANSLATION:.*$',
        ]
        for pattern in meta_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Convert Markdown headers to MediaWiki
        text = re.sub(r'^####\s+(.*?)\s*$', r'==== \1 ====', text, flags=re.MULTILINE)
        text = re.sub(r'^###\s+(.*?)\s*$', r'=== \1 ===', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.*?)\s*$', r'== \1 ==', text, flags=re.MULTILINE)
        
        # Convert bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r"'''\1'''", text)
        
        # Clean excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text

# ==========================================
class AITranslator:
    """Handles translation via GitHub Models API"""
    
    SYSTEM_PROMPT = """ROLE: Academic Uzbek Architect V3.0
MISSION: Translate English Wikipedia content into professional, academic Uzbek wikitext.

## 📜 CORE PRINCIPLES
1. **Academic Tone:** Use formal, encyclopedic Uzbek (O'zbek adabiy tili). Follow SOV (Subject-Object-Verb) structure.
2. **MediaWiki Integrity:** Preserve all internal links [[Link]], files [[File:Image.jpg]], and templates.
3. **Reference Preservation:** Keep all <ref> tags exactly where they are, but ensure they follow punctuation (e.g., .<ref> instead of <ref>.).
4. **Translation Rules:**
   - Translate captions for files: [[File:Image.jpg|thumb|Translated Caption]]
   - Translate display text in links: [[Target|Translated Text]]
   - Use proper Uzbek suffixes (e.g., "Polshada", "Varshavaning").
   - Use guillemets « » for quotes.

## 🚨 OUTPUT REQUIREMENTS
- Output ONLY raw MediaWiki wikitext.
- NO triple quotes, NO markdown blocks, NO preamble chatter.
- NO "Here is the translation".
- Start directly with the translated content."""
    
    def __init__(self):
        """Initialize OpenAI client"""
        try:
            # Explicitly use the environment variables to ensure correct routing
            self.client = OpenAI(
                api_key=Config.GITHUB_TOKEN,
                base_url=Config.GITHUB_ENDPOINT
            )
            logger.info(f"✅ OpenAI client initialized")
        except Exception as e:
            logger.critical(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def translate(self, data: Dict[str, Any], english_text: str, english_title: str) -> Optional[str]:
        """Translate English text to Uzbek"""
        name = data['name']
        
        user_prompt = f"""TARGET ARTICLE: {name}
SOURCE TEXT:
{english_text}

Translate the above to Academic Uzbek. Preserve all <ref> tags and [[File:...]] tags. Translate captions."""
        
        logger.info(f"🤖 Translating via AI ({Config.AI_MODEL})...")
        
        result = self._call_api_with_retry(user_prompt)
        if result:
            return TextSanitizer.clean(result)
        
        return None
    
    def _call_api_with_retry(self, user_prompt: str) -> Optional[str]:
        """Call AI API with retry logic"""
        retry_delay = Config.INITIAL_RETRY_DELAY
        
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                if attempt > 1:
                    time.sleep(retry_delay)
                
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    max_tokens=Config.MAX_TOKENS,
                    model=Config.AI_MODEL
                )
                
                if response and response.choices:
                    return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"API attempt {attempt} failed: {e}")
                retry_delay = min(retry_delay * 1.5, Config.MAX_RETRY_DELAY)
        
        return None

# ==========================================
# 📝 ARTICLE BUILDER
# ==========================================
class ArticleBuilder:
    """Constructs complete Wikipedia articles with infoboxes"""
    @staticmethod
    def build(data: Dict[str, Any], uzbek_text: str, english_title: str) -> str:
        """Assemble final article"""
        infobox = ArticleBuilder._build_infobox(data)
        references = "\n\n== Manbalar ==\n{{manbalar}}\n"
        categories = ArticleBuilder._build_categories(data)
        
        article = f"{infobox}\n\n{uzbek_text}{references}{categories}"
        return article

    @staticmethod
    def _build_infobox(data: Dict[str, Any]) -> str:
        """Generate {{Bilgiquti aholi punkti}} infobox with strict formatting"""
        template_name = BilgiquitiMapper.get_template_name("settlement")
        
        params = {
            'oʻzbekcha nomi': data['name'],
            'mavqe': 'Shahar',
            'mamlakat': Config.COUNTRY_NAME,
        }
        
        if data.get('image'):
            params['tasvir'] = data['image']
        
        if data.get('region'):
            params['mintaqa turi'] = '[[Voevodalik]]'
            params['mintaqa'] = f"[[{data['region']}]]"
        
        if data.get('area'):
            params['maydon'] = data['area']
        
        if data.get('elev'):
            params['AP markazi balandligi'] = data['elev']
        
        if data.get('pop'):
            params['aholi'] = data['pop']
            params['sanalgan yil'] = '[[Wikidata]]'
        
        if data.get('postal'):
            params['pochta indeksi'] = data['postal']
        
        if data.get('coords_lat') and data.get('coords_lon'):
            # Convert to deg/min/sec if possible or use decimal
            params['lat_deg'] = data['coords_lat']
            params['lon_deg'] = data['coords_lon']
        
        # Build template with strict formatting
        lines = [f"{{{{{template_name}}}"]
        for key, value in params.items():
            lines.append(f"| {key} = {value}")
        lines.append("}}")
        
        return "\n".join(lines)

    @staticmethod
    def _build_categories(data: Dict[str, Any]) -> str:
        """Generate category tags at the bottom"""
        cats = [f"\n[[Turkum:{Config.COUNTRY_NAME} shaharlari]]"]
        if data.get('region'):
            cats.append(f"[[Turkum:{data['region']} shaharlari]]")
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
            labels = item.labels
            name = labels.get('uz') or labels.get('en')
            
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
                'name': name,
                'region': region,
                'pop': pop,
                'area': area,
                'elev': elev,
                'postal': postal,
                'image': image,
                'coords_lat': coords_lat,
                'coords_lon': coords_lon
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
        
        if not Config.SIMULATION_OR_DRAFT_MODE:
            self.site.login()
            logger.info("✅ Logged in to uz.wikipedia.org")
        else:
            logger.info("🧪 Running in DRAFT MODE")

    def run(self):
        logger.info("=" * 60)
        logger.info(f"🤖 UZBEK WIKIPEDIA BOT v3.0")
        logger.info("=" * 60)
        
        # Test with Poland (Q36) or a specific city like Warsaw (Q270)
        # For testing, let's pick a specific city: Warsaw (Q270)
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
            
            page.text = full_article
            summary = f"Bot qoralama: ingliz Vikipediyadan tarjima ([[en:{english_title}]])"
            page.save(summary=summary, bot=True)
            logger.info(f"✅ DRAFT SAVED: {page.title()}")
            
        except Exception as e:
            logger.error(f"💥 Error: {e}")

if __name__ == "__main__":
    bot = UzbekWikiBot()
    bot.run()
