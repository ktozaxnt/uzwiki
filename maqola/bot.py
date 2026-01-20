#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
========================================
UZBEK WIKIPEDIA BOT - POLISH CITIES
========================================
Version: 22.0 (MEDIAWIKI SYNTAX FIX)
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

# GitHub Models SDK
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
class Config:
    """Centralized configuration with validation"""
    SIMULATION_OR_DRAFT_MODE = False  # NEW: If True, saves to draft namespace
    BOT_USERNAME = "DanikBotUZ"  # NEW: Bot username for draft namespace
    
    COUNTRY_QID = 'Q36'  # Poland
    COUNTRY_NAME = 'Polsha'
    MAX_ARTICLES = 1
    
    # GitHub Models Configuration
    GITHUB_TOKEN = "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"
    GITHUB_ENDPOINT = "https://models.github.ai/inference"
    AI_MODEL = "meta/Llama-4-Scout-17B-16E-Instruct"
    
    # Model Parameters
    TEMPERATURE = 0.7
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
        if not cls.GITHUB_TOKEN or not cls.GITHUB_TOKEN.startswith("github_pat_"):
            raise ValueError("⚠️ CRITICAL: Invalid GitHub token in Config")
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
        'name': 'nomi',
        'official_name': 'rasmiy_nomi',
        'native_name': 'mahalliy_nomi',
        'settlement_type': 'turi',
        'image_skyline': 'rasm',
        'image_caption': 'izoh',
        'image_flag': 'bayroq',
        'image_shield': 'gerb',
        'nickname': 'laqab',
        'motto': 'shiori',
        'subdivision_type': 'mamlakat_turi',
        'subdivision_name': 'mamlakat',
        'subdivision_type1': 'viloyat_turi',
        'subdivision_name1': 'viloyat',
        'subdivision_type2': 'tuman_turi',
        'subdivision_name2': 'tuman',
        'established_title': 'tashkil_topgan',
        'established_date': 'tashkil_topgan_sana',
        'founder': 'asos_solgan',
        'area_total_km2': 'maydoni_km2',
        'area_land_km2': 'quruqlik_maydoni',
        'area_water_km2': 'suv_maydoni',
        'elevation_m': 'balandligi_m',
        'elevation_ft': 'balandligi_fut',
        'population_total': 'aholisi',
        'population_as_of': 'aholi_yili',
        'population_density_km2': 'aholi_zichligi',
        'population_footnotes': 'aholi_izohi',
        'timezone': 'vaqt_mintaqasi',
        'utc_offset': 'UTC',
        'postal_code_type': 'pochta_turi',
        'postal_code': 'pochta_indeksi',
        'area_code': 'telefon_kodi',
        'website': 'veb_sayt',
        'coordinates': 'koordinatalar',
        'pushpin_map': 'xarita',
        'pushpin_label_position': 'yorliq_joyi',
    }
    
    @classmethod
    def get_template_name(cls, entity_type: str) -> str:
        """Determine appropriate Uzbek template based on entity type"""
        # Default to settlement infobox for cities
        return "Turar-joy bilgiqutisi"
    
    @classmethod
    def map_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map English parameter names to Uzbek equivalents"""
        mapped = {}
        for eng_key, value in params.items():
            uz_key = cls.SETTLEMENT_PARAM_MAP.get(eng_key, eng_key)
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
            
            summary = page.extract(intro=True)
            
            if not summary or len(summary.strip()) < 50:
                logger.warning(f"English article '{title}' has no substantial intro")
                return None, None
            
            logger.debug(f"✅ Fetched English summary for '{title}' ({len(summary)} chars)")
            return summary, title
            
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
        
        # CRITICAL FIX #1: Strip all Python/Markdown code delimiters
        # Remove triple-quote blocks (Python docstrings)
        text = re.sub(r'^""".*?"""', '', text, flags=re.DOTALL | re.MULTILINE)
        text = re.sub(r"^'''.*?'''", '', text, flags=re.DOTALL | re.MULTILINE)
        
        # Remove Markdown code fences (```wikitext, ```mediawiki, etc.)
        text = re.sub(r'^```[a-z]*\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```', '', text, flags=re.MULTILINE)
        
        # Remove leading/trailing standalone quotes
        text = text.strip('"').strip("'").strip('`')
        
        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove TITLE: prefix if present
        text = re.sub(r'^TITLE:\s*.*?\n+', '', text, flags=re.IGNORECASE)
        
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
        text = re.sub(r'(?<!\')\'(?!\')', "''", text)
        
        # Clean excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text

# ==========================================
# 🤖 AI TRANSLATOR (GITHUB MODELS)
# ==========================================
class AITranslator:
    """Handles translation via GitHub Models API"""
    
    SYSTEM_PROMPT = """You are an expert Wikipedia editor and translator specializing in Uzbek Academic style.
Your task is to translate English Wikipedia content into high-quality Uzbek MediaWiki wikitext.

## 📜 MODULE 1: CORE TRANSLATION PRINCIPLES
1. **Academic Tone:** Use formal, encyclopedic Uzbek (O'zbek adabiy tili). Avoid colloquialisms.
2. **MediaWiki Integrity:** Preserve all internal links, templates, and formatting.
3. **Precision:** Ensure all numbers, dates, and names are accurately transferred.
4. **No Preamble:** Output ONLY the translated wikitext. No "Here is the translation" or code blocks.

## 🛠️ MODULE 2: SYNTAX RULES
- **Internal Links:** Translate the display text, but keep the target if it's a proper noun or check if an Uzbek equivalent exists.
  - `[[Poland|Polish]]` -> `[[Polsha|Polsha]]`
- **Categories:** Translate categories to `[[Turkum:Category Name]]`.
- **Templates:** Translate parameter values where appropriate, but keep parameter names if they are standard.
- **References:** Keep `<ref>` tags exactly where they are in the source.

## 🇺🇿 MODULE 3: UZBEK LINGUISTIC SPECIFICS
- Use "va" instead of "&".
- Use proper Uzbek suffixes (e.g., "Polshada", "Varshavaning").
- Use guillemets « » for quotes instead of " ".

## 🚨 MODULE 4: ERROR PREVENTION
- Do not hallucinate facts.
- If a term is untranslatable, keep it in English and put it in parentheses.
- Ensure all templates are closed with `}}`.

## 🎓 FINAL INSTRUCTION
Translate now. No preamble. No code delimiters. Start directly with raw MediaWiki wikitext."""
    
    def __init__(self):
        """Initialize GitHub Models client"""
        try:
            self.client = ChatCompletionsClient(
                endpoint=Config.GITHUB_ENDPOINT,
                credential=AzureKeyCredential(Config.GITHUB_TOKEN)
            )
            logger.info(f"✅ GitHub Models client initialized")
            logger.info(f"🤖 Model: {Config.AI_MODEL}")
        except Exception as e:
            logger.critical(f"Failed to initialize GitHub Models client: {e}")
            raise
    
    def translate(self, data: Dict[str, Any], english_text: str, english_title: str) -> Optional[str]:
        """Translate English text to Uzbek using DeepSeek V3 via GitHub Models"""
        name = data['name']
        region = data.get('region', 'Poland')
        
        user_prompt = f"""TARGET ARTICLE: {name} (Poland, {region})
SOURCE: en.wikipedia.org/wiki/{english_title.replace(' ', '_')}

INPUT SOURCE CODE:
{english_text}

Translate this to Academic Uzbek following all rules above. Output ONLY raw MediaWiki wikitext with NO delimiters."""
        
        logger.info(f"🤖 Translating via GitHub Models ({Config.AI_MODEL})...")
        
        result = self._call_api_with_retry(user_prompt)
        if result:
            return TextSanitizer.clean(result)
        
        logger.error("❌ Translation failed after all retries")
        return None
    
    def _call_api_with_retry(self, user_prompt: str) -> Optional[str]:
        """Call GitHub Models API with exponential backoff retry"""
        retry_delay = Config.INITIAL_RETRY_DELAY
        
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                logger.debug(f"📡 API attempt {attempt}/{Config.MAX_RETRIES}")
                
                if attempt > 1:
                    logger.info(f"⏳ Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=self.SYSTEM_PROMPT),
                        UserMessage(content=user_prompt)
                    ],
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    max_tokens=Config.MAX_TOKENS,
                    model=Config.AI_MODEL
                )
                
                if response and response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content
                    logger.info(f"✅ Translation received ({len(text)} chars)")
                    return text
                else:
                    logger.warning("Empty response from API")
                    retry_delay = min(retry_delay * 1.5, Config.MAX_RETRY_DELAY)
                    continue
            
            except HttpResponseError as e:
                status_code = e.status_code if hasattr(e, 'status_code') else 'unknown'
                
                if status_code == 429:
                    logger.warning(f"⏳ Rate limit (429) - attempt {attempt}/{Config.MAX_RETRIES}")
                    retry_delay = Config.RATE_LIMIT_DELAY
                
                elif status_code == 401:
                    logger.critical("🔒 Authentication failed (401) - check GitHub token permissions")
                    logger.critical("Token needs 'models:read' scope")
                    return None
                
                elif status_code == 404:
                    logger.error(f"❌ Model '{Config.AI_MODEL}' not found (404)")
                    return None
                
                elif isinstance(status_code, int) and 500 <= status_code < 600:
                    logger.warning(f"🔥 Server error {status_code} - retry in {retry_delay}s")
                    retry_delay = min(retry_delay * 1.5, Config.MAX_RETRY_DELAY)
                
                else:
                    logger.error(f"⚠️ HTTP error {status_code}: {str(e)[:200]}")
                    retry_delay = min(retry_delay * 1.3, Config.MAX_RETRY_DELAY)
            
            except Exception as e:
                logger.error(f"💥 Unexpected error: {type(e).__name__}: {str(e)[:200]}")
                retry_delay = min(retry_delay * 1.5, Config.MAX_RETRY_DELAY)
        
        logger.error(f"Exhausted all {Config.MAX_RETRIES} retries")
        return None

# ==========================================
# 📝 ARTICLE BUILDER
# ==========================================
class ArticleBuilder:
    """Constructs complete Wikipedia articles with infoboxes"""
    @staticmethod
    def build(data: Dict[str, Any], uzbek_text: str, english_title: str) -> str:
        """Assemble final article: Infobox + Translated Text + References + Categories"""
        infobox = ArticleBuilder._build_infobox(data)
        references = "\n\n== Manbalar ==\n{{manbalar}}\n"
        categories = ArticleBuilder._build_categories(data)
        
        article = f"{infobox}\n\n{uzbek_text}{references}{categories}"
        
        logger.debug(f"Article assembled: {len(article)} total chars")
        return article

    @staticmethod
    def _build_infobox(data: Dict[str, Any]) -> str:
        """Generate {{Turar-joy bilgiqutisi}} infobox with proper line breaks"""
        template_name = BilgiquitiMapper.get_template_name("settlement")
        
        # Build raw parameters
        raw_params = {
            'name': data['name'],
            'settlement_type': 'Shahar',
            'subdivision_type': 'Mamlakat',
            'subdivision_name': Config.COUNTRY_NAME,
        }
        
        if data.get('image'):
            raw_params['image_skyline'] = data['image']
        
        if data.get('region'):
            raw_params['subdivision_type1'] = 'Voevodalik'
            raw_params['subdivision_name1'] = f"[[{data['region']}]]"
        
        if data.get('area'):
            raw_params['area_total_km2'] = data['area']
        
        if data.get('elev'):
            raw_params['elevation_m'] = data['elev']
        
        if data.get('pop'):
            raw_params['population_total'] = data['pop']
            raw_params['population_footnotes'] = '<ref name="wikidata">Wikidata ma\'lumotlari</ref>'
        
        if data.get('postal'):
            raw_params['postal_code_type'] = 'Pochta indeksi'
            raw_params['postal_code'] = data['postal']
        
        if data.get('coords_lat') and data.get('coords_lon'):
            raw_params['pushpin_map'] = 'Poland'
            raw_params['pushpin_label_position'] = 'bottom'
            raw_params['coordinates'] = f"{{{{coord|{data['coords_lat']}|{data['coords_lon']}|type:city_region:PL|display=inline,title}}}}"
        
        # Map to Uzbek parameter names
        mapped_params = BilgiquitiMapper.map_parameters(raw_params)
        
        # CRITICAL FIX #3: Build template with proper line breaks
        lines = [f"{{{{{template_name}}}"]
        for key, value in mapped_params.items():
            lines.append(f"|{key} = {value}")
        lines.append("}}")
        
        return "\n".join(lines)

    @staticmethod
    def _build_categories(data: Dict[str, Any]) -> str:
        """Generate category tags"""
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
        """Extract all relevant data fields from a Wikidata item"""
        try:
            item.get()
            
            labels = item.labels
            # Prioritize Uzbek label, fallback to English
            name = labels.get('uz') or labels.get('en')
            
            if not name:
                logger.warning(f"Item {item.id} has no uz/en label - skipping")
                return None
            
            region = WikidataExtractor._get_region(item)
            
            pop = WikidataExtractor._get_claim_value(item, 'P1082', int)
            area = WikidataExtractor._get_claim_value(item, 'P2046', float)
            elev = WikidataExtractor._get_claim_value(item, 'P2044', float)
            postal = WikidataExtractor._get_claim_value(item, 'P281', str)
            
            image = None
            if 'P18' in item.claims:
                try:
                    img_target = item.claims['P18'][0].getTarget()
                    image = img_target.title(with_ns=False) if img_target else None
                except:
                    pass
            
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
        """Get region name (P131 - voivodeship) - prefer Uzbek label"""
        if 'P131' not in item.claims:
            return None
        
        try:
            region_item = item.claims['P131'][0].getTarget()
            region_item.get()
            # Prioritize Uzbek label
            return region_item.labels.get('uz') or region_item.labels.get('en')
        except:
            return None

    @staticmethod
    def _get_claim_value(item: Any, prop: str, type_converter=None):
        """Generic claim value extractor with type conversion"""
        if prop not in item.claims:
            return None
        
        try:
            target = item.claims[prop][0].getTarget()
            
            if hasattr(target, 'amount'):
                value = target.amount
                return type_converter(value) if type_converter else value
            
            return type_converter(target) if type_converter else target
        except:
            return None

    @staticmethod
    def _get_coordinates(item: Any) -> Tuple[Optional[float], Optional[float]]:
        """Extract latitude and longitude"""
        if 'P625' not in item.claims:
            return None, None
        
        try:
            coord = item.claims['P625'][0].getTarget()
            return coord.lat, coord.lon
        except:
            return None, None

# ==========================================
# 🚀 MAIN BOT ENGINE
# ==========================================
class UzbekWikiBot:
    """Main bot orchestrator"""
    def __init__(self):
        Config.validate()
        self.translator = AITranslator()
        self.fetcher = EnglishWikiFetcher()
        self.extractor = WikidataExtractor()
        self.builder = ArticleBuilder()
        
        self._cleanup_lock_files()
        
        self.site = pywikibot.Site('uz', 'wikipedia')
        self.repo = self.site.data_repository()
        
        if not Config.SIMULATION_OR_DRAFT_MODE:
            try:
                self.site.login()
                logger.info("✅ Logged in to uz.wikipedia.org")
            except Exception as e:
                logger.warning(f"Login failed (continuing anyway): {e}")
        else:
            logger.info("🧪 Running in DRAFT MODE - articles will be saved to user subpages")

    def _cleanup_lock_files(self):
        """Remove Pywikibot lock files from previous sessions"""
        for pattern in ["*.lwp", "pywikibot-*.lwp", "apicache-*.sqlite3"]:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    logger.debug(f"Removed lock file: {file}")
                except:
                    pass

    def _get_page_title(self, article_name: str) -> str:
        """Determine target page title based on mode"""
        if Config.SIMULATION_OR_DRAFT_MODE:
            # Save to user draft subpage
            return f"Foydalanuvchi:{Config.BOT_USERNAME}/Qoralama/{article_name}"
        else:
            # Save to main namespace
            return article_name

    def run(self):
        """Main execution loop"""
        logger.info("=" * 60)
        logger.info(f"🤖 UZBEK WIKIPEDIA BOT v22.0 (MEDIAWIKI SYNTAX FIX)")
        logger.info(f"📍 Target: Polish cities (Q{Config.COUNTRY_QID})")
        logger.info(f"🎯 Max articles: {Config.MAX_ARTICLES}")
        logger.info(f"🧪 Draft mode: {Config.SIMULATION_OR_DRAFT_MODE}")
        logger.info(f"🤖 Model: {Config.AI_MODEL}")
        logger.info(f"📄 Log file: {Config.LOG_FILE}")
        logger.info("=" * 60)
        
        query = f"""
        SELECT DISTINCT ?item WHERE {{
          ?item wdt:P31/wdt:P279* wd:Q486972;
                wdt:P17 wd:{Config.COUNTRY_QID}.
        }}
        LIMIT 100
        """
        
        generator = pagegenerators.WikidataSPARQLPageGenerator(query, site=self.repo)
        
        count = 0
        skipped = 0
        failed = 0
        
        for item in generator:
            if count >= Config.MAX_ARTICLES:
                logger.info(f"🎯 Reached target of {Config.MAX_ARTICLES} articles - stopping")
                break
            
            try:
                data = self.extractor.extract(item)
                if not data:
                    skipped += 1
                    continue
                
                name = data['name']
                logger.info(f"\n{'=' * 60}")
                logger.info(f"[{count + 1}] Processing: {name} ({item.id})")
                
                # Determine target page based on mode
                page_title = self._get_page_title(name)
                page = pywikibot.Page(self.site, page_title)
                
                if page.exists():
                    logger.info(f"⭐️ SKIP: Page already exists at {page_title}")
                    skipped += 1
                    continue
                
                english_text, english_title = self.fetcher.get_article_summary(item)
                if not english_text:
                    logger.info(f"⭐️ SKIP: No English Wikipedia article found")
                    skipped += 1
                    continue
                
                logger.info(f"📖 Source: en.wikipedia.org/wiki/{english_title}")
                
                uzbek_text = self.translator.translate(data, english_text, english_title)
                if not uzbek_text:
                    logger.error(f"❌ FAIL: Translation failed for {name}")
                    failed += 1
                    continue
                
                full_article = self.builder.build(data, uzbek_text, english_title)
                
                self._save_article(page, full_article, english_title, name)
                
                count += 1
                
            except KeyboardInterrupt:
                logger.critical("⚠️ MANUAL INTERRUPTION - Stopping bot")
                break
            
            except Exception as e:
                logger.error(f"💥 CRITICAL ERROR in main loop: {type(e).__name__}: {e}")
                failed += 1
                continue
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"✅ Created: {count}")
        logger.info(f"⭐️ Skipped: {skipped}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📄 Full log: {Config.LOG_FILE}")
        logger.info("=" * 60)

    def _save_article(self, page, content: str, source_title: str, name: str):
        """Save article to Wikipedia with draft/live mode logic"""
        try:
            page.text = content
            
            # Build appropriate summary based on mode
            if Config.SIMULATION_OR_DRAFT_MODE:
                summary = f"Bot qoralama: ingliz Vikipediyadan tarjima ([[en:{source_title}]]) - tekshirish uchun"
            else:
                summary = f"Bot: Maqola ingliz Vikipediyadan tarjima qilindi ([[en:{source_title}]])"
            
            page.save(summary=summary, bot=True, minor=False)
            
            if Config.SIMULATION_OR_DRAFT_MODE:
                logger.info(f"✅ DRAFT SAVED: {page.title()}")
            else:
                logger.info(f"✅ PUBLISHED: {name}")
            
            logger.info(f"⏳ Cooling down {Config.EDIT_DELAY}s to avoid Captcha...")
            time.sleep(Config.EDIT_DELAY)
        
        except pywikibot.exceptions.CaptchaError:
            logger.critical("🛑 CAPTCHA TRIGGERED - Manual intervention required")
            logger.critical("Please solve the Captcha in your browser and restart the bot")
            sys.exit(1)
        
        except pywikibot.exceptions.EditConflictError:
            logger.error("⚠️ Edit conflict detected - skipping this article")
        
        except pywikibot.exceptions.SpamblacklistError as e:
            logger.error(f"⚠️ Spam blacklist hit: {e}")
        
        except pywikibot.exceptions.LockedPageError:
            logger.error("⚠️ Page is locked - cannot edit")
        
        except Exception as e:
            logger.error(f"❌ Save failed: {type(e).__name__}: {e}")
            raise

# ==========================================
# 🎬 ENTRY POINT
# ==========================================
if __name__ == "__main__":
    try:
        bot = UzbekWikiBot()
        bot.run()
    except Exception as e:
        logger.critical(f"💀 FATAL ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
