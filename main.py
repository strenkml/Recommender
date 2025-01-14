import sys
import random
import os
import contextlib
import re
import sqlite3
import torch
import requests
import json
import logging
import numpy as np
import threading
import asyncio
import aiohttp
import time
import tempfile
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union, Set
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, QPushButton, QGroupBox, QFormLayout, QLineEdit, QListWidget, 
    QListWidgetItem, QMessageBox, QAbstractItemView, QDialog, QRadioButton, 
    QDialogButtonBox, QButtonGroup
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QUrl, QObject, QEventLoop, QSize
)
from PySide6.QtGui import (
    QPalette, QColor, QPixmap, QPainter, QFont, QIcon, QCloseEvent
)
from PySide6.QtNetwork import (
    QNetworkAccessManager, QNetworkRequest, QNetworkReply
)

from plexapi.server import PlexServer
from aiohttp import ClientTimeout, ClientSession, TCPConnector, ClientError
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
import nltk
from transformers import AutoModel, AutoTokenizer

class Database:
    _instance = None

    @staticmethod
    def get_instance():
        if Database._instance is None:
            Database._instance = Database()
        return Database._instance

    def __init__(self):
        self.conn = None
        self.write_lock = threading.Lock()
        self._connect()
        self._optimize_connection()
        self._create_tables()
        self._create_indices()
        self.logger = logging.getLogger(__name__)

    def _connect(self):
        try:
            self.conn = sqlite3.connect('media_recommender.db', check_same_thread=False)
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise

    def _optimize_connection(self):
        cursor = None
        try:
            cursor = self.conn.cursor()
            pragmas = [
                'PRAGMA journal_mode = WAL',
                'PRAGMA synchronous = NORMAL',
                'PRAGMA cache_size = -2000000',
                'PRAGMA mmap_size = 30000000000',
                'PRAGMA temp_store = MEMORY',
                'PRAGMA page_size = 4096',
                'PRAGMA foreign_keys = ON'
            ]
            for pragma in pragmas:
                cursor.execute(pragma)
            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()

    def _create_tables(self):
        with self.write_lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS media_items (
                        id INTEGER PRIMARY KEY,
                        title TEXT,
                        type TEXT,
                        year INTEGER,
                        runtime TEXT,
                        summary TEXT,
                        genres TEXT,
                        poster_url TEXT,
                        tvdb_id TEXT,
                        tmdb_id TEXT,
                        original_title TEXT,
                        overview TEXT,
                        popularity REAL,
                        vote_average REAL,
                        vote_count INTEGER,
                        status TEXT,
                        tagline TEXT,
                        backdrop_path TEXT,
                        release_date TEXT,
                        content_rating TEXT,
                        network TEXT,
                        credits TEXT,
                        keywords TEXT,
                        videos TEXT,
                        language TEXT,
                        production_companies TEXT,
                        reviews TEXT,
                        episodes TEXT,
                        season_count INTEGER,
                        episode_count INTEGER,
                        first_air_date TEXT,
                        last_air_date TEXT,
                        score_adjustment REAL DEFAULT 0
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS genre_preferences (
                        genre TEXT PRIMARY KEY,
                        rating_sum REAL,
                        rating_count INTEGER
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY,
                        media_id INTEGER,
                        rating INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (media_id) REFERENCES media_items (id)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        media_id INTEGER PRIMARY KEY,
                        embedding BLOB,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (media_id) REFERENCES media_items (id)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS similarity_matrix (
                        item1_id INTEGER,
                        item2_id INTEGER,
                        similarity REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (item1_id, item2_id),
                        FOREIGN KEY (item1_id) REFERENCES media_items (id),
                        FOREIGN KEY (item2_id) REFERENCES media_items (id)
                    )
                ''')

                self.conn.commit()
            finally:
                cursor.close()

    def _create_indices(self):
        cursor = self.conn.cursor()
        try:
            indices = [
                'CREATE INDEX IF NOT EXISTS idx_media_type_year ON media_items (type, year)',
                'CREATE INDEX IF NOT EXISTS idx_media_vote_popularity ON media_items (vote_average, vote_count, popularity)',
                'CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback (timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_embedding_last_updated ON embedding_cache (last_updated)',
                'CREATE INDEX IF NOT EXISTS idx_similarity_last_updated ON similarity_matrix (last_updated)',
                'CREATE INDEX IF NOT EXISTS idx_media_recommendations ON media_items (type, vote_average, vote_count, popularity, id, title, genres)',
                'CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON user_feedback (media_id, rating, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_media_title_search ON media_items(title COLLATE NOCASE)',
                'CREATE INDEX IF NOT EXISTS idx_genre_preferences_rating ON genre_preferences(rating_sum, rating_count)',
                'CREATE INDEX IF NOT EXISTS idx_active_media ON media_items(id, type) WHERE status != "Ended" AND status != "Cancelled"',
                'CREATE INDEX IF NOT EXISTS idx_embedding_cleanup ON embedding_cache(last_updated)',
                'CREATE INDEX IF NOT EXISTS idx_media_items_title_year ON media_items (title, year)',
                'CREATE INDEX IF NOT EXISTS idx_user_feedback_media_id ON user_feedback (media_id)',
                'CREATE INDEX IF NOT EXISTS idx_feedback_media_rating ON user_feedback (media_id, rating)',
                'CREATE INDEX IF NOT EXISTS idx_items_type ON media_items (type)',
                'CREATE INDEX IF NOT EXISTS idx_media_genres ON media_items (genres)',
                'CREATE INDEX IF NOT EXISTS idx_media_embedding_join ON media_items (id)',
                'CREATE INDEX IF NOT EXISTS idx_feedback_media_join ON user_feedback (media_id)',
                'CREATE INDEX IF NOT EXISTS idx_sim_item1 ON similarity_matrix (item1_id)',
                'CREATE INDEX IF NOT EXISTS idx_sim_item2 ON similarity_matrix (item2_id)'
            ]
            for index in indices:
                cursor.execute(index)

            cursor.execute('ANALYZE media_items')
            cursor.execute('ANALYZE user_feedback')
            cursor.execute('ANALYZE embedding_cache')
            cursor.execute('ANALYZE similarity_matrix')
            cursor.execute('ANALYZE genre_preferences')

            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    @contextlib.contextmanager
    def get_cursor(self):
        cursor = None
        try:
            cursor = self.conn.cursor()
            yield cursor
        finally:
            if cursor:
                cursor.close()

    def execute_write(self, query, params=None):
        if not self.write_lock.acquire(timeout=10):
            raise TimeoutError("Could not acquire database write lock")
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params or ())
                self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            self.write_lock.release()

    def execute_read(self, query, params=None):
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()

    def save_embedding(self, media_id, embedding):
        with self.write_lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO embedding_cache (media_id, embedding, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (media_id, embedding))
                self.conn.commit()
            except sqlite3.Error as e:
                self.conn.rollback()
                raise
            finally:
                cursor.close()

    def load_embedding(self, media_id):
        try:
            with self.get_cursor() as cursor:
                cursor.execute('SELECT embedding FROM embedding_cache WHERE media_id = ?', (media_id,))
                result = cursor.fetchone()
                return result[0] if result and result[0] else None
        except sqlite3.Error as e:
            self.logger.error(f"Error loading embedding for media_id {media_id}: {e}")
            return None

    def save_feedback(self, media_id, rating):
        with self.write_lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO user_feedback (media_id, rating)
                    VALUES (?, ?)
                ''', (media_id, rating))
                self.conn.commit()
            except sqlite3.Error as e:
                self.conn.rollback()
                raise
            finally:
                cursor.close()

    def get_feedback(self):
        with self.get_cursor() as cursor:
            cursor.execute('SELECT media_id, AVG(rating) as avg_rating FROM user_feedback GROUP BY media_id')
            return cursor.fetchall()

    def get_media_items(self, filters=None):
        filters = filters or {}
        query = "SELECT id, title, genres FROM media_items"
        conditions = []
        params = []

        if 'type' in filters:
            conditions.append("type = ?")
            params.append(filters['type'])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [{'id': row[0], 'title': row[1], 'genres': row[2]} for row in rows]
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving media items: {e}")
            return []

    def cleanup_old_cache(self):
        try:
            self.execute_write('DELETE FROM embedding_cache WHERE last_updated < datetime("now", "-30 days")')
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up cache: {e}")

    def optimize_database(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute('VACUUM')
            cursor.execute('ANALYZE')
            cursor.execute('ANALYZE media_items')
            cursor.execute('ANALYZE user_feedback')
            cursor.execute('ANALYZE embedding_cache')
            cursor.execute('ANALYZE similarity_matrix')
            cursor.execute('ANALYZE genre_preferences')
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error optimizing database: {e}")
            raise
        finally:
            cursor.close()

    def get_database_stats(self):
        with self.get_cursor() as cursor:
            stats = {}
            tables = ['media_items', 'user_feedback', 'embedding_cache', 'similarity_matrix', 'genre_preferences']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['database_size_mb'] = (page_count * page_size) / (1024 * 1024)

            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index'")
            stats['indices'] = cursor.fetchall()

            return stats

    def close(self):
        if not self.write_lock.acquire(timeout=5):
            self.logger.warning("Could not acquire lock for database closing")
            return False 
        
        success = False
        try:
            if self.conn:
                try:
                    if self.conn:
                        self.conn.commit() 
                    self.conn.close()
                    success = True
                except sqlite3.Error as e:
                    self.logger.error(f"Error closing database: {e}")
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                finally:
                    self.conn = None  
        finally:
            self.write_lock.release()
        return success

    def __del__(self):
        if hasattr(self, 'conn') and self.conn is not None:
            self.close()

class APIClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def close(self):
        with self._lock:
            if self.session:
                self.session.close()

class TVDBClient(APIClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://api4.thetvdb.com/v4"
        self.token = None
        self.token_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    async def ensure_token(self):
        with self.token_lock:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/login",
                        json={"apikey": self.api_key},
                        timeout=10
                    ) as response:
                        if response.status != 200:
                            raise ValueError(f"TVDB authentication failed: {await response.text()}")
                        data = await response.json()
                        self.token = data.get('data', {}).get('token')
                        if not self.token:
                            raise ValueError("No authentication token received")
                       
                        self.session.headers.update({
                            "Authorization": f"Bearer {self.token}"
                        })
            except Exception as e:
                self.logger.error(f"Token refresh failed: {str(e)}")
                raise

    async def search(self, title: str) -> Optional[List[Dict[str, Any]]]:
        await self.ensure_token()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/search"
                params = {"query": title, "type": "series"}
                headers = {"Authorization": f"Bearer {self.token}"}
               
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    return None
        except Exception as e:
            self.logger.error(f"TVDB search error: {str(e)}")
            return None

    async def fetch_series_data(self, tvdb_id: str) -> Dict:
        tvdb_id = tvdb_id.replace('series-', '')
        await self.ensure_token()
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.token}"}
               
                if not tvdb_id.isdigit():
                    raise ValueError(f"Invalid TVDB ID format: {tvdb_id}")

                series_url = f"{self.base_url}/series/{tvdb_id}/extended"
                async with session.get(series_url, headers=headers) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to fetch series data: {response.status}")
                    series_data = await response.json()
                   
                episodes_url = f"{self.base_url}/series/{tvdb_id}/episodes/default"
                async with session.get(episodes_url, headers=headers) as response:
                    if response.status == 200:
                        episodes_data = await response.json()
                    else:
                        episodes_data = {"data": []}
                       
                artwork_url = f"{self.base_url}/series/{tvdb_id}/artworks"
                async with session.get(artwork_url, headers=headers) as response:
                    if response.status == 200:
                        artwork_data = await response.json()
                    else:
                        artwork_data = {"data": []}

                return {
                    "id": tvdb_id,
                    "name": series_data.get("data", {}).get("name"),
                    "overview": series_data.get("data", {}).get("overview"),
                    "first_aired": series_data.get("data", {}).get("firstAired"),
                    "network": series_data.get("data", {}).get("network"),
                    "status": series_data.get("data", {}).get("status"),
                    "genres": series_data.get("data", {}).get("genres", []),
                    "episodes": episodes_data.get("data", []),
                    "artworks": artwork_data.get("data", []),
                    "image_url": series_data.get("data", {}).get("image"),
                    "rating": series_data.get("data", {}).get("rating"),
                    "runtime": series_data.get("data", {}).get("runtime"),
                    "language": series_data.get("data", {}).get("originalLanguage"),
                    "aliases": series_data.get("data", {}).get("aliases", []),
                }
               
        except Exception as e:
            self.logger.error(f"Error fetching TVDB data: {str(e)}")
            raise

    async def fetch_episode_data(self, episode_id: str) -> Optional[Dict]:
        await self.ensure_token()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/episodes/{episode_id}/extended"
                headers = {"Authorization": f"Bearer {self.token}"}
               
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching episode data: {str(e)}")
            return None

    async def fetch_artwork(self, series_id: str) -> Optional[List[Dict]]:
        await self.ensure_token()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/series/{series_id}/artworks"
                headers = {"Authorization": f"Bearer {self.token}"}
               
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching artwork: {str(e)}")
            return None

    async def fetch_translations(self, series_id: str, language: str) -> Optional[Dict]:
        await self.ensure_token()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/series/{series_id}/translations/{language}"
                headers = {"Authorization": f"Bearer {self.token}"}
               
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching translations: {str(e)}")
            return None

    async def close(self):
        try:
            if isinstance(self.session, aiohttp.ClientSession) and not self.session.closed:
                await self.session.close()
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
        finally:
            self.session = None

class TMDBClient:
   
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://api.themoviedb.org/3" 
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.configuration = None
        self.image_base_url = None
       
    async def __aenter__(self):
        await self._initialize_session()
        return self
       
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
       
    async def fetch_configuration(self) -> bool:
        try:
            if not self.session:
                await self._initialize_session()

            data = await self._make_request('GET', '/configuration')
            if not data:
                raise ValueError("Failed to fetch configuration")

            self.configuration = data
            if "images" in self.configuration:
                self.image_base_url = self.configuration["images"].get("secure_base_url")
            return True

        except Exception as e:
            self.logger.error(f"Failed to fetch TMDB configuration: {str(e)}")
            raise

    async def _initialize_session(self) -> None:
        if self.session is None or self.session.closed:
            timeout = ClientTimeout(total=30, connect=10, sock_read=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=10,
                    force_close=True,
                    enable_cleanup_closed=True
                )
            )

    async def close(self) -> None:
        try:
            if isinstance(self.session, aiohttp.ClientSession) and not self.session.closed:
                await self.session.close()
                await asyncio.sleep(0.25) 
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
        finally:
            self.session = None


    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        if not self.session:
            await self._initialize_session()

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "accept": "application/json"
        }

        if 'headers' in kwargs:
            kwargs['headers'].update(headers)
        else:
            kwargs['headers'] = headers

        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"Making request to {url}")

        retries = 3
        for attempt in range(retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 401:
                        error_text = await response.text()
                        self.logger.error(f"Authentication error. URL: {url}, Response: {error_text}")
                        raise ValueError(f"Invalid TMDB access token. Error: {error_text}")
                       
                    if response.status == 404:
                        return {}
                       
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        await asyncio.sleep(retry_after)
                        continue
                       
                    response.raise_for_status()
                    return await response.json()
                   
            except aiohttp.ClientError as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
               
        raise RuntimeError("Max retries exceeded")

    async def create_request_token(self, redirect_to: Optional[str] = None) -> Dict:
        data = {"redirect_to": redirect_to} if redirect_to else {}
        return await self._make_request('POST', '/auth/request_token', json=data)

    async def create_access_token(self, request_token: str) -> Dict:
        return await self._make_request(
            'POST', 
            '/auth/access_token',
            json={"request_token": request_token}
        )

    async def delete_access_token(self, access_token: str) -> Dict:
        return await self._make_request(
            'DELETE',
            '/auth/access_token',
            json={"access_token": access_token}
        )

    async def get_list(self, list_id: int, language: str = "en-US", page: int = 1) -> Dict:
        params = {
            "language": language,
            "page": page
        }
        return await self._make_request('GET', f'/list/{list_id}', params=params)

    async def update_list(self, list_id: int, updates: Dict) -> Dict:
        return await self._make_request('PUT', f'/list/{list_id}', json=updates)

    async def create_list(self, name: str, description: str = "", 
                         iso_639_1: str = "en", iso_3166_1: str = "US",
                         public: bool = True) -> Dict:
        data = {
            "name": name,
            "description": description,
            "iso_639_1": iso_639_1,
            "iso_3166_1": iso_3166_1,
            "public": public
        }
        return await self._make_request('POST', '/list', json=data)

    async def clear_list(self, list_id: int) -> Dict:
        return await self._make_request('GET', f'/list/{list_id}/clear')

    async def delete_list(self, list_id: int) -> Dict:
        return await self._make_request('DELETE', f'/list/{list_id}')

    async def add_items(self, list_id: int, items: List[Dict]) -> Dict:
        return await self._make_request('POST', f'/list/{list_id}/items', json={"items": items})

    async def update_items(self, list_id: int, items: List[Dict]) -> Dict:
        return await self._make_request('PUT', f'/list/{list_id}/items', json={"items": items})

    async def remove_items(self, list_id: int, items: List[Dict]) -> Dict:
        return await self._make_request('DELETE', f'/list/{list_id}/items', json={"items": items})

    async def check_item_status(self, list_id: int, media_id: int, media_type: str) -> Dict:
        params = {
            "media_id": media_id,
            "media_type": media_type
        }
        return await self._make_request('GET', f'/list/{list_id}/item_status', params=params)

    async def get_account_lists(self, account_id: str, page: int = 1) -> Dict:
        params = {"page": page}
        return await self._make_request('GET', f'/account/{account_id}/lists', params=params)

    async def get_account_favorite_movies(self, account_id: str, page: int = 1,
                                         language: str = "en-US",
                                         sort_by: str = "created_at.asc") -> Dict:
        params = {
            "page": page,
            "language": language,
            "sort_by": sort_by
        }
        return await self._make_request('GET', f'/account/{account_id}/movie/favorites', params=params)

    async def get_account_favorite_tv(self, account_id: str, page: int = 1,
                                     language: str = "en-US",
                                     sort_by: str = "created_at.asc") -> Dict:
        params = {
            "page": page,
            "language": language,
            "sort_by": sort_by
        }
        return await self._make_request('GET', f'/account/{account_id}/tv/favorites', params=params)

    async def get_account_movie_recommendations(self, account_id: str, page: int = 1,
                                               language: str = "en-US") -> Dict:
        params = {
            "page": page,
            "language": language
        }
        return await self._make_request(
            'GET',
            f'/account/{account_id}/movie/recommendations',
            params=params
        )

    async def get_account_tv_recommendations(self, account_id: str, page: int = 1,
                                            language: str = "en-US") -> Dict:
        params = {
            "page": page,
            "language": language
        }
        return await self._make_request(
            'GET',
            f'/account/{account_id}/tv/recommendations',
            params=params
        )

    async def get_movie_watchlist(self, account_id: str, page: int = 1,
                                 language: str = "en-US",
                                 sort_by: str = "created_at.asc") -> Dict:
        params = {
            "page": page,
            "language": language,
            "sort_by": sort_by
        }
        return await self._make_request(
            'GET',
            f'/account/{account_id}/movie/watchlist',
            params=params
        )

    async def get_tv_watchlist(self, account_id: str, page: int = 1,
                               language: str = "en-US",
                               sort_by: str = "created_at.asc") -> Dict:
        params = {
            "page": page,
            "language": language,
            "sort_by": sort_by
        }
        return await self._make_request(
            'GET',
            f'/account/{account_id}/tv/watchlist',
            params=params
        )

    async def get_rated_movies(self, account_id: str, page: int = 1,
                               language: str = "en-US",
                               sort_by: str = "created_at.asc") -> Dict:
        params = {
            "page": page,
            "language": language,
            "sort_by": sort_by
        }
        return await self._make_request(
            'GET',
            f'/account/{account_id}/movie/rated',
            params=params
        )

    async def get_rated_tv(self, account_id: str, page: int = 1,
                             language: str = "en-US",
                             sort_by: str = "created_at.asc") -> Dict:
        params = {
            "page": page,
            "language": language,
            "sort_by": sort_by
        }
        return await self._make_request(
            'GET',
            f'/account/{account_id}/tv/rated',
            params=params
        )

    async def fetch_media_data(self, title: str, year: Optional[int], media_type: str) -> Dict:
        try:
            search_results = await self._search_media(title, year, media_type)
            if not search_results:
                self.logger.warning(f"No results found for {title} ({year})")
                return {}

            media_id = search_results[0]['id']
           
            details = await self._fetch_details(media_id, media_type)
            if not details:
                return {}

            return self._process_media_data(details, media_type)

        except Exception as e:
            self.logger.error(f"Error fetching media data for {title}: {str(e)}")
            raise

    async def _search_media(self, title: str, year: Optional[int], media_type: str) -> List[Dict]:
        search_type = 'tv' if media_type.lower() in ['show', 'tv'] else media_type
           
        params = {
            "query": title,
            "language": "en-US",
            "include_adult": "false"
        }
        if year:
            params["year" if search_type == "movie" else "first_air_date_year"] = str(year)

        try:
            self.logger.debug(f"TMDB search params: {params}")
            results = await self._make_request('GET', f'/search/{search_type}', params=params)
            return results.get('results', [])
        except Exception as e:
            self.logger.error(f"TMDB fetch failed for {title}: {str(e)}")
            raise

    async def _fetch_details(self, media_id: int, media_type: str) -> Dict:
        params = {
            "append_to_response": (
                "credits,keywords,videos,reviews,similar,recommendations,"
                "watch/providers,external_ids,content_ratings,images,"
                "aggregate_credits"
            ),
            "language": "en-US"
        }

        try:
            data = await self._make_request('GET', f'/{media_type}/{media_id}', params=params)

            if media_type == "tv" and data.get("number_of_seasons", 0) > 0:
                seasons_data = await self._fetch_all_seasons(media_id, data["number_of_seasons"])
                data["seasons_data"] = seasons_data

            return data

        except Exception as e:
            self.logger.error(f"Error fetching details for {media_id}: {str(e)}")
            raise

    async def _fetch_all_seasons(self, series_id: int, num_seasons: int) -> List[Dict]:
        seasons = []
        for season_num in range(1, num_seasons + 1):
            try:
                params = {
                    "append_to_response": "credits,videos,images",
                    "language": "en-US"
                }
                season_data = await self._make_request(
                    'GET',
                    f'/tv/{series_id}/season/{season_num}',
                    params=params
                )
                if season_data:
                    seasons.append(season_data)
            except Exception as e:
                self.logger.error(f"Error fetching season {season_num}: {str(e)}")
                continue
        return seasons

    def _process_media_data(self, data: Dict, media_type: str) -> Dict:
        processed = {
            'tmdb_id': data.get('id'),
            'title': data.get('title' if media_type == 'movie' else 'name'),
            'original_title': data.get('original_title' if media_type == 'movie' else 'original_name'),
            'overview': data.get('overview'),
            'popularity': data.get('popularity'),
            'vote_average': data.get('vote_average'),
            'vote_count': data.get('vote_count'),
            'release_date': data.get('release_date' if media_type == 'movie' else 'first_air_date'),
            'genres': [genre['name'] for genre in data.get('genres', [])],
            'runtime': data.get('runtime') if media_type == 'movie' else data.get('episode_run_time', [None])[0],
            'status': data.get('status'),
            'tagline': data.get('tagline'),
            'poster_path': data.get('poster_path'),
            'backdrop_path': data.get('backdrop_path'),
            'budget': data.get('budget') if media_type == 'movie' else None,
            'revenue': data.get('revenue') if media_type == 'movie' else None,
            'keywords': [kw['name'] for kw in data.get('keywords', {}).get('keywords', [])],
            'videos': self._process_videos(data.get('videos', {}).get('results', [])),
            'credits': self._process_credits(data.get('credits', {})),
            'reviews': self._process_reviews(data.get('reviews', {}).get('results', [])),
            'similar': self._process_similar(data.get('similar', {}).get('results', [])),
            'recommendations': self._process_recommendations(data.get('recommendations', {}).get('results', [])),
            'watch_providers': data.get('watch/providers', {}).get('results', {}),
            'images': self._process_images(data.get('images', {}))
        }

        if media_type == "tv":
            processed.update({
                'created_by': data.get('created_by', []),
                'episode_run_time': data.get('episode_run_time', []),
                'first_air_date': data.get('first_air_date'),
                'last_air_date': data.get('last_air_date'),
                'networks': data.get('networks', []),
                'number_of_episodes': data.get('number_of_episodes'),
                'number_of_seasons': data.get('number_of_seasons'),
                'seasons': data.get('seasons_data', []),
                'type': data.get('type'),
                'origin_country': data.get('origin_country', [])
            })

        return processed

    def _process_videos(self, videos: List[Dict]) -> List[Dict]:
        return [{
            'key': video['key'],
            'name': video['name'],
            'site': video['site'],
            'type': video['type'],
            'official': video.get('official', True)
        } for video in videos if video.get('site') == 'YouTube']

    def _process_credits(self, credits: Dict) -> Dict:
        return {
            'cast': [{
                'id': person['id'],
                'name': person['name'],
                'character': person.get('character'),
                'order': person.get('order'),
                'profile_path': person.get('profile_path')
            } for person in credits.get('cast', [])],
            'crew': [{
                'id': person['id'],
                'name': person['name'],
                'job': person.get('job'),
                'department': person.get('department')
            } for person in credits.get('crew', [])]
        }

    def _process_reviews(self, reviews: List[Dict]) -> List[Dict]:
        return [{
            'author': review['author'],
            'content': review['content'],
            'created_at': review['created_at'],
            'rating': review.get('author_details', {}).get('rating')
        } for review in reviews]

    def _process_similar(self, similar: List[Dict]) -> List[Dict]:
        return [{
            'id': item['id'],
            'title': item.get('title', item.get('name')),
            'overview': item.get('overview'),
            'poster_path': item.get('poster_path')
        } for item in similar]

    def _process_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        return [{
            'id': item['id'],
            'title': item.get('title', item.get('name')),
            'overview': item.get('overview'),
            'poster_path': item.get('poster_path')
        } for item in recommendations]

    def _process_images(self, images: Dict) -> Dict:
        return {
            'posters': [{
                'file_path': img['file_path'],
                'width': img['width'],
                'height': img['height'],
                'vote_average': img.get('vote_average', 0)
            } for img in images.get('posters', [])],
            'backdrops': [{
                'file_path': img['file_path'],
                'width': img['width'],
                'height': img['height'],
                'vote_average': img.get('vote_average', 0)
            } for img in images.get('backdrops', [])]
        }

    def get_image_url(self, path: str, size: str = "original") -> str:
        if not path or not self.image_base_url:
            return ""
        return f"{self.image_base_url}{size}{path}"

class CleanupWorker(QObject):
    finished = Signal()

    def __init__(self, app):
        super().__init__()
        self.app = app

    def cleanup(self):
        try:
            if hasattr(self.app, 'recommender') and self.app.recommender:
                self.app.recommender._stop_flag = True
                if hasattr(self.app.recommender, 'similarity_thread') and self.app.recommender.similarity_thread.isRunning():
                    self.app.recommender.similarity_worker.stop()
                    self.app.recommender.similarity_thread.quit()
                    self.app.recommender.similarity_thread.wait(3000)
                self.app.recommender.cleanup()
                self.app.recommender = None

            if hasattr(self.app, 'scan_thread') and self.app.scan_thread.isRunning():
                self.app.scan_thread.stop()
                self.app.scan_thread.wait(5000)

            if self.app.background_trainer:
                self.app.background_trainer._stop_flag = True
                if self.app.background_trainer.isRunning():
                    self.app.background_trainer.wait(2000)
                self.app.background_trainer = None

            if self.app.db:
                self.app.db.cleanup_old_cache()
                self.app.db.close()
                self.app.db = None

            if self.app.media_scanner:
                if hasattr(self.app.media_scanner, 'close'):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.app.media_scanner.close())
                        loop.close()
                    except Exception:
                        pass
                self.app.media_scanner = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if hasattr(self.app, 'poster_downloader'):
                self.app.poster_downloader.cleanup()

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        finally:
            self.finished.emit()
                                       
class MediaRecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recommend For Plex")
        self.setMinimumSize(1200, 800)
       
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        temp_dir = tempfile.gettempdir()
        log_file_path = os.path.join(temp_dir, 'plex_recommender.log')
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
       
        self.media_widgets = {}
        self.background_trainer = None
       
        self.init_ui()
       
        self.load_config()
       
        self.db = None
        self.recommender = None
        self.media_scanner = None
        self.poster_downloader = PosterDownloader()
       
        QTimer.singleShot(0, self.init_background_components)

    def init_background_components(self):
        try:
            print("Initializing background components...")
            
            self.db = Database.get_instance()
            print("Database initialized")
            
            self.recommender = RecommendationEngine(self.db)
            print("Recommendation engine initialized")
            
            self.media_scanner = MediaScanner()
            print("Media scanner initialized")
            
            self.schedule_cache_updates()
            print("Cache updates scheduled")
            
            if not hasattr(self, 'background_trainer') or self.background_trainer is None:
                self.background_trainer = BackgroundTrainer(self.recommender, self.db)
                self.background_trainer.training_status.connect(self.update_progress)
                self.background_trainer.training_status.connect(self._handle_training_status)
                self.background_trainer.start()
                print("Background trainer initialized and started")
                
            try:
                self.media_scanner.configure(
                    plex_url=self.plex_url.text(),
                    plex_token=self.plex_token.text(),
                    tvdb_key=self.tvdb_key.text(),
                    tmdb_key=self.tmdb_key.text()
                )
                print("Media scanner configured")
            except Exception as e:
                self.logger.error(f"Warning: Could not configure media scanner: {e}")
            
            QTimer.singleShot(1000, self._load_initial_recommendations)
            
        except Exception as e:
            self.logger.error(f"Error in background initialization: {e}")
            print(f"Error initializing components: {e}")

    def _handle_training_status(self, message: str):
        try:
            if "error" in message.lower():
                self.logger.error(message)
                self.show_error_notification(message)
            elif "complete" in message.lower():
                self._refresh_recommendations()
        except Exception as e:
            self.logger.error(f"Error handling training status: {e}")


    def _load_initial_recommendations(self):
        try:
            print("\nLoading initial recommendations...")
            cursor = self.db.conn.cursor()
            
            
            cursor.execute("SELECT COUNT(*) FROM media_items WHERE type = 'movie'")
            movie_count = cursor.fetchone()[0]
            print(f"Found {movie_count} movies in database")
            
            
            cursor.execute("SELECT COUNT(*) FROM media_items WHERE type = 'show'")
            show_count = cursor.fetchone()[0]
            print(f"Found {show_count} shows in database")
            
            if movie_count == 0 and show_count == 0:
                print("No items found in database")
                self.display_item(None, 'movie')
                self.display_item(None, 'show')
                return
            
            print("Loading movie recommendation...")
            if movie_count > 0:
                self.show_next_recommendation('movie')
            else:
                self.display_item(None, 'movie')
                
            print("Loading show recommendation...")
            if show_count > 0:
                self.show_next_recommendation('show')
            else:
                self.display_item(None, 'show')
                
        except Exception as e:
            self.logger.error(f"Error loading initial recommendations: {e}")
            print(f"Error loading recommendations: {e}")
            self.display_item(None, 'movie')
            self.display_item(None, 'show')
            
    def schedule_cache_updates(self):
        db = self.db
        if db and db.conn:
            db.cleanup_old_cache()
        else:
            self.logger.warning("Database connection is closed during cache cleanup.")
        QTimer.singleShot(24 * 60 * 60 * 1000, self.schedule_cache_updates)
                       
    def init_ui(self):
        self.setWindowTitle("Recommend For Plex")
        self.setMinimumSize(1200, 800)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.init_movies_tab()
        self.init_tv_tab()
        self.init_config_tab()
       
    def init_movies_tab(self):
        self.movies_tab = self.create_media_tab("movie")
        self.tabs.addTab(self.movies_tab, "Movies")
       
    def init_tv_tab(self):
        self.tv_tab = self.create_media_tab("show")
        self.tabs.addTab(self.tv_tab, "TV Shows")
       
    def create_media_tab(self, media_type):
        tab_widget = QWidget()
        layout = QVBoxLayout()
        
        media_layout = QHBoxLayout()
        
        poster_label = QLabel()
        poster_label.setFixedSize(300, 450)
        poster_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        media_layout.addWidget(poster_label)
        
        details_layout = QVBoxLayout()
        
        title_label = QLabel()
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_label.setTextFormat(Qt.TextFormat.PlainText)
        title_label.setWordWrap(True)
        details_layout.addWidget(title_label)
        
        year_runtime_label = QLabel()
        year_runtime_label.setTextFormat(Qt.TextFormat.PlainText)
        details_layout.addWidget(year_runtime_label)
        
        genres_label = QLabel()
        genres_label.setTextFormat(Qt.TextFormat.PlainText)
        genres_label.setWordWrap(True)
        details_layout.addWidget(genres_label)
        
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        summary_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        details_layout.addWidget(summary_text)
        
        media_layout.addLayout(details_layout)
        layout.addLayout(media_layout)
        
        rating_layout = QHBoxLayout()
        rating_label = QLabel("Rate (1-10):")
        rating_layout.addWidget(rating_label)
        
        rating_buttons = QButtonGroup()
        for i in range(10):
            rating = i + 1
            button = QPushButton(str(rating))
            button.setFixedWidth(40)
            button.clicked.connect(lambda checked, r=rating: self.submit_rating(r, media_type))
            rating_layout.addWidget(button)
            rating_buttons.addButton(button)
        
        layout.addLayout(rating_layout)
        
        tab_widget.setLayout(layout)
        
        self.media_widgets[media_type] = {
            'poster_label': poster_label,
            'title_label': title_label,
            'year_runtime_label': year_runtime_label,
            'genres_label': genres_label,
            'summary_text': summary_text,
            'current_item_id': None
        }
        
        return tab_widget
       
    def init_config_tab(self):
        config_widget = QWidget()
        layout = QVBoxLayout()

        config_button_layout = QHBoxLayout()
        save_config_button = QPushButton("Save Configuration")
        save_config_button.clicked.connect(self.save_config)
        config_button_layout.addWidget(save_config_button)

        load_libraries_button = QPushButton("Load Libraries")
        load_libraries_button.clicked.connect(self.load_plex_libraries)
        config_button_layout.addWidget(load_libraries_button)

        layout.addLayout(config_button_layout)

        plex_group = QGroupBox("Plex Configuration")
        plex_layout = QFormLayout()
        self.plex_url = QLineEdit()
        self.plex_token = QLineEdit()
        plex_layout.addRow("Plex URL:", self.plex_url)
        plex_layout.addRow("Plex Token:", self.plex_token)
        plex_group.setLayout(plex_layout)
        layout.addWidget(plex_group)

        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout()
        self.tvdb_key = QLineEdit()
        self.tmdb_key = QLineEdit()
        api_layout.addRow("TVDB API Key:", self.tvdb_key)
        api_layout.addRow("TMDB API Key:", self.tmdb_key)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        library_group = QGroupBox("Library Selection")
        library_layout = QVBoxLayout()
        self.library_list = QListWidget()
        self.library_list.setSelectionMode(QAbstractItemView.MultiSelection)
        library_layout.addWidget(self.library_list)
        library_group.setLayout(library_layout)
        layout.addWidget(library_group)

        model_group = QGroupBox("Model Management")
        model_layout = QVBoxLayout()

        reset_button = QPushButton("Reset Model")
        reset_button.setToolTip("Clear all ratings and start fresh")
        reset_button.clicked.connect(self.reset_model)
        model_layout.addWidget(reset_button)

        scan_button = QPushButton("Scan Libraries")
        scan_button.clicked.connect(self.scan_libraries)
        model_layout.addWidget(scan_button)
       
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        layout.addWidget(self.progress_text)

        config_widget.setLayout(layout)
        self.tabs.addTab(config_widget, "Configuration")
                                   
    def show_rating_dialog(self, rating_type, media_type):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Rate how much you {'liked' if rating_type == 'like' else 'disliked'} it")
       
        layout = QVBoxLayout()
       
        star_layout = QHBoxLayout()
        star_group = QButtonGroup()
       
        for i in range(5):
            radio = QRadioButton(str(i + 1))
            star_layout.addWidget(radio)
            star_group.addButton(radio, i + 1)
           
        layout.addLayout(star_layout)
       
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
       
        dialog.setLayout(layout)
       
        if dialog.exec_() == QDialog.Accepted:
            stars = star_group.checkedId()
            if stars > 0:
                self.submit_rating(rating_type, stars, media_type)
               
    def submit_rating(self, rating: int, media_type: str):
        current_id = self.media_widgets[media_type]['current_item_id']
        if current_id is not None:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute('''
                    SELECT * FROM media_items WHERE id = ?
                ''', (current_id,))
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    content_data = dict(zip(columns, row))
                    
                    self.recommender.train_with_feedback(current_id, rating, content_data)
                    
                    if hasattr(self, 'background_trainer') and self.background_trainer:
                        self.background_trainer.queue_item_for_training(content_data, priority=True)
                    
                    self.show_next_recommendation(media_type)
                cursor.close()
            except Exception as e:
                self.logger.error(f"Error submitting rating: {e}")
                QMessageBox.warning(self, "Error", f"Failed to submit rating: {str(e)}")
       
    def load_config(self):
        config_path = 'config.json'
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.plex_url.setText(config.get('plex_url', ''))
                    self.plex_token.setText(config.get('plex_token', ''))
                    self.tvdb_key.setText(config.get('tvdb_key', ''))
                    self.tmdb_key.setText(config.get('tmdb_key', ''))
                print("Configuration loaded successfully!")
            else:
                print("Configuration file not found.")
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

    def display_item(self, item: Optional[Dict], media_type: str) -> None:
        print(f"\nAttempting to display item for {media_type}")
        
        widgets = self.media_widgets[media_type]
        
        if not item or not isinstance(item, dict):
            print(f"No valid item to display for {media_type}")
            widgets['current_item_id'] = None
            widgets['title_label'].setText("No Items Available")
            widgets['year_runtime_label'].setText("")
            widgets['genres_label'].setText("")
            widgets['summary_text'].setPlainText(
                "No recommendations available. Please ensure your libraries are scanned."
            )
            self.poster_downloader._set_default_poster(widgets['poster_label']) 
            return

        try:
            widgets['current_item_id'] = item.get('id')
            title = item.get('title', 'Unknown Title')
            print(f"Displaying: {title}")
            
            widgets['title_label'].setText(title)
            
            year = item.get('year', '')
            runtime = item.get('runtime', '')
            runtime_str = f"{int(runtime) // 60} min" if str(runtime).isdigit() else runtime
            year_runtime = ' | '.join(filter(None, [str(year), runtime_str]))
            widgets['year_runtime_label'].setText(year_runtime)
            
            genres = item.get('genres', [])
            if isinstance(genres, str):
                genres = json.loads(genres) if genres.startswith('[') else genres.split(',')
            genres_str = ", ".join(str(genre).strip() for genre in genres if genre)
            widgets['genres_label'].setText(genres_str)
            
            summary = item.get('summary', 'No summary available')
            widgets['summary_text'].setPlainText(summary)
            
            self._update_poster_image(item, widgets['poster_label'])
            
            print(f"Successfully displayed item: {title}")

        except Exception as e:
            print(f"Error displaying item: {str(e)}")
            self.logger.error(f"Error displaying item: {str(e)}")
            widgets['current_item_id'] = None
            widgets['title_label'].setText("Error Loading Item")
            widgets['year_runtime_label'].setText("")
            widgets['genres_label'].setText("")
            widgets['summary_text'].setPlainText(f"Error loading item: {str(e)}")
            self.poster_downloader._set_default_poster(widgets['poster_label'])
                
    def show_next_recommendation(self, media_type):
        print(f"\nFetching next recommendation for {media_type}")
        try:
            if not hasattr(self, 'recommender') or self.recommender is None:
                print("Recommender not initialized")
                self.display_item(None, media_type)
                return

            print("Getting recommendation from engine...")
            recommendation = self.recommender.get_next_recommendation(media_type)

            if not recommendation:
                print("No recommendation returned")
                self.display_item(None, media_type)
                return

            if not isinstance(recommendation, dict):
                print(f"Invalid recommendation type: {type(recommendation)}")
                self.display_item(None, media_type)
                return

            if 'id' not in recommendation:
                print("Recommendation missing ID")
                self.display_item(None, media_type)
                return

            print(f"Displaying recommendation: {recommendation.get('title', 'Unknown')}")
            self.display_item(recommendation, media_type)

        except Exception as e:
            print(f"Error in show_next_recommendation: {str(e)}")
            self.logger.error(f"Error showing next recommendation: {str(e)}")
            self.display_item(None, media_type)


    def _update_poster_image(self, item: Dict, poster_label: QLabel) -> None:
        if not item or not item.get('poster_url'):
            self.poster_downloader._set_default_poster(poster_label)
            return

        try:
            url = self._get_full_poster_url(item['poster_url'])
            if not url:
                self.poster_downloader._set_default_poster(poster_label)
                return

            self.poster_downloader.download_poster(url, poster_label)

        except Exception as e:
            self.logger.error(f"Error updating poster: {str(e)}")
            self.poster_downloader._set_default_poster(poster_label)

    
    def _get_full_poster_url(self, poster_url: str) -> Optional[str]:
        try:
            if poster_url.startswith('/'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                   
                plex_base_url = config.get('plex_url', '').rstrip('/')
                if not plex_base_url:
                    return None

                url = f"{plex_base_url}{poster_url}"
               
                plex_token = config.get('plex_token', '')
                if plex_token:
                    url += f"?X-Plex-Token={plex_token}"
                   
                return url
            return poster_url

        except Exception as e:
            self.logger.error(f"Error constructing poster URL: {str(e)}")
            return None

    def cleanup_background_trainer(self):
        if hasattr(self, 'background_trainer') and self.background_trainer:
            try:
                self.background_trainer._stop_flag.set()
                start_time = time.time()
                while self.background_trainer.isRunning() and (time.time() - start_time) < 5:
                    QApplication.processEvents()
                    time.sleep(0.1)
                
                if self.background_trainer.isRunning():
                    self.background_trainer.terminate()
                    self.background_trainer.wait(1000)
                
                self.background_trainer = None
            except Exception as e:
                self.logger.error(f"Error cleaning up background trainer: {e}")
       
    def scan_libraries(self):
        selected = self.library_list.selectedItems()
        library_names = [item.text().split(' (')[0] for item in selected]

        if not library_names:
            QMessageBox.warning(self, "Warning", "No libraries selected for scanning.")
            return

        try:
            self.media_scanner.configure(
                plex_url=self.plex_url.text(),
                plex_token=self.plex_token.text(),
                tvdb_key=self.tvdb_key.text(),
                tmdb_key=self.tmdb_key.text()
            )

            if hasattr(self, 'scan_thread') and self.scan_thread.isRunning():
                self.scan_thread.stop()
                self.scan_thread.wait()

            self.scan_thread = ScanThread(self.media_scanner, library_names)
            self.scan_thread.progress.connect(self.update_progress)
            self.scan_thread.finished.connect(self._on_scan_completed)
            self.scan_thread.start()

            if self.background_trainer:
                self.background_trainer._stop_flag.set()
                self.background_trainer.wait()
                self.background_trainer._stop_flag.clear()
                self.background_trainer.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to scan libraries: {str(e)}")

    def _on_scan_completed(self):
        self.progress_text.append("Scan completed!")
        if self.recommender:
            self.recommender.knowledge_graph = self.recommender._initialize_knowledge_graph()
    
        QTimer.singleShot(1000, self._load_initial_recommendations)

    def show_error_notification(self, message: str):
        QMessageBox.warning(self, "Training Error", message)

    def _refresh_recommendations(self):
        try:
            self.show_next_recommendation('movie')
            self.show_next_recommendation('show')
        except Exception as e:
            self.logger.error(f"Error refreshing recommendations: {e}")
        
    def closeEvent(self, event):
        print("\nInitiating application shutdown...")
        
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Shutting Down")
        layout = QVBoxLayout()
        status_label = QLabel("Cleaning up resources...")
        layout.addWidget(status_label)
        progress_dialog.setLayout(layout)
        progress_dialog.setModal(True)
        progress_dialog.show()
        QApplication.processEvents()

        cleanup_thread = QThread()
        cleanup_worker = CleanupWorker(self)
        cleanup_worker.moveToThread(cleanup_thread)
        
        cleanup_thread.started.connect(cleanup_worker.cleanup)
        cleanup_worker.finished.connect(cleanup_thread.quit)
        cleanup_worker.finished.connect(cleanup_worker.deleteLater)
        cleanup_thread.finished.connect(cleanup_thread.deleteLater)
        cleanup_worker.finished.connect(progress_dialog.close)
        cleanup_worker.finished.connect(lambda: event.accept())
        
        QTimer.singleShot(10000, lambda: self._force_cleanup(cleanup_thread, event, progress_dialog))
        
        cleanup_thread.start()

    def _force_cleanup(self, thread: QThread, event: QCloseEvent, dialog: QDialog):
        if thread.isRunning():
            print("Cleanup taking too long, forcing shutdown...")
            thread.terminate()
            thread.wait(1000)
            dialog.close()
            event.accept()

    def update_progress(self, message):
        self.progress_text.append(message)
       
    def scan_completed(self):
        self.progress_text.append("Scan completed!")

    def reset_model(self):
        reply = QMessageBox.question(
            self,
            "Reset Recommendation System",
            "Are you sure you want to reset the recommendation system?\n\n"
            "This will:\n"
            "• Clear all your ratings\n"
            "• Reset learning patterns\n"
            "• Clear cached embeddings\n"
            "• Start fresh with no preferences\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
       
        if reply == QMessageBox.Yes:
            try:
                result = self.recommender.reset_model()
                QMessageBox.information(self, "Success", result)
                self.show_next_recommendation('movie')
                self.show_next_recommendation('show')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset system: {str(e)}")

    def load_plex_libraries(self):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
           
            self.library_list.clear()
           
            plex = PlexServer(self.plex_url.text(), self.plex_token.text())
           
            for library in plex.library.sections():
                if library.type in ['movie', 'show']:
                    item = QListWidgetItem(f"{library.title} ({library.type})")
                    self.library_list.addItem(item)
                   
            QMessageBox.information(self, "Success", "Libraries loaded successfully!")
           
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load libraries: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
           
    def save_config(self):
        config = {
            'plex_url': self.plex_url.text().strip(),
            'plex_token': self.plex_token.text().strip(),
            'tvdb_key': self.tvdb_key.text().strip(),
            'tmdb_key': self.tmdb_key.text().strip()
        }

        try:
            with open('config.json', 'w') as config_file:
                json.dump(config, config_file, indent=4)

            QMessageBox.information(self, "Success", "Configuration saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")


           
    def populate_library_list(self):
        try:
            plex = PlexServer(self.plex_url.text(), self.plex_token.text())
            self.library_list.clear()
            for library in plex.library.sections():
                self.library_list.addItem(library.title)
        except Exception as e:
            self.show_error("Error connecting to Plex", str(e))
           
    def show_error(self, title, message):
        QMessageBox.critical(self, title, message)

class ContentEncoder(nn.Module):
    def __init__(self, input_dim=1540, hidden_dim=384, output_dim=384):
        super().__init__()
        self.input_dim = 2180
        print(f"Initializing ContentEncoder with dims: input={self.input_dim}, hidden={hidden_dim}, output={output_dim}")
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        
        self.attention = nn.Linear(output_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.attention.weight)
        
        self.debug = True

    def forward(self, x):
        if self.debug:
            print(f"\nContentEncoder forward pass:")
            print(f"Input shape: {x.shape}")
        
        x = self.fc1(x)
        if self.debug:
            print(f"After fc1 shape: {x.shape}")
        
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if self.debug:
            print(f"After fc2 shape: {x.shape}")
        
        x = F.gelu(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        if self.debug:
            print(f"After fc3 shape: {x.shape}")
        
        x = self.norm3(x)
        
        attention = torch.sigmoid(self.attention(x))
        x = x * attention
        
        if self.debug:
            print(f"Final output shape: {x.shape}")
            print(f"Output mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
            
        return x

class GenreEncoder(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=128, output_dim=256):

        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):

        x = F.gelu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        return x

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        return x

class GraphConvNetwork(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=341):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=341, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = GATConv(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x

class GraphSageNetwork(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=342):  
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x

class SimilarityWorker(QObject):
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, engine, reference_item, candidates):
        super().__init__()
        self.engine = engine
        self.reference_item = reference_item
        self.candidates = candidates
        self._stop = False

    def process_batch(self):
        try:
            similarities = []
            batch_size = 5
            processed = 0
            total = len(self.candidates)

            
            with torch.no_grad():
                reference_embedding = self.engine._compute_embedding(self.reference_item)
                reference_cpu = reference_embedding.cpu()
                reference_cpu = F.normalize(reference_cpu, p=2, dim=1).squeeze()

            for i in range(0, total, batch_size):
                if self._stop:
                    break

                batch = self.candidates[i:i + batch_size]
                batch_similarities = []

                
                for item in batch:
                    if self._stop:
                        break

                    try:
                        
                        with torch.no_grad():
                            other_embedding = self.engine._compute_embedding(item)
                            other_cpu = other_embedding.cpu()
                            other_cpu = F.normalize(other_cpu, p=2, dim=1).squeeze()
                            
                            
                            similarity = float(torch.dot(reference_cpu, other_cpu))
                            batch_similarities.append((item['id'], item['title'], similarity))

                            
                            del other_embedding
                            del other_cpu
                            torch.cuda.empty_cache()

                    except Exception as e:
                        self.engine.logger.error(f"Error processing item {item['id']}: {e}")
                        continue

                
                similarities.extend(batch_similarities)
                processed += len(batch)
                self.progress.emit(f"Processed {processed}/{total} items")

                
                if processed % 20 == 0:
                    self.engine._store_similarities(
                        self.reference_item['id'], 
                        similarities,
                        is_final=False
                    )

            
            if not self._stop:
                self.engine._store_similarities(
                    self.reference_item['id'], 
                    similarities,
                    is_final=True
                )

            self.progress.emit("Similarity computation completed")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            
            if 'reference_embedding' in locals():
                del reference_embedding
            if 'reference_cpu' in locals():
                del reference_cpu
            torch.cuda.empty_cache()

    def stop(self):
        self._stop = True

class RecommendationEngine:
    def __init__(self, db):
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embeddings_lock = threading.Lock()
        self.preference_lock = threading.Lock()
        self.graph_lock = threading.Lock()
        self.embedding_cache = {}
        self.similarity_matrix = None
        self.last_matrix_update = None
        self.tensor_lock = threading.Lock()

        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            self.logger.warning(f"Could not download NLTK data: {e}")
            
        self._setup_models()
        self.knowledge_graph = self._initialize_knowledge_graph()
        
        self.optimizer = torch.optim.Adam([
            {'params': self.content_encoder.parameters()},
            {'params': self.genre_encoder.parameters()},
            {'params': self.temporal_encoder.parameters()},
            {'params': self.metadata_encoder.parameters()},
            {'params': self.graph_conv.parameters()}
        ], lr=1e-4)
        
        self.criterion = nn.MSELoss()
        
        self.prediction_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)

    def _compute_graph_features(self, item: Dict) -> torch.Tensor:
        try:
            with self.graph_lock:
                if 'id' not in item:
                    self.logger.error("Item missing ID for graph features")
                    return torch.zeros((1, 384), device=self.device)
                    
                node_id = f"media_{item['id']}"
                
                if node_id not in self.knowledge_graph['nodes']:
                    self.logger.warning(f"Node {node_id} not found in knowledge graph")
                    return torch.zeros((1, 384), device=self.device)
                    
                subgraph = self._extract_subgraph(node_id)
                if not subgraph['nodes']:
                    self.logger.warning(f"Empty subgraph for node {node_id}")
                    return torch.zeros((1, 384), device=self.device)
                    
                edge_index, edge_type, node_features = self._prepare_graph_data(subgraph)

                with torch.no_grad():
                    features = []
                    
                    try:
                        graph_embedding = self.graph_conv(node_features, edge_index)
                        graph_embedding = graph_embedding.mean(dim=0, keepdim=True)
                        features.append(graph_embedding)
                    except Exception as e:
                        self.logger.error(f"Error computing GCN features: {str(e)}")
                        features.append(torch.zeros((1, 341), device=self.device))

                    try:
                        attention_embedding = self.graph_attention(node_features, edge_index)
                        attention_embedding = attention_embedding.mean(dim=0, keepdim=True)
                        features.append(attention_embedding)
                    except Exception as e:
                        self.logger.error(f"Error computing GAT features: {str(e)}")
                        features.append(torch.zeros((1, 341), device=self.device))

                    try:
                        sage_embedding = self.graph_sage(node_features, edge_index)
                        sage_embedding = sage_embedding.mean(dim=0, keepdim=True)
                        features.append(sage_embedding)
                    except Exception as e:
                        self.logger.error(f"Error computing GraphSAGE features: {str(e)}")
                        features.append(torch.zeros((1, 342), device=self.device))

                    combined_embedding = torch.cat(features, dim=1)
                    combined_embedding = combined_embedding.detach()

                    return combined_embedding

        except Exception as e:
            self.logger.error(f"Error computing graph features: {str(e)}")
            return torch.zeros((1, 384), device=self.device)
        finally:
            if 'edge_index' in locals(): del edge_index
            if 'edge_type' in locals(): del edge_type
            if 'node_features' in locals(): del node_features
            if 'graph_embedding' in locals(): del graph_embedding
            if 'attention_embedding' in locals(): del attention_embedding
            if 'sage_embedding' in locals(): del sage_embedding
            torch.cuda.empty_cache()
    
    def _load_embedding(self, media_id: int) -> Optional[torch.Tensor]:
        try:
            cached_embedding = self.db.load_embedding(media_id)
            if cached_embedding is not None:
                array = np.frombuffer(cached_embedding, dtype=np.float32).copy() 
                tensor = torch.from_numpy(array).to(self.device)
                return tensor.view(1, -1)
            return None
        except Exception as e:
            self.logger.error(f"Error loading embedding for media_id {media_id}: {str(e)}")
            return None

    def _setup_models(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
            self.text_encoder = AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
            self.text_encoder.to(self.device)
            
            self.content_encoder = ContentEncoder(
                input_dim=1540,
                hidden_dim=384,
                output_dim=384
            ).to(self.device)
            
            self.genre_encoder = GenreEncoder().to(self.device)
            self.temporal_encoder = TemporalEncoder().to(self.device)
            self.metadata_encoder = MetadataEncoder().to(self.device)
            
            self.graph_conv = GraphConvNetwork().to(self.device)
            self.graph_attention = GraphAttentionNetwork().to(self.device)
            self.graph_sage = GraphSageNetwork().to(self.device)
            
            for model in [self.text_encoder, self.content_encoder, self.genre_encoder,
                        self.temporal_encoder, self.metadata_encoder, self.graph_conv,
                        self.graph_attention, self.graph_sage]:
                model.eval()
                
        except Exception as e:
            self.logger.error(f"Error setting up models: {str(e)}")
            raise

    def _extract_entities(self, text: str) -> List[str]:
        try:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            entities = []
            current_entity = []
            
            for word, tag in pos_tags:
                if tag.startswith(('NNP', 'NNPS')):
                    current_entity.append(word)
                elif current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
            
            if current_entity:
                entities.append(' '.join(current_entity))
            
            entities = list(set(entity for entity in entities if len(entity) > 1))
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []
    def train_model(self, epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2):
        try:
            
            self.content_encoder.train()
            self.genre_encoder.train()
            self.temporal_encoder.train()
            self.metadata_encoder.train()
            self.graph_conv.train()
            self.graph_attention.train()
            self.graph_sage.train()
            
            
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT m.*, f.rating 
                FROM media_items m 
                JOIN user_feedback f ON m.id = f.media_id
                ORDER BY RANDOM()
            ''')
            columns = [description[0] for description in cursor.description]
            all_data = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            if not all_data:
                self.logger.warning("No training data available")
                return
                
            
            split_idx = int(len(all_data) * (1 - validation_split))
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            
            self.optimizer = torch.optim.AdamW([
                {'params': self.content_encoder.parameters(), 'lr': 1e-4},
                {'params': self.genre_encoder.parameters(), 'lr': 1e-4},
                {'params': self.temporal_encoder.parameters(), 'lr': 1e-4},
                {'params': self.metadata_encoder.parameters(), 'lr': 1e-4},
                {'params': self.graph_conv.parameters(), 'lr': 1e-4},
                {'params': self.graph_attention.parameters(), 'lr': 1e-4},
                {'params': self.graph_sage.parameters(), 'lr': 1e-4},
                {'params': self.prediction_head.parameters(), 'lr': 1e-4}
            ], weight_decay=0.01)
            
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 5
            
            for epoch in range(epochs):
                
                self.logger.info(f"Epoch {epoch + 1}/{epochs}")
                total_train_loss = 0
                random.shuffle(train_data)
                
                
                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i:i + batch_size]
                    batch_loss = self._train_batch(batch)
                    total_train_loss += batch_loss
                    
                    if i % 100 == 0:
                        self.logger.info(f"Batch {i//batch_size + 1}: Loss = {batch_loss:.4f}")
                
                avg_train_loss = total_train_loss / (len(train_data) / batch_size)
                
                
                val_loss = self._validate(val_data, batch_size)
                
                self.logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                
                scheduler.step(val_loss)
                
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_model_state()  
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        self.logger.info("Early stopping triggered")
                        break
            
            
            self._load_model_state()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            
            self.content_encoder.eval()
            self.genre_encoder.eval()
            self.temporal_encoder.eval()
            self.metadata_encoder.eval()
            self.graph_conv.eval()
            self.graph_attention.eval()
            self.graph_sage.eval()
            self.prediction_head.eval()

    def _train_batch(self, batch: List[Dict]) -> float:
        try:
            self.optimizer.zero_grad()
            total_loss = 0
            
            for item in batch:
                
                embedding = self._compute_embedding(item)
                
                
                rating = torch.tensor(
                    item['rating'] / 10.0, 
                    dtype=torch.float32, 
                    device=self.device
                ).view(1, 1)
                
                
                predicted_rating = self.prediction_head(embedding)
                
                
                loss = self.criterion(predicted_rating, rating)
                total_loss += loss.item()
                
                
                loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group['params']], 
                max_norm=1.0
            )
            
            self.optimizer.step()
            return total_loss / len(batch)
            
        except Exception as e:
            self.logger.error(f"Error in batch training: {str(e)}")
            return 0.0

    def _validate(self, val_data: List[Dict], batch_size: int) -> float:
        total_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                batch_loss = 0
                
                for item in batch:
                    embedding = self._compute_embedding(item)
                    rating = torch.tensor(
                        item['rating'] / 10.0, 
                        dtype=torch.float32, 
                        device=self.device
                    ).view(1, 1)
                    
                    predicted_rating = self.prediction_head(embedding)
                    loss = self.criterion(predicted_rating, rating)
                    batch_loss += loss.item()
                
                total_loss += batch_loss / len(batch)
        
        return total_loss / (len(val_data) / batch_size)

    def _save_model_state(self):
        state = {
            'content_encoder': self.content_encoder.state_dict(),
            'genre_encoder': self.genre_encoder.state_dict(),
            'temporal_encoder': self.temporal_encoder.state_dict(),
            'metadata_encoder': self.metadata_encoder.state_dict(),
            'graph_conv': self.graph_conv.state_dict(),
            'graph_attention': self.graph_attention.state_dict(),
            'graph_sage': self.graph_sage.state_dict(),
            'prediction_head': self.prediction_head.state_dict()
        }
        torch.save(state, 'recommendation_model.pt')

    def _load_model_state(self):
        if os.path.exists('recommendation_model.pt'):
            state = torch.load('recommendation_model.pt')
            self.content_encoder.load_state_dict(state['content_encoder'])
            self.genre_encoder.load_state_dict(state['genre_encoder'])
            self.temporal_encoder.load_state_dict(state['temporal_encoder'])
            self.metadata_encoder.load_state_dict(state['metadata_encoder'])
            self.graph_conv.load_state_dict(state['graph_conv'])
            self.graph_attention.load_state_dict(state['graph_attention'])
            self.graph_sage.load_state_dict(state['graph_sage'])
            self.prediction_head.load_state_dict(state['prediction_head'])

    def _update_genre_preferences(self, genres: List[str], rating: float) -> None:
        try:
            with self.preference_lock:
                cursor = self.db.conn.cursor()
                
                for genre in genres:
                    if not genre:  
                        continue
                        
                    cursor.execute('''
                        INSERT INTO genre_preferences (genre, rating_sum, rating_count)
                        VALUES (?, ?, 1)
                        ON CONFLICT(genre) DO UPDATE SET
                            rating_sum = rating_sum + ?,
                            rating_count = rating_count + 1
                    ''', (genre, float(rating), float(rating)))
                        
                self.db.conn.commit()
                print(f"Updated preferences for genres: {genres}")
                
        except Exception as e:
            self.logger.error(f"Error updating genre preferences: {str(e)}")
            print(f"Error in genre preference update: {e}")
            
    def _get_candidates(self, media_type: str) -> List[int]:
        cursor = self.db.conn.cursor()
        try:
            cursor.execute('SELECT COUNT(DISTINCT media_id) FROM user_feedback')
            feedback_count = cursor.fetchone()[0]
            
            if feedback_count <= 20:
                exploration_rate = 0.25
            elif feedback_count <= 50:
                exploration_rate = 0.13
            elif feedback_count <= 75:
                exploration_rate = 0.05
            else:
                exploration_rate = 0.02
                
            if random.random() < exploration_rate:
                cursor.execute('''
                    WITH FeedbackGenres AS (
                        SELECT DISTINCT json_each.value as genre
                        FROM media_items m
                        JOIN user_feedback f ON m.id = f.media_id
                        CROSS JOIN json_each(m.genres)
                    )
                    SELECT id 
                    FROM media_items m
                    WHERE type = ?
                    AND id NOT IN (SELECT media_id FROM user_feedback)
                    AND EXISTS (
                        SELECT 1 
                        FROM json_each(m.genres) 
                        WHERE json_each.value NOT IN (SELECT genre FROM FeedbackGenres)
                    )
                    ORDER BY RANDOM()
                    LIMIT 100
                ''', (media_type,))
            else:
                cursor.execute('''
                    SELECT id 
                    FROM media_items 
                    WHERE type = ?
                    AND id NOT IN (SELECT media_id FROM user_feedback)
                    ORDER BY 
                        COALESCE(vote_average * vote_count + popularity, 0) DESC,
                        RANDOM()
                    LIMIT 100
                ''', (media_type,))
            
            candidates = [row[0] for row in cursor.fetchall()]
            print(f"Found {len(candidates)} candidates (exploration_rate: {exploration_rate:.2f})")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error getting candidates: {str(e)}")
            print(f"Database error in _get_candidates: {str(e)}")
            return []
                                    
    def _get_content_based_scores(self, candidates):
        scores = {}
        cursor = self.db.conn.cursor()
        for item_id in candidates:
            cursor.execute('SELECT * FROM media_items WHERE id = ?', (item_id,))
            columns = [description[0] for description in cursor.description]
            item_data = dict(zip(columns, cursor.fetchone()))
            item_embedding = self._compute_embedding(item_data)
            
            similarity_scores = []
            cursor.execute('''
                SELECT m.* FROM media_items m
                JOIN user_feedback f ON m.id = f.media_id
                WHERE f.rating >= 7
            ''')
            rated_items = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            for rated_item in rated_items:
                rated_embedding = self._compute_embedding(rated_item)
                similarity = F.cosine_similarity(
                    item_embedding.unsqueeze(0),
                    rated_embedding.unsqueeze(0)
                ).item()
                similarity_scores.append(similarity)
            
            scores[item_id] = max(similarity_scores) if similarity_scores else 0
        
        return scores

    def _get_graph_based_scores(self, candidates):
        scores = {}
        for item_id in candidates:
            node_id = f"media_{item_id}"
            subgraph = self._extract_subgraph(node_id)
            edge_index, edge_type, node_features = self._prepare_graph_data(subgraph)
            
            with torch.no_grad():
                graph_embedding = self.graph_sage(node_features, edge_index)[0]
            
            similarity_scores = []
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT m.* FROM media_items m
                JOIN user_feedback f ON m.id = f.media_id
                WHERE f.rating >= 7
            ''')
            columns = [description[0] for description in cursor.description]
            rated_items = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            for rated_item in rated_items:
                rated_node_id = f"media_{rated_item['id']}"
                rated_subgraph = self._extract_subgraph(rated_node_id)
                rated_edge_index, rated_edge_type, rated_node_features = self._prepare_graph_data(rated_subgraph)
                
                with torch.no_grad():
                    rated_graph_embedding = self.graph_sage(rated_node_features, rated_edge_index)[0]
                
                similarity = F.cosine_similarity(
                    graph_embedding.unsqueeze(0),
                    rated_graph_embedding.unsqueeze(0)
                ).item()
                similarity_scores.append(similarity)
            
            scores[item_id] = max(similarity_scores) if similarity_scores else 0
        
        return scores

    def _get_collaborative_scores(self, candidates):
        if self.similarity_matrix is None:
            self._update_similarity_matrix()
        
        scores = {}
        cursor = self.db.conn.cursor()
        
        for item_id in candidates:
            cursor.execute('''
                SELECT m.*, f.rating FROM media_items m
                JOIN user_feedback f ON m.id = f.media_id
            ''')
            columns = [description[0] for description in cursor.description]
            rated_items = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            item_scores = []
            for rated_item in rated_items:
                similarity = self.similarity_matrix[item_id, rated_item['id']].item()
                rating = rated_item['rating']
                item_scores.append(similarity * rating)
            
            scores[item_id] = sum(item_scores) / len(item_scores) if item_scores else 0
        
        return scores

    def get_next_recommendation(self, media_type: str) -> Optional[Dict]:
        try:
            print(f"\nGetting recommendation for media_type: {media_type}")
            
            candidates = self._get_candidates(media_type)
            print(f"Found {len(candidates)} candidates")
            
            if not candidates:
                print("No candidates found!")
                return None

            cursor = self.db.conn.cursor()
            best_item = None
            best_score = float('-inf')

            for item_id in candidates:
                try:
                    cursor.execute('''
                        SELECT * FROM media_items 
                        WHERE id = ? AND type = ?
                    ''', (item_id, media_type))
                    
                    columns = [description[0] for description in cursor.description]
                    item_data = dict(zip(columns, cursor.fetchone()))
                    
                    if not item_data:
                        continue

                    score = self._calculate_item_score(item_data)
                    
                    if score > best_score:
                        best_score = score
                        best_item = item_data
                        print(f"New best item found: {item_data.get('title')} (score: {score:.2f})")

                except Exception as e:
                    print(f"Error processing item {item_id}: {str(e)}")
                    continue

            if best_item:
                print(f"Selected item: {best_item.get('title')} (ID: {best_item.get('id')})")
                return best_item

            print("No suitable recommendation found")
            return None

        except Exception as e:
            self.logger.error(f"Error getting recommendation: {str(e)}")
            print(f"Error in get_next_recommendation: {str(e)}")
            return None
        
    def _calculate_item_score(self, item_data: Dict) -> float:
        try:
            popularity = float(item_data.get('popularity', 0))
            vote_average = float(item_data.get('vote_average', 0))
            vote_count = float(item_data.get('vote_count', 0))
            
            norm_popularity = min(popularity / 100.0, 1.0)
            norm_vote_avg = vote_average / 10.0
            norm_votes = min(vote_count / 1000.0, 1.0)
            
            genre_bonus = self._calculate_genre_match_bonus(item_data.get('genres', '[]'))
            
            score = (
                0.35 * norm_popularity + 
                0.35 * norm_vote_avg + 
                0.20 * norm_votes +
                0.10 * genre_bonus
            ) * 100
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating item score: {str(e)}")
            return 0.0

    def _calculate_genre_match_bonus(self, genres_json: str) -> float:
        try:
            if not genres_json:
                return 0.0
                
            genres = json.loads(genres_json) if isinstance(genres_json, str) else genres_json
            if not genres:
                return 0.0
                
            genre_names = [
                genre.get('name') if isinstance(genre, dict) else str(genre)
                for genre in genres
            ]
            
            cursor = self.db.conn.cursor()
            total_bonus = 0.0
            
            for genre in genre_names:
                cursor.execute('''
                    SELECT rating_sum / CAST(rating_count AS FLOAT)
                    FROM genre_preferences
                    WHERE genre = ?
                ''', (genre,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    total_bonus += float(result[0]) / 10.0
                    
            return min(total_bonus / len(genre_names), 1.0) if genre_names else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating genre bonus: {str(e)}")
            return 0.0
        
    def train_with_feedback(self, media_id: int, rating: float, content_data: Dict) -> None:
            try:
                print(f"\nProcessing feedback for item {media_id} with rating {rating}")

                
                self.db.save_feedback(media_id, rating)
                print("Feedback saved to database")

                
                with torch.no_grad():
                    embedding_tensor = self._compute_embedding(content_data)
                    embedding_tensor = embedding_tensor.cpu().detach()
                    embedding_np = embedding_tensor.numpy().ravel()
                    
                self.db.save_embedding(media_id, embedding_np.tobytes())
                print("Embedding saved to cache")

                
                self._update_knowledge_graph(media_id, rating, content_data)
                print(f"Updated knowledge graph for {content_data.get('title', 'Unknown')}")

                
                try:
                    
                    if hasattr(self, 'similarity_thread') and self.similarity_thread.isRunning():
                        self.similarity_thread.quit()
                        self.similarity_thread.wait()

                    
                    self.similarity_thread = QThread()
                    targeted_items = self._get_targeted_candidates(content_data)
                    self.similarity_worker = SimilarityWorker(self, content_data, targeted_items)

                    
                    self.similarity_worker.moveToThread(self.similarity_thread)

                    
                    self.similarity_thread.started.connect(self.similarity_worker.process_batch)
                    self.similarity_worker.finished.connect(self.similarity_thread.quit)
                    self.similarity_worker.finished.connect(self.similarity_worker.deleteLater)
                    
                    self.similarity_worker.progress.connect(lambda msg: print(msg))
                    self.similarity_worker.error.connect(lambda msg: print(f"Error: {msg}"))

                    
                    print(f"Computing similarities for {len(targeted_items)} targeted candidates...")
                    self.similarity_thread.start()

                except Exception as e:
                    print(f"Error starting similarity computation: {e}")

                
                genres = self._process_json_field(content_data.get('genres', '[]'))
                self._update_genre_preferences(genres, rating)
                print("Genre preferences updated")

            except Exception as e:
                self.logger.error(f"Error training with feedback: {str(e)}")
                print(f"Error in feedback processing: {str(e)}")
            
    def _get_temporal_scores(self, candidates):
        scores = {}
        current_year = datetime.now().year
        
        cursor = self.db.conn.cursor()
        for item_id in candidates:
            cursor.execute('SELECT year FROM media_items WHERE id = ?', (item_id,))
            year = cursor.fetchone()[0]
            
            if year:
                age = current_year - int(year)
                temporal_score = 1.0 / (1.0 + 0.1 * age)
                scores[item_id] = temporal_score
            else:
                scores[item_id] = 0.5
        
        return scores

    def _get_method_weights(self) -> Dict[str, float]:
        try:
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT AVG(rating) as avg_rating,
                       COUNT(*) as rating_count
                FROM user_feedback
            ''')
            metrics = cursor.fetchone()
            
            if not metrics or metrics[1] == 0:
                return {
                    'content': 0.4,
                    'graph': 0.3,
                    'collaborative': 0.2,
                    'temporal': 0.1
                }
            
            avg_rating = metrics[0]
            rating_count = metrics[1]
            
            content_weight = 0.4 * (avg_rating / 10)
            graph_weight = 0.3 * min(1.0, rating_count / 100)
            collaborative_weight = 0.2 * min(1.0, rating_count / 50)
            temporal_weight = 0.1
            
            total = content_weight + graph_weight + collaborative_weight + temporal_weight
            
            return {
                'content': content_weight / total,
                'graph': graph_weight / total,
                'collaborative': collaborative_weight / total,
                'temporal': temporal_weight / total
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating method weights: {str(e)}")
            return {'content': 0.4, 'graph': 0.3, 'collaborative': 0.2, 'temporal': 0.1}

    def _update_similarity_matrix(self) -> None:
        try:
            current_time = datetime.now()
            if (self.last_matrix_update is None or
                (current_time - self.last_matrix_update).total_seconds() > 3600):
                
                cursor = self.db.conn.cursor()
                cursor.execute('SELECT id FROM media_items')
                item_ids = [row[0] for row in cursor.fetchall()]
                
                matrix_size = len(item_ids)
                self.similarity_matrix = torch.zeros((matrix_size, matrix_size), device=self.device)
                
                batch_size = 50
                for i in range(0, matrix_size, batch_size):
                    batch_ids = item_ids[i:i + batch_size]
                    embeddings_i = []
                    
                    for item_id in batch_ids:
                        cursor.execute('SELECT * FROM media_items WHERE id = ?', (item_id,))
                        columns = [description[0] for description in cursor.description]
                        item_data = dict(zip(columns, cursor.fetchone()))
                        embedding = self._compute_embedding(item_data)
                        
                        embedding = embedding.squeeze(0)  
                        embeddings_i.append(embedding)
                    
                    if embeddings_i:
                        embeddings_i = torch.stack(embeddings_i)  
                        
                        for j in range(0, matrix_size, batch_size):
                            batch_ids_j = item_ids[j:j + batch_size]
                            embeddings_j = []
                            
                            for item_id in batch_ids_j:
                                cursor.execute('SELECT * FROM media_items WHERE id = ?', (item_id,))
                                columns = [description[0] for description in cursor.description]
                                item_data = dict(zip(columns, cursor.fetchone()))
                                embedding = self._compute_embedding(item_data)
                                embedding = embedding.squeeze(0)  
                                embeddings_j.append(embedding)
                            
                            if embeddings_j:
                                embeddings_j = torch.stack(embeddings_j)  
                                
                                
                                similarities = F.cosine_similarity(
                                    embeddings_i.unsqueeze(1),  
                                    embeddings_j.unsqueeze(0),  
                                    dim=2  
                                )
                                
                                
                                i_end = min(i + batch_size, matrix_size)
                                j_end = min(j + batch_size, matrix_size)
                                self.similarity_matrix[i:i_end, j:j_end] = similarities
                
                self.last_matrix_update = current_time
                
        except Exception as e:
            self.logger.error(f"Error updating similarity matrix: {str(e)}")
            
    def _compute_embedding(self, media_item: Dict) -> torch.Tensor:
        try:
            cache_key = media_item['id']
            with self.embeddings_lock:
                cached_embedding = self.db.load_embedding(cache_key)
                if cached_embedding is not None:
                    return torch.from_numpy(np.frombuffer(cached_embedding, dtype=np.float32)).view(1, -1)

            
            text_features = self._encode_text_features(media_item)  
            numerical_features = self._encode_numerical_features(media_item)  
            categorical_features = self._encode_categorical_features(media_item)  
            temporal_features = self._encode_temporal_features(media_item)  
            metadata_features = self._encode_metadata_features(media_item)  
            graph_features = self._compute_graph_features(media_item)  

            
            print(f"text_features shape: {text_features.shape}")
            print(f"numerical_features shape: {numerical_features.shape}")
            print(f"categorical_features shape: {categorical_features.shape}")
            print(f"temporal_features shape: {temporal_features.shape}")
            print(f"metadata_features shape: {metadata_features.shape}")
            print(f"graph_features shape: {graph_features.shape}")

            
            combined_features = torch.cat([
                text_features,
                numerical_features,
                categorical_features,
                temporal_features,
                metadata_features,
                graph_features
            ], dim=1)

            
            print(f"combined_features shape: {combined_features.shape}")

            with torch.no_grad():
                embedding = self.content_encoder(combined_features)

            with self.embeddings_lock:
                self.db.save_embedding(cache_key, embedding.cpu().numpy().tobytes())

            return embedding

        except Exception as e:
            self.logger.error(f"Error computing embedding: {str(e)}")
            return torch.zeros((1, 384), device=self.device)

    def _extract_subgraph(self, node_id, depth=2):
        subgraph = {
            'nodes': {},
            'edges': defaultdict(list),
            'node_features': {},
            'edge_types': set()
        }
        
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_node, current_depth = queue.pop(0)
            
            if current_node in visited or current_depth > depth:
                continue
            
            visited.add(current_node)
            subgraph['nodes'][current_node] = self.knowledge_graph['nodes'][current_node]
            
            for neighbor, edge_type in self.knowledge_graph['edges'][current_node]:
                subgraph['edges'][current_node].append((neighbor, edge_type))
                subgraph['edge_types'].add(edge_type)
                
                if neighbor not in visited and current_depth < depth:
                    queue.append((neighbor, current_depth + 1))
        
        return subgraph

    def _prepare_graph_data(self, subgraph):
        try:
            node_mapping = {node: idx for idx, node in enumerate(subgraph['nodes'].keys())}
            reverse_mapping = {idx: node for node, idx in node_mapping.items()}
            edge_type_list = list(self.knowledge_graph['edge_types'])
            
            
            edge_lists = []
            edge_types = []
            for source in subgraph['edges']:
                for target, edge_type in subgraph['edges'][source]:
                    if target in node_mapping:
                        edge_lists.append([node_mapping[source], node_mapping[target]])
                        edge_types.append(edge_type_list.index(edge_type))
            
            
            if not edge_lists:
                for node in node_mapping:
                    edge_lists.append([node_mapping[node], node_mapping[node]])
                    edge_types.append(0)
            
            
            edge_index = torch.tensor(edge_lists, dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            
            
            node_features = []
            for idx in range(len(node_mapping)):
                node = reverse_mapping[idx]
                node_data = subgraph['nodes'][node]
                features = self._get_node_features(node_data)
                node_features.append(features)
            
            
            node_features = torch.cat(node_features, dim=0)
            
            
            return (
                edge_index.to(self.device).t().contiguous(),
                edge_type.to(self.device),
                node_features.to(self.device)
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing graph data: {str(e)}")
            
            return (
                torch.zeros((2, 1), dtype=torch.long, device=self.device),
                torch.zeros(1, dtype=torch.long, device=self.device),
                torch.zeros((1, 384), device=self.device)
            )

    def _initialize_knowledge_graph(self) -> Dict:
        try:
            cursor = self.db.conn.cursor()
            cursor.execute('SELECT * FROM media_items')
            columns = [description[0] for description in cursor.description]
            items = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            graph = {
                'nodes': {},
                'edges': defaultdict(list),
                'node_features': {},
                'edge_types': set()
            }
            
            for item in items:
                node_id = f"media_{item['id']}"
                graph['nodes'][node_id] = {
                    'type': 'media',
                    'data': item
                }
                
                genres = self._process_json_field(item.get('genres', '[]'))
                for genre in genres:
                    genre_id = f"genre_{genre}"
                    if genre_id not in graph['nodes']:
                        graph['nodes'][genre_id] = {
                            'type': 'genre',
                            'data': {'name': genre}
                        }
                    graph['edges'][node_id].append((genre_id, 'has_genre'))
                    graph['edge_types'].add('has_genre')
                
                keywords = self._process_json_field(item.get('keywords', '[]'))
                for keyword in keywords:
                    keyword_id = f"keyword_{keyword}"
                    if keyword_id not in graph['nodes']:
                        graph['nodes'][keyword_id] = {
                            'type': 'keyword',
                            'data': {'name': keyword}
                        }
                    graph['edges'][node_id].append((keyword_id, 'has_keyword'))
                    graph['edge_types'].add('has_keyword')
                
                text_fields = [
                    item.get('title', ''),
                    item.get('summary', ''),
                    item.get('overview', ''),
                    item.get('tagline', '')
                ]
                entities = self._extract_entities(' '.join(text_fields))
                for entity in entities:
                    entity_id = f"entity_{entity}"
                    if entity_id not in graph['nodes']:
                        graph['nodes'][entity_id] = {
                            'type': 'entity',
                            'data': {'name': entity}
                        }
                    graph['edges'][node_id].append((entity_id, 'mentions_entity'))
                    graph['edge_types'].add('mentions_entity')
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Error initializing knowledge graph: {str(e)}")
            return {'nodes': {}, 'edges': defaultdict(list), 'node_features': {}, 'edge_types': set()}

    def reset_model(self):
        try:
            cursor = self.db.conn.cursor()
            
            with self.preference_lock:
                cursor.execute('DELETE FROM user_feedback')
                cursor.execute('DELETE FROM genre_preferences')
                cursor.execute('DELETE FROM embedding_cache')
                self.db.conn.commit()
            
            with self.embeddings_lock:
                self.embedding_cache.clear()
                
            with self.graph_lock:
                self.knowledge_graph = self._initialize_knowledge_graph()
                
            self.similarity_matrix = None
            self.last_matrix_update = None
            
            return "Recommendation system has been reset successfully"
            
        except Exception as e:
            self.logger.error(f"Error resetting model: {str(e)}")
            return f"Error resetting system: {str(e)}"

    def cleanup_similarity_worker(self):
        try:
            if hasattr(self, 'similarity_worker'):
                self.similarity_worker.stop()
            if hasattr(self, 'similarity_thread') and self.similarity_thread.isRunning():
                self.similarity_thread.quit()
                self.similarity_thread.wait()
        except Exception as e:
            self.logger.error(f"Error cleaning up similarity worker: {e}")

    def cleanup(self):
        try:
            self.cleanup_similarity_worker()
            
            with self.embeddings_lock:
                self.embedding_cache.clear()
            
            
            models_to_cleanup = [
                'text_encoder', 'content_encoder', 'genre_encoder',
                'temporal_encoder', 'metadata_encoder', 'graph_conv',
                'graph_attention', 'graph_sage'
            ]
            
            for model_name in models_to_cleanup:
                if hasattr(self, model_name):
                    delattr(self, model_name)
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def _safe_text_conversion(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, dict)):
            try:
                if isinstance(value, list):
                    return " ".join(
                        str(item.get('name', item.get('content', item))) 
                        if isinstance(item, dict) 
                        else str(item) 
                        for item in value
                    )
                else:
                    return str(value.get('name', value.get('content', str(value))))
            except Exception:
                return str(value)
        return str(value)

    def _encode_text_features(self, item: Dict) -> torch.Tensor:
        try:
            text_fields = []
            

            fields_to_process = [
                item.get('title', ''),
                item.get('summary', ''),
                item.get('overview', ''),
                item.get('tagline', ''),
                self._process_json_field(item.get('keywords', '[]')),
                self._process_json_field(item.get('reviews', '[]'))
            ]
            
            text_fields = [self._safe_text_conversion(field) for field in fields_to_process]
            
            combined_text = ' [SEP] '.join(filter(None, text_fields))
            
            if not combined_text.strip():
                combined_text = "unknown content"
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    combined_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                outputs = self.text_encoder(**inputs)
                text_embedding = outputs.last_hidden_state.mean(dim=1)
                
            return text_embedding
            
        except Exception as e:
            self.logger.error(f"Error in text encoding: {str(e)}")
            return torch.zeros((1, 384), device=self.device)
            
    def _encode_numerical_features(self, item: Dict) -> torch.Tensor:
        features = [
            float(item.get('popularity', 0)),
            float(item.get('vote_average', 0)) / 10.0,
            float(item.get('vote_count', 0)) / 10000.0,
            float(item.get('year', 2000)) / 2025.0,
        ]
        
        numerical_features = torch.tensor(
            features, 
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)  
        
        return numerical_features

    def _encode_categorical_features(self, item: Dict) -> torch.Tensor:
        
        genres = self._process_json_field(item.get('genres', '[]'))
        genre_encoding = self._one_hot_genres(genres)  
        
        
        status_encoding = torch.tensor(
            self._one_hot_encode(
                item.get('status', ''),
                ['Released', 'In Production', 'Ended', 'Cancelled']  
            ),
            device=self.device,
            dtype=torch.float32
        )  
        
        type_encoding = torch.tensor(
            self._one_hot_encode(
                item.get('type', ''),
                ['movie', 'show']  
            ),
            device=self.device,
            dtype=torch.float32
        )  
        
        language_encoding = torch.tensor(
            self._one_hot_encode(
                item.get('language', ''),
                ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh']  
            ),
            device=self.device,
            dtype=torch.float32
        )  
        
        
        
        combined_categorical = torch.cat([
            genre_encoding,
            status_encoding,
            type_encoding,
            language_encoding
        ])
        
        
        combined_categorical = combined_categorical.unsqueeze(0)
        
        
        categorical_embedding = self.genre_encoder(combined_categorical)  
        
        return categorical_embedding

    def _encode_temporal_features(self, item: Dict) -> torch.Tensor:
        temporal_features = torch.zeros(32, device=self.device)
        
        
        year = int(item.get('year', datetime.now().year))
        current_year = datetime.now().year
        years_old = (current_year - year) / 100.0
        temporal_features[0] = years_old
        
        
        release_date = item.get('release_date', '')
        if release_date:
            try:
                release_month = datetime.strptime(release_date, '%Y-%m-%d').month
                temporal_features[release_month] = 1.0
            except ValueError:
                pass
        
        
        temporal_features = temporal_features.unsqueeze(0)  
        encoded_temporal = self.temporal_encoder(temporal_features)  
        
        return encoded_temporal

    def _encode_metadata_features(self, item: Dict) -> torch.Tensor:
        metadata_fields = [
            self._process_json_field(item.get('production_companies', '[]')),
            self._process_json_field(item.get('credits', '[]')),
            self._process_json_field(item.get('keywords', '[]'))
        ]
        
        metadata_text = ' '.join(filter(None, metadata_fields))
        
        with torch.no_grad():
            inputs = self.tokenizer(
                metadata_text,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.device)
            
            outputs = self.text_encoder(**inputs)
            metadata_embedding = outputs.last_hidden_state.mean(dim=1)  
        
        
        encoded_metadata = self.metadata_encoder(metadata_embedding)  
        
        return encoded_metadata

    def _process_json_field(self, field: Union[str, List, Dict, None]) -> str:
        try:
            if field is None:
                return ""
            if isinstance(field, (list, dict)):
                return json.dumps(field)  
            if isinstance(field, str):
                try:
                    parsed = json.loads(field)
                    if isinstance(parsed, (list, dict)):
                        return json.dumps(parsed)
                    return str(parsed)
                except json.JSONDecodeError:
                    return field
            return str(field)
        except Exception as e:
            self.logger.error(f"Error processing field: {field}, {str(e)}")
            return ""


    def _one_hot_genres(self, genres_str: Union[str, List]) -> torch.Tensor:
        standard_genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
            'Mystery', 'Romance', 'Science Fiction', 'Thriller',
            'War', 'Western', 'Biography', 'History', 'Music',
            'Sport', 'Reality-TV', 'News'
        ]
        
        if isinstance(genres_str, str):
            genres = set(g.strip() for g in genres_str.split(','))
        elif isinstance(genres_str, list):
            genres = set(str(g).strip() for g in genres_str)
        else:
            genres = set()
        
        encoding = torch.zeros(len(standard_genres), device=self.device)
        for i, genre in enumerate(standard_genres):
            if genre in genres:
                encoding[i] = 1.0
                
        return encoding

    def _one_hot_encode(self, value: str, possible_values: List[str]) -> List[float]:
        encoding = [0.0] * len(possible_values)
        try:
            if value in possible_values:
                encoding[possible_values.index(value)] = 1.0
        except Exception:
            pass
        return encoding

    def _update_knowledge_graph(self, media_id: int, rating: float, content_data: Dict) -> None:
        try:
            with self.graph_lock:
                node_id = f"media_{media_id}"
                print(f"Updating knowledge graph for {content_data.get('title', 'Unknown')}")
                
                if node_id not in self.knowledge_graph['nodes']:
                    self.knowledge_graph['nodes'][node_id] = {
                        'type': 'media',
                        'data': {
                            k: v for k, v in content_data.items() 
                            if not isinstance(v, torch.Tensor)  
                        },
                        'rating': float(rating)
                    }
                else:
                    self.knowledge_graph['nodes'][node_id]['rating'] = float(rating)
                    
                print(f"Updated node: {node_id}")
                
                try:
                    similar_items = self._find_similar_items(content_data)
                    print(f"Found {len(similar_items)} similar items")
                    
                    for similar_id in similar_items:
                        similar_node_id = f"media_{similar_id}"
                        edge_tuple = (similar_node_id, 'similar_to')
                        
                        self.knowledge_graph['edges'].setdefault(similar_node_id, [])
                        self.knowledge_graph['edges'].setdefault(node_id, [])
                            
                        if edge_tuple not in self.knowledge_graph['edges'][node_id]:
                            self.knowledge_graph['edges'][node_id].append(edge_tuple)
                            self.knowledge_graph['edge_types'].add('similar_to')
                            print(f"Added edge: {node_id} -> {similar_node_id}")
                            
                except Exception as e:
                    print(f"Error processing similar items: {e}")
                
        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {str(e)}")
            print(f"Error in knowledge graph update: {e}")
            print(f"Traceback: {traceback.format_exc()}")            
        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {str(e)}")
            print(f"Error in knowledge graph update: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
    def _find_similar_items(self, content_data: Dict) -> List[int]:
        try:
            cursor = self.db.conn.cursor()
            item_id = content_data['id']
            
            similar_items = self._get_cached_similarities(item_id)
            if similar_items:
                return similar_items

            targeted_items = self._get_targeted_candidates(content_data)
            print(f"Computing similarities for {len(targeted_items)} targeted candidates...")

            content_data = self._sanitize_json_fields(content_data)
            
            similarities = self._compute_similarities_for_batch(content_data, targeted_items)
            
            self._store_similarities(item_id, similarities)
            
            similar_ids = [item_id for item_id, _, _ in similarities[:20]]
            return similar_ids

        except Exception as e:
            self.logger.error(f"Error finding similar items: {str(e)}")
            return []
    
    def _sanitize_json_fields(self, data: Dict) -> Dict:
        sanitized = data.copy()
        
        json_fields = ['genres', 'keywords', 'credits', 'videos', 'reviews', 
                      'network', 'production_companies']
        
        for field in json_fields:
            value = sanitized.get(field)
            if value is not None:
                if isinstance(value, str):
                    try:
                        json.loads(value)
                    except json.JSONDecodeError:
                        sanitized[field] = json.dumps([value])
                else:
                    sanitized[field] = json.dumps(value)
            else:
                sanitized[field] = '[]'
                
        return sanitized
            
    def _get_cached_similarities(self, item_id: int) -> Optional[List[int]]:
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT item2_id, similarity 
            FROM similarity_matrix 
            WHERE item1_id = ? 
            AND last_updated > datetime('now', '-24 hours')
            ORDER BY similarity DESC
            LIMIT 20
        ''', (item_id,))
        results = cursor.fetchall()
        
        if results:
            return [r[0] for r in results]
        return None

    def _get_targeted_candidates(self, content_data: Dict) -> List[Dict]:
        cursor = self.db.conn.cursor()
        
        
        genres = json.loads(content_data.get('genres', '[]')) if isinstance(content_data.get('genres'), str) else content_data.get('genres', [])
        year = content_data.get('year')
        
        
        query = '''
            SELECT DISTINCT m.* FROM media_items m
            WHERE m.id != ? 
            AND m.type = ?
        '''
        params = [content_data['id'], content_data['type']]
        
        
        if genres:
            genre_conditions = []
            for genre in genres:
                genre_conditions.append("m.genres LIKE ?")
                params.append(f"%{genre}%")
            query += f" AND ({' OR '.join(genre_conditions)})"
        
        if year:
            query += " AND m.year BETWEEN ? AND ?"
            params.extend([year - 10, year + 10])
        
        
        query += '''
            UNION
            SELECT DISTINCT m.* FROM media_items m
            JOIN user_feedback f ON m.id = f.media_id
            WHERE m.id != ?
            AND m.type = ?
            AND f.rating >= 7
        '''
        params.extend([content_data['id'], content_data['type']])
        
        
        query += " LIMIT 200"
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _extract_set_from_json(self, json_str: str, key: str = 'name') -> Set[str]:
        try:
            if not json_str:
                return set()
            if isinstance(json_str, str):
                data = json.loads(json_str)
                if isinstance(data, list):
                    return {str(item.get(key, '')) if isinstance(item, dict) else str(item) for item in data}
                return {str(data)}
            return set()
        except json.JSONDecodeError:
            return {str(json_str)}
        except Exception:
            return set()

    def _compute_similarities_for_batch(self, reference_item: Dict, candidates: List[Dict]) -> List[Tuple[int, str, float]]:
        similarities = []
        batch_size = 5
        
        try:
            reference_embedding = self._compute_embedding(reference_item)
            reference_embedding = reference_embedding.view(-1)
            reference_embedding = F.normalize(reference_embedding, p=2, dim=0)
            
            ref_genres = self._safe_extract_list(reference_item.get('genres'))
            ref_keywords = self._safe_extract_list(reference_item.get('keywords'))
            ref_networks = self._safe_extract_list(reference_item.get('network'))
            ref_production = self._safe_extract_list(reference_item.get('production_companies'))
            
            ref_year = self._safe_float(reference_item.get('year', 0))
            ref_runtime = self._safe_float(reference_item.get('runtime', 0))
            ref_language = str(reference_item.get('language', ''))
            ref_status = str(reference_item.get('status', ''))
            
            ref_cast_ids = set()
            ref_director_ids = set()
            ref_writer_ids = set()
            
            try:
                credits = self._safe_extract_list(reference_item.get('credits'))
                if isinstance(credits, dict):
                    cast = credits.get('cast', [])
                    crew = credits.get('crew', [])
                    
                    if isinstance(cast, list):
                        ref_cast_ids = {str(c.get('id', '')) for c in cast[:10] if isinstance(c, dict)}
                    
                    if isinstance(crew, list):
                        ref_director_ids = {str(c.get('id', '')) for c in crew if isinstance(c, dict) and c.get('job') == 'Director'}
                        ref_writer_ids = {str(c.get('id', '')) for c in crew if isinstance(c, dict) and c.get('job') in ['Writer', 'Screenplay']}
            except Exception as e:
                self.logger.error(f"Error processing reference credits: {e}")
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                batch_similarities = []
                
                for item in batch:
                    try:
                        with torch.no_grad():
                            other_embedding = self._compute_embedding(item)
                            other_embedding = other_embedding.view(-1)
                            other_embedding = F.normalize(other_embedding, p=2, dim=0)
                            embedding_similarity = torch.dot(reference_embedding, other_embedding).item()
                        
                        item_genres = self._safe_extract_list(item.get('genres'))
                        item_keywords = self._safe_extract_list(item.get('keywords'))
                        item_networks = self._safe_extract_list(item.get('network'))
                        item_production = self._safe_extract_list(item.get('production_companies'))
                        
                        item_cast_ids = set()
                        item_director_ids = set()
                        item_writer_ids = set()
                        
                        try:
                            credits = self._safe_extract_list(item.get('credits'))
                            if isinstance(credits, dict):
                                cast = credits.get('cast', [])
                                crew = credits.get('crew', [])
                                
                                if isinstance(cast, list):
                                    item_cast_ids = {str(c.get('id', '')) for c in cast[:10] if isinstance(c, dict)}
                                
                                if isinstance(crew, list):
                                    item_director_ids = {str(c.get('id', '')) for c in crew if isinstance(c, dict) and c.get('job') == 'Director'}
                                    item_writer_ids = {str(c.get('id', '')) for c in crew if isinstance(c, dict) and c.get('job') in ['Writer', 'Screenplay']}
                        except Exception as e:
                            self.logger.error(f"Error processing candidate credits: {e}")
                        
                        genre_similarity = self._safe_set_similarity(ref_genres, item_genres)
                        year_similarity = max(0, 1 - abs(ref_year - self._safe_float(item.get('year', 0))) / 100)
                        runtime_similarity = max(0, 1 - abs(ref_runtime - self._safe_float(item.get('runtime', 0))) / 120)
                        keyword_similarity = self._safe_set_similarity(ref_keywords, item_keywords)
                        cast_similarity = self._safe_set_similarity(ref_cast_ids, item_cast_ids)
                        director_similarity = self._safe_set_similarity(ref_director_ids, item_director_ids)
                        writer_similarity = self._safe_set_similarity(ref_writer_ids, item_writer_ids)
                        production_similarity = self._safe_set_similarity(ref_production, item_production)
                        network_similarity = self._safe_set_similarity(ref_networks, item_networks)
                        
                        language_match = 1.0 if ref_language == str(item.get('language', '')) else 0.0
                        status_match = 1.0 if ref_status == str(item.get('status', '')) else 0.0
                        
                        popularity_score = min(1.0, self._safe_float(item.get('popularity', 0)) / 100.0)
                        vote_average = self._safe_float(item.get('vote_average', 0)) / 10.0
                        vote_count = min(1.0, self._safe_float(item.get('vote_count', 0)) / 1000.0)
                        
                        final_similarity = (
                            0.25 * embedding_similarity +
                            0.15 * genre_similarity +
                            0.10 * keyword_similarity +
                            0.10 * cast_similarity +
                            0.08 * director_similarity +
                            0.05 * writer_similarity +
                            0.05 * production_similarity +
                            0.05 * year_similarity +
                            0.03 * runtime_similarity +
                            0.03 * network_similarity +
                            0.03 * language_match +
                            0.02 * status_match +
                            0.04 * vote_average +
                            0.02 * vote_count +
                            0.02 * popularity_score
                        )
                        
                        batch_similarities.append((item['id'], item['title'], final_similarity))
                        
                    except Exception as e:
                        self.logger.error(f"Error processing comparison item {item.get('id')}: {e}")
                        continue
                    finally:
                        if 'other_embedding' in locals():
                            del other_embedding
                        torch.cuda.empty_cache()
                
                similarities.extend(batch_similarities)
                
            return sorted(similarities, key=lambda x: x[2], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error computing similarities: {str(e)}")
            return []
        finally:
            if 'reference_embedding' in locals():
                del reference_embedding
            torch.cuda.empty_cache()

    def _safe_extract_list(self, value: Any) -> Set[str]:
        try:
            if not value:
                return set()
                
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return {value}
            
            if isinstance(value, dict):
                return {str(value.get('name', value.get('id', '')))}
                
            if isinstance(value, list):
                result = set()
                for item in value:
                    if isinstance(item, dict):
                        name = item.get('name', item.get('id', ''))
                        if name:
                            result.add(str(name))
                    else:
                        if item:
                            result.add(str(item))
                return result
                
            return {str(value)}
            
        except Exception as e:
            self.logger.error(f"Error in safe_extract_list: {e}")
            return set()

    def _safe_float(self, value: Any) -> float:
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str) and value.strip():
                return float(value)
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _safe_set_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        try:
            if not set1 or not set2:
                return 0.0
            union_size = len(set1 | set2)
            if union_size == 0:
                return 0.0
            return len(set1 & set2) / union_size
        except Exception:
            return 0.0
    
    def _extract_names_from_json(self, json_str: str) -> Set[str]:
        try:
            if not json_str:
                return set()
            
            data = self._parse_json_field(json_str)
            if not data:
                return set()
            
            if isinstance(data, list):
                return {
                    str(item.get('name', item)) if isinstance(item, dict) else str(item)
                    for item in data
                    if item is not None
                }
            
            return {str(data)}
        except Exception as e:
            self.logger.error(f"Error extracting names: {str(e)}")
            return set()

    def _parse_json_field(self, field: Union[str, Dict, List]) -> Union[Dict, List]:
        if isinstance(field, (dict, list)):
            return field
        
        if isinstance(field, str):
            try:
                return json.loads(field)
            except json.JSONDecodeError:
                return [field] if field else []
        
        return []

    def _extract_ids_from_credits(self, credits: List, limit: int = 10) -> Set[str]:
        try:
            if not isinstance(credits, list):
                return set()
            
            return {
                str(person.get('id', '')) 
                for person in credits[:limit] 
                if person is not None and isinstance(person, dict)
            }
        except Exception as e:
            self.logger.error(f"Error extracting credit IDs: {str(e)}")
            return set()
            
    def _get_node_features(self, node_data):
        try:
            if node_data['type'] == 'media':
                cursor = self.db.conn.cursor()
                cursor.execute('SELECT * FROM media_items WHERE id = ?', (node_data['data']['id'],))
                columns = [description[0] for description in cursor.description]
                item_data = dict(zip(columns, cursor.fetchone()))
                
                
                text = f"{item_data.get('title', '')} {item_data.get('summary', '')}"
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128,
                    padding=True
                )
                
                
                with torch.no_grad():
                    outputs = self.text_encoder(**{k: v for k, v in inputs.items()})
                    features = outputs.last_hidden_state.mean(dim=1)
            else:
                
                node_hash = hash(str(node_data))
                torch.manual_seed(node_hash)
                features = torch.randn(1, 384)
            
            return features.cpu()  
            
        except Exception as e:
            self.logger.error(f"Error getting node features: {str(e)}")
            return torch.zeros((1, 384))

    def _store_similarities(self, item_id: int, similarities: List[Tuple], is_final: bool = True):
        if not similarities:
            self.logger.debug(f"No similarities to store for item {item_id}")
            return
                
        try:
            with self.db.write_lock:
                cursor = self.db.conn.cursor()
                
                if is_final:
                    cursor.execute(
                        'DELETE FROM similarity_matrix WHERE item1_id = ?', 
                        (item_id,)
                    )

                valid_similarities = []
                for sim in similarities:
                    try:
                        if len(sim) >= 3 and all(s is not None for s in (sim[0], sim[2])):
                            valid_similarities.append((item_id, sim[0], sim[2]))
                    except (IndexError, TypeError) as e:
                        self.logger.error(f"Invalid similarity data: {sim}, Error: {str(e)}")
                        continue
                
                if valid_similarities:
                    cursor.executemany('''
                        INSERT OR REPLACE INTO similarity_matrix 
                        (item1_id, item2_id, similarity, last_updated)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ''', valid_similarities)
                
                self.db.conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing similarities: {str(e)}")
        finally:
            if cursor:
                cursor.close()
                    
    def _compute_graph_features(self, item):
        try:
            with self.graph_lock:
                node_id = f"media_{item['id']}"
                subgraph = self._extract_subgraph(node_id)
                edge_index, edge_type, node_features = self._prepare_graph_data(subgraph)

                
                with torch.no_grad():
                    features = []
                    
                    
                    try:
                        graph_embedding = self.graph_conv(node_features, edge_index)
                        graph_embedding = graph_embedding.mean(dim=0, keepdim=True).cpu()
                        features.append(graph_embedding)
                    except Exception as e:
                        self.logger.error(f"GCN error: {str(e)}")
                        features.append(torch.zeros((1, 341)))

                    
                    try:
                        attention_embedding = self.graph_attention(node_features, edge_index)
                        attention_embedding = attention_embedding.mean(dim=0, keepdim=True).cpu()
                        features.append(attention_embedding)
                    except Exception as e:
                        self.logger.error(f"GAT error: {str(e)}")
                        features.append(torch.zeros((1, 341)))

                    
                    try:
                        sage_embedding = self.graph_sage(node_features, edge_index)
                        sage_embedding = sage_embedding.mean(dim=0, keepdim=True).cpu()
                        features.append(sage_embedding)
                    except Exception as e:
                        self.logger.error(f"GraphSAGE error: {str(e)}")
                        features.append(torch.zeros((1, 342)))

                    
                    combined_embedding = torch.cat(features, dim=1)
                    
                    
                    return combined_embedding.to(self.device)

        except Exception as e:
            self.logger.error(f"Error computing graph features: {str(e)}")
            return torch.zeros((1, 384), device=self.device)
        finally:
            
            if 'edge_index' in locals(): del edge_index
            if 'edge_type' in locals(): del edge_type
            if 'node_features' in locals(): del node_features
            if 'graph_embedding' in locals(): del graph_embedding
            if 'attention_embedding' in locals(): del attention_embedding
            if 'sage_embedding' in locals(): del sage_embedding
            torch.cuda.empty_cache()

class PosterDownloader(QObject):
    def __init__(self):
        super().__init__()
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self._handle_network_response)
        self._request_data = {}
        self._request_counter = 0
        self._lock = threading.Lock()
        self._active_requests = set()
        self._cache = {}
        self._max_retries = 3
        self._retry_delay = 1000
        self.logger = logging.getLogger(__name__)

    def download_poster(self, url: str, poster_label: QLabel) -> None:
        try:
            if url in self._cache:
                pixmap = self._cache[url]
                self._apply_pixmap(pixmap, poster_label)
                return

            if url in self._active_requests:
                return

            request = QNetworkRequest(QUrl(url))
            request.setMaximumRedirectsAllowed(5)
            request.setAttribute(QNetworkRequest.RedirectPolicyAttribute, 
                              QNetworkRequest.NoLessSafeRedirectPolicy)

            try:
                request.setAttribute(QNetworkRequest.Attribute.User, poster_label)
                using_attribute = True
            except (AttributeError, TypeError):
                with self._lock:
                    self._request_counter += 1
                    request_id = self._request_counter
                    self._request_data[request_id] = {
                        'label': poster_label,
                        'retries': 0,
                        'url': url
                    }
                    request.setHeader(QNetworkRequest.CustomHeader, str(request_id).encode())
                using_attribute = False

            self._active_requests.add(url)
            self.network_manager.get(request)

        except Exception as e:
            self.logger.error(f"Error starting poster download: {str(e)}")
            self._set_default_poster(poster_label)
            self._active_requests.discard(url)

    def _handle_network_response(self, reply: QNetworkReply) -> None:
        url = reply.url().toString()
        self._active_requests.discard(url)

        try:
            try:
                poster_label = reply.request().attribute(QNetworkRequest.Attribute.User)
                request_data = None
            except (AttributeError, TypeError):
                request_id = int(reply.request().rawHeader(b"CustomHeader"))
                with self._lock:
                    request_data = self._request_data.pop(request_id, None)
                    poster_label = request_data['label'] if request_data else None

            if not poster_label:
                self.logger.error("Could not find poster label for network reply")
                return

            if reply.error() == QNetworkReply.NetworkError.NoError:
                data = reply.readAll()
                pixmap = QPixmap()
                if pixmap.loadFromData(data):
                    self._cache[url] = pixmap
                    self._apply_pixmap(pixmap, poster_label)
                else:
                    self._handle_download_error(reply, request_data)
            else:
                self._handle_download_error(reply, request_data)

        except Exception as e:
            self.logger.error(f"Error handling network response: {str(e)}")
            if 'poster_label' in locals() and poster_label:
                self._set_default_poster(poster_label)
        finally:
            reply.deleteLater()

    def _apply_pixmap(self, pixmap: QPixmap, label: QLabel) -> None:
        try:
            scaled_pixmap = pixmap.scaled(
                300, 450,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception as e:
            self.logger.error(f"Error applying pixmap: {str(e)}")
            self._set_default_poster(label)

    def _handle_download_error(self, reply: QNetworkReply, request_data: Optional[Dict]) -> None:
        if not request_data:
            self.logger.error("No request data available for retry")
            return

        if request_data['retries'] < self._max_retries:
            request_data['retries'] += 1
            QTimer.singleShot(
                self._retry_delay * request_data['retries'],
                lambda: self.download_poster(request_data['url'], request_data['label'])
            )
        else:
            self.logger.error(f"Max retries exceeded: {reply.errorString()}")
            self._set_default_poster(request_data['label'])

    def _set_default_poster(self, label: QLabel) -> None:
        try:
            pixmap = QPixmap(300, 450)
            pixmap.fill(QColor(200, 200, 200))

            with QPainter(pixmap) as painter:
                painter.setPen(QColor(100, 100, 100))
                painter.setFont(QFont('Arial', 14))
                painter.drawText(
                    pixmap.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "No\nPoster\nAvailable"
                )

            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
            self.logger.error(f"Error setting default poster: {str(e)}")

    def cleanup(self):
        with self._lock:
            self._request_data.clear()
            self._request_counter = 0
            self._active_requests.clear()
            self._cache.clear()

        if hasattr(self, 'network_manager'):
            self.network_manager.finished.disconnect(self._handle_network_response)
            self.network_manager.deleteLater()
            self.network_manager = None

class BackgroundTrainer(QThread):
    training_status = Signal(str)
    
    def __init__(self, recommender, db):
        super().__init__()
        self.recommender = recommender
        self.db = db
        self.logger = logging.getLogger(__name__)
        self._stop_flag = threading.Event()
        self._is_training = threading.Event()
        self.min_wait_time = 300
        self.max_wait_time = 3600
        self.batch_size = 32
        self.preference_lock = threading.RLock()
        self._batch_lock = threading.Lock()
        self._embedding_lock = threading.Lock()
        self._training_queue = deque(maxlen=1000)
        self._queue_lock = threading.Lock()
            
    def run(self):
        while not self._stop_flag.is_set():
            try:
                if not self._is_training.is_set():
                    self._is_training.set()
                    try:
                        self._run_training_cycle()
                    finally:
                        self._is_training.clear()
                
                if self._stop_flag.is_set():
                    break
                
                sleep_time = self._calculate_sleep_time()
                self.training_status.emit(f"Sleeping for {sleep_time/60:.1f} minutes until next training cycle")
                
                for _ in range(int(sleep_time * 10)):
                    if self._stop_flag.is_set():
                        return
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in background training: {str(e)}")
                if self._stop_flag.is_set():
                    break
                time.sleep(1)

    def queue_item_for_training(self, item_data, priority=False):
        with self._queue_lock:
            if priority:
                self._training_queue.appendleft(item_data)
            else:
                self._training_queue.append(item_data)
                
    def _run_training_cycle(self):
        self.training_status.emit("Starting background training cycle")
        
        with self._batch_lock:
            self._update_missing_embeddings()
        
        with self._batch_lock:
            self._refresh_old_embeddings()
            
        with self._embedding_lock:
            self._update_similarity_matrix()
            
        with self._batch_lock:
            self._train_on_feedback()
            
        with self.preference_lock:
            self._update_genre_preferences_from_feedback()
            
        self._process_training_queue()
        
    def _process_training_queue(self):
        while True:
            with self._queue_lock:
                if not self._training_queue:
                    break
                item_data = self._training_queue.popleft()
            
            try:
                with self._batch_lock:
                    self._train_single_item(item_data)
            except Exception as e:
                self.logger.error(f"Error training queued item: {str(e)}")

    def _update_genre_preferences_from_feedback(self):
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT m.genres, f.rating 
                FROM media_items m 
                JOIN user_feedback f ON m.id = f.media_id
                WHERE m.genres IS NOT NULL
            """)
            
            for genres, rating in cursor.fetchall():
                if genres and rating:
                    try:
                        if isinstance(genres, str):
                            genres_list = json.loads(genres)
                        else:
                            genres_list = genres
                        
                        if isinstance(genres_list, list):
                            with self.preference_lock:
                                for genre in genres_list:
                                    genre_name = genre.get('name') if isinstance(genre, dict) else str(genre)
                                    if genre_name:
                                        cursor.execute('''
                                            INSERT INTO genre_preferences (genre, rating_sum, rating_count)
                                            VALUES (?, ?, 1)
                                            ON CONFLICT(genre) DO UPDATE SET
                                                rating_sum = rating_sum + ?,
                                                rating_count = rating_count + 1
                                        ''', (genre_name, float(rating), float(rating)))
                                self.db.conn.commit()
                                
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing genre preferences: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error updating genre preferences: {str(e)}")

    def _train_single_item(self, item_data):
        try:
            with self._embedding_lock:
                embedding = None
                try:
                    embedding = self.recommender._compute_embedding(item_data)
                    self.db.save_embedding(item_data['id'], embedding.cpu().numpy().tobytes())
                finally:
                    if embedding is not None:
                        del embedding
                        torch.cuda.empty_cache()
                    
            with self.preference_lock:
                self._update_item_preferences(item_data)
                    
        except Exception as e:
            self.logger.error(f"Error training item {item_data.get('id')}: {str(e)}")

    def _update_missing_embeddings(self):
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT m.* FROM media_items m 
            LEFT JOIN embedding_cache e ON m.id = e.media_id
            WHERE e.media_id IS NULL
            LIMIT 100
        """)
        
        columns = [description[0] for description in cursor.description]
        items = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if items:
            self.training_status.emit(f"Generating embeddings for {len(items)} new items")
            
            for item in items:
                if self._stop_flag.is_set():
                    break
                    
                try:
                    with self._embedding_lock:
                        embedding = self.recommender._compute_embedding(item)
                        self.db.save_embedding(item['id'], embedding.cpu().numpy().tobytes())
                except Exception as e:
                    self.logger.error(f"Error generating embedding for item {item['id']}: {str(e)}")
                    continue

    def _refresh_old_embeddings(self):
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT m.* FROM media_items m 
            JOIN embedding_cache e ON m.id = e.media_id
            WHERE e.last_updated < datetime('now', '-30 days')
            LIMIT 50
        """)
        
        columns = [description[0] for description in cursor.description]
        items = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if items:
            self.training_status.emit(f"Refreshing embeddings for {len(items)} items")
            
            for item in items:
                if self._stop_flag.is_set():
                    break
                    
                try:
                    with self._embedding_lock:
                        embedding = self.recommender._compute_embedding(item)
                        self.db.save_embedding(item['id'], embedding.cpu().numpy().tobytes())
                except Exception as e:
                    self.logger.error(f"Error refreshing embedding for item {item['id']}: {str(e)}")
                    continue

    def _calculate_sleep_time(self):
        cursor = self.db.conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM media_items")
            total_items = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM embedding_cache 
                WHERE last_updated > datetime('now', '-24 hours')
            """)
            recent_updates = cursor.fetchone()[0]
            
            update_ratio = recent_updates / max(total_items, 1)
            
            if update_ratio > 0.2: 
                sleep_time = self.min_wait_time
            elif update_ratio > 0.1:  
                sleep_time = (self.min_wait_time + self.max_wait_time) / 2
            else: 
                sleep_time = self.max_wait_time
                
            return max(self.min_wait_time, min(sleep_time, self.max_wait_time))
            
        except Exception as e:
            self.logger.error(f"Error calculating sleep time: {e}")
            return self.max_wait_time 
        finally:
            if cursor:
                cursor.close()

    def _update_similarity_matrix(self):
        self.training_status.emit("Updating similarity matrix")
        self.recommender._update_similarity_matrix()

    def _train_on_feedback(self):
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT m.*, f.rating FROM media_items m
            JOIN user_feedback f ON m.id = f.media_id
            ORDER BY f.timestamp DESC
            LIMIT 1000
        """)
        
        columns = [description[0] for description in cursor.description]
        feedback_data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if feedback_data:
            batches = [feedback_data[i:i + self.batch_size] for i in range(0, len(feedback_data), self.batch_size)]
            
            for batch in batches:
                if self._stop_flag.is_set():
                    break
                    
                try:
                    loss = self.recommender._train_batch(batch)
                    self.training_status.emit(f"Batch training loss: {loss:.4f}")
                except Exception as e:
                    self.logger.error(f"Error training batch: {str(e)}")
                    continue

    def _update_item_preferences(self, item_data):
        if not isinstance(item_data, dict):
            return
            
        try:
            genres = self._process_json_field(item_data.get('genres', '[]'))
            if genres:
                self._update_genre_preferences(genres, item_data.get('vote_average', 5.0))
        except Exception as e:
            self.logger.error(f"Error updating preferences: {str(e)}")

    def stop(self):
        self._stop_flag.set()
        
        start_time = time.time()
        while self.isRunning() and (time.time() - start_time) < 5:
            time.sleep(0.1)
        
        if self.isRunning():
            self.terminate()
            self.wait(1000)

class SimilarityComputationThread(QThread):
    progress = Signal(str)
    finished = Signal()

    def __init__(self, engine, reference_item, candidates):
        super().__init__()
        self.engine = engine
        self.reference_item = reference_item
        self.candidates = candidates
        self._stop_flag = False

    def run(self):
        reference_embedding = None
        reference_cpu = None
        try:
            similarities = []
            batch_size = 5

            with torch.no_grad():
                reference_embedding = self.engine._compute_embedding(self.reference_item)
                reference_embedding = reference_embedding.view(-1)
                reference_embedding = F.normalize(reference_embedding, p=2, dim=0)
                reference_cpu = reference_embedding.cpu()

            for i in range(0, len(self.candidates), batch_size):
                if self._stop_flag:
                    break

                batch = self.candidates[i:i + batch_size]
                batch_similarities = []

                for item in batch:
                    if self._stop_flag:
                        break

                    other_embedding = None
                    other_cpu = None
                    try:
                        with torch.no_grad():
                            other_embedding = self.engine._compute_embedding(item)
                            other_embedding = other_embedding.view(-1)
                            other_embedding = F.normalize(other_embedding, p=2, dim=0)
                            other_cpu = other_embedding.cpu()

                            similarity = torch.dot(reference_cpu, other_cpu).item()
                            batch_similarities.append((item['id'], item['title'], similarity))

                    except Exception as e:
                        self.engine.logger.error(f"Error processing item {item['id']}: {e}")
                        continue
                    finally:
                        if other_embedding is not None:
                            del other_embedding
                        if other_cpu is not None:
                            del other_cpu
                        torch.cuda.empty_cache()

                similarities.extend(batch_similarities)
                progress = f"Computing similarities: {min(i + batch_size, len(self.candidates))}/{len(self.candidates)}"
                self.progress.emit(progress)

            if not self._stop_flag:
                self.engine._store_similarities(self.reference_item['id'], similarities)

        except Exception as e:
            self.engine.logger.error(f"Error in similarity computation: {str(e)}")
        finally:
            if reference_embedding is not None:
                del reference_embedding
            if reference_cpu is not None:
                del reference_cpu
            torch.cuda.empty_cache()
            self.finished.emit()
            
    def stop(self):
        self._stop_flag = True

class MediaScanner:
    def __init__(self):
        self.plex: Optional[PlexServer] = None
        self.tvdb_client: Optional[TVDBClient] = None
        self.tmdb_client: Optional[TMDBClient] = None
        self.tmdb_api_key: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._processed_items: Set[str] = set()

        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def initialize_session(self) -> None:
        if self.session is None or self.session.closed:
            timeout = ClientTimeout(total=30, connect=10, sock_read=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=10,
                    force_close=False,
                    enable_cleanup_closed=True,
                    keepalive_timeout=30
                )
            )

    async def close(self) -> None:
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                await asyncio.sleep(0.25)
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
        finally:
            self.session = None

    def configure(self, plex_url: str, plex_token: str, tvdb_key: str, tmdb_key: str) -> None:
        if not all([plex_url, plex_token, tvdb_key, tmdb_key]):
            raise ValueError("All configuration parameters must be provided")

        try:
            
            self.tmdb_client = TMDBClient(tmdb_key)
            self.tvdb_client = TVDBClient(tvdb_key)

            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            
            loop.run_until_complete(self.tmdb_client._initialize_session())
            loop.run_until_complete(self.tmdb_client.fetch_configuration())

            
            loop.run_until_complete(self.tvdb_client.ensure_token())

            
            self.plex = PlexServer(plex_url, plex_token, timeout=10)
            self.plex.library.sections()  

        except Exception as e:
            self.logger.error(f"Configuration error: {str(e)}")
            self.plex = None
            self.tvdb_client = None
            self.tmdb_client = None
            raise ConnectionError(f"Failed to configure media scanner: {str(e)}")
        
        self.logger.info("Media scanner configured successfully")

    async def scan_library(self, library, progress_callback) -> None:
        if library.type not in ['movie', 'show']:
            self.logger.warning(f"Skipping unsupported library type: {library.type}")
            progress_callback.emit(f"Skipping unsupported library type: {library.type}")
            return

        self.logger.info(f"Starting scan of library: {library.title} ({library.type})")
        progress_callback.emit(f"Starting scan of library: {library.title}")

        db = Database.get_instance()
        cursor = db.conn.cursor()

        items_to_insert = []
        total_items = 0
        matched_items = 0
        error_items = []

        batch_size = 50
        all_items = list(library.all())

        for i in range(0, len(all_items), batch_size):
            batch = all_items[i:i + batch_size]

            for item in batch:
                total_items += 1
                try:
                    if self._get_item_key(item) in self._processed_items:
                        self.logger.debug(f"Skipping already processed item: {item.title}")
                        continue

                    progress_callback.emit(
                        f"Processing {item.title} ({total_items}/{len(all_items)})"
                    )
                    self.logger.info(f"Processing item: {item.title}")

                    if await self._item_exists(cursor, item):
                        self.logger.info(f"Item already exists: {item.title}")
                        continue

                    metadata = await self._fetch_metadata(item, library.type)

                    if metadata:
                        matched_items += 1
                        processed_item = await self._process_item(item, metadata, library.type)

                        if processed_item:
                            items_to_insert.append(processed_item)
                            self._processed_items.add(self._get_item_key(item))

                            if len(items_to_insert) >= 100:
                                await self._commit_items(cursor, items_to_insert)
                                items_to_insert = []
                    else:
                        error_items.append(item.title)

                except Exception as e:
                    error_items.append(item.title)
                    self.logger.error(f"Error processing {item.title}: {str(e)}")
                    progress_callback.emit(f"Error processing {item.title}: {str(e)}")
                    continue

            if items_to_insert:
                await self._commit_items(cursor, items_to_insert)
                items_to_insert = []

            progress_percentage = (total_items / len(all_items)) * 100
            progress_callback.emit(
                f"Progress: {progress_percentage:.1f}% - "
                f"Matched: {matched_items}/{total_items}"
            )

        match_rate = (matched_items / total_items * 100) if total_items > 0 else 0
        summary_msg = (
            f"Completed scanning {library.title}. "
            f"Processed {total_items} items, matched {matched_items} ({match_rate:.1f}%). "
            f"Errors: {len(error_items)}"
        )

        if error_items:
            error_summary = f"\nItems with errors: {', '.join(error_items[:10])}"
            if len(error_items) > 10:
                error_summary += f" and {len(error_items) - 10} more..."
            summary_msg += error_summary

        self.logger.info(summary_msg)
        progress_callback.emit(summary_msg)

    async def _fetch_metadata(self, item: Any, media_type: str) -> Optional[Dict[str, Any]]:
        metadata = None

        
        try:
            metadata = await self.fetch_tmdb_metadata(item.title, getattr(item, 'year', None), media_type)
            if metadata:
                self.logger.info(f"TMDB match found for {item.title}")
        except Exception as e:
            self.logger.error(f"TMDB fetch failed for {item.title}: {str(e)}")

        
        if not metadata and media_type == 'show':
            try:
                tvdb_id = self._extract_tvdb_id(item) or await self._guess_tvdb_id(item.title)
                if not tvdb_id:
                    self.logger.warning(f"Cannot fetch TVDB metadata for {item.title}, no ID found")
                else:
                    metadata = await self.tvdb_client.fetch_series_data(tvdb_id)
                    if metadata:
                        self.logger.info(f"TVDB match found for {item.title}")
            except Exception as e:
                self.logger.error(f"TVDB fetch failed for {item.title}: {str(e)}")

        return metadata

    async def fetch_tvdb_metadata(self, item: Any) -> Optional[Dict[str, Any]]:
        if not self.tvdb_client:
            raise RuntimeError("TVDB client not configured")

        tvdb_id = self._extract_tvdb_id(item)
        if not tvdb_id:
            self.logger.warning(f"Falling back to title-based search for {item.title}")
            tvdb_id = await self._fallback_to_title_search(item.title)


        retries = 3
        for attempt in range(retries):
            try:
                await self.initialize_session()
                await self.tvdb_client.ensure_token()
                series_data = await self.tvdb_client.fetch_series_data(tvdb_id)

                if not series_data or not series_data.get('name'):
                    return None

                return {
                    'id': tvdb_id,
                    'title': series_data['name'],
                    'original_title': series_data.get('originalName'),
                    'year': self._extract_year(series_data.get('firstAired')),
                    'genres': self._sanitize_list(series_data.get('genres', [])),
                    'summary': series_data.get('overview', '').strip(),
                    'status': series_data.get('status'),
                    'rating': self._sanitize_rating(series_data.get('rating')),
                    'runtime': self._sanitize_runtime(series_data.get('runtime')),
                    'network': series_data.get('network'),
                    'content_rating': series_data.get('contentRating'),
                    'imdb_id': series_data.get('imdbId'),
                    'air_days': series_data.get('airsDays'),
                    'air_time': series_data.get('airsTime'),
                    'production_countries': series_data.get('originalCountry'),
                    'spoken_languages': series_data.get('originalLanguage'),
                    'awards': series_data.get('awards', []),
                    'characters': series_data.get('characters', []),
                    'season_count': series_data.get('seasonCount'),
                    'episode_count': series_data.get('episodeCount'),
                    'last_aired': series_data.get('lastAired'),
                    'score': series_data.get('score'),
                    'episodes': self._sanitize_episodes(series_data.get('episodes', [])),
                    'image_urls': self._extract_image_urls(series_data)
                }

            except aiohttp.ClientError as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Unexpected error fetching TVDB data: {str(e)}")
                raise

    async def fetch_tmdb_metadata(self, title: str, year: Optional[int], media_type: str) -> Optional[Dict[str, Any]]:
        if not self.tmdb_client:
            raise RuntimeError("TMDB client not configured")

        retries = 3
        for attempt in range(retries):
            try:
                await self.initialize_session()

                search_results = await self._tmdb_search(title, year, media_type)
                if not search_results:
                    self.logger.error(f"TMDB search returned no results for {title} ({year})")
                    return None

                item_id = search_results[0]['id']
                metadata = await self._tmdb_fetch_details(item_id, media_type)
                if not metadata:
                    return None

                return self._process_tmdb_metadata(metadata, media_type)

            except aiohttp.ClientError as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Unexpected error fetching TMDB data: {str(e)}")
                raise

    async def _tmdb_search(self, title: str, year: Optional[int], media_type: str) -> List[Dict]:
        search_type = 'tv' if media_type.lower() in ['show', 'tv'] else media_type

        params = {
            "query": title,
            "language": "en-US",
            "include_adult": "false"
        }
        if year:
            params["year" if search_type == "movie" else "first_air_date_year"] = str(year)

        headers = {
            "Authorization": f"Bearer {self.tmdb_client.access_token}",
            "accept": "application/json"
        }

        try:
            self.logger.debug(f"TMDB search params: {params}")
            async with self.session.get(
                f"{self.tmdb_client.base_url}/search/{search_type}",
                headers=headers,
                params=params
            ) as response:
                if response.status == 404:
                    self.logger.error(f"TMDB fetch failed for {title}: 404, message='Not Found', url='{response.url}'")
                    return []

                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    await asyncio.sleep(retry_after)
                    return await self._tmdb_search(title, year, media_type)

                response.raise_for_status()
                data = await response.json()
                return data.get('results', [])

        except Exception as e:
            self.logger.error(f"TMDB fetch failed for {title}: {str(e)}")
            raise

    async def _tmdb_fetch_details(self, item_id: int, media_type: str) -> Optional[Dict]:
        headers = {
            "Authorization": f"Bearer {self.tmdb_client.access_token}",
            "accept": "application/json"
        }

        params = {
            "append_to_response": (
                "credits,keywords,videos,reviews,similar,recommendations,"
                "watch/providers,external_ids,content_ratings,aggregate_credits,"
                "images,episode_groups,screened_theatrically,lists"
            ),
            "language": "en-US",
            "include_adult": "false"
        }

        try:
            async with self.session.get(
                f"{self.tmdb_client.base_url}/{media_type}/{str(item_id)}",  
                headers=headers,
                params=params
            ) as response:
                if response.status == 404:
                    self.logger.error(f"TMDB fetch failed for item_id {item_id}: 404, message='Not Found', url='{response.url}'")
                    return None

                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    await asyncio.sleep(retry_after)
                    return await self._tmdb_fetch_details(item_id, media_type)

                response.raise_for_status()
                data = await response.json()

                if media_type == 'tv' and data.get('number_of_seasons', 0) > 0:
                    seasons_data = []
                    for season_num in range(1, data['number_of_seasons'] + 1):
                        try:
                            season_params = {
                                "append_to_response": "credits,videos,images",
                                "language": "en-US",
                                "include_adult": "false"
                            }
                            async with self.session.get(
                                f"{self.tmdb_client.base_url}/tv/{str(item_id)}/season/{str(season_num)}",
                                headers=headers,
                                params=season_params
                            ) as season_response:
                                if season_response.status == 200:
                                    season_data = await season_response.json()
                                    seasons_data.append(season_data)
                        except Exception as e:
                            self.logger.error(f"Error fetching season {season_num}: {str(e)}")
                            continue

                    data['seasons_data'] = seasons_data

                return data

        except Exception as e:
            self.logger.error(f"Error fetching details for {item_id}: {str(e)}")
            raise
                    
    async def _item_exists(self, cursor: Any, item: Any) -> bool:
        cursor.execute(
            'SELECT id FROM media_items WHERE title = ? AND year = ?',
            (item.title, getattr(item, 'year', None))
        )
        return bool(cursor.fetchone())

    async def _process_item(self, item: Any, metadata: Dict[str, Any], media_type: str) -> Optional[tuple]:
        try:
            
            def safe_json_dumps(value):
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                return value

            
            def safe_str(value):
                if isinstance(value, dict):
                    return str(value.get('name', '')) or str(value.get('title', '')) or ''
                return str(value) if value is not None else ''

            
            return (
                safe_str(item.title),                                     
                media_type,                                              
                getattr(item, 'year', None),                            
                str(getattr(item, 'duration', '')),                     
                
                safe_str(metadata.get('overview')) or 
                safe_str(metadata.get('summary')) or 
                safe_str(getattr(item, 'summary', '')),                 
                safe_json_dumps(metadata.get('genres', [])),            
                safe_str(getattr(item, 'thumb', '')),                   
                safe_str(metadata.get('id')),                           
                safe_str(metadata.get('tmdb_id')),                      
                safe_str(metadata.get('original_title')),               
                safe_str(metadata.get('overview')),                     
                float(metadata.get('popularity', 0)),                   
                float(metadata.get('vote_average', 0)),                 
                int(metadata.get('vote_count', 0)),                     
                safe_str(metadata.get('status')),                       
                safe_str(metadata.get('tagline')),                      
                safe_str(metadata.get('backdrop_path')),                
                safe_str(metadata.get('release_date')) or 
                safe_str(metadata.get('first_air_date')),              
                safe_str(metadata.get('content_rating')),               
                safe_str(metadata.get('network')),                      
                safe_json_dumps(metadata.get('credits')),               
                safe_json_dumps(metadata.get('keywords')),              
                safe_json_dumps(metadata.get('videos')),                
                safe_str(metadata.get('language')),                     
                safe_json_dumps(metadata.get('production_companies')),  
                safe_json_dumps(metadata.get('reviews')),               
                safe_json_dumps(metadata.get('episodes')),              
                int(metadata.get('season_count', 0) or 0),             
                int(metadata.get('episode_count', 0) or 0),            
                safe_str(metadata.get('first_air_date')),              
                safe_str(metadata.get('last_air_date'))                
            )
        except Exception as e:
            self.logger.error(f"Error processing item {item.title}: {str(e)}")
            return None

    async def _commit_items(self, cursor: Any, items: List[tuple]) -> None:
        try:
            cursor.execute('BEGIN')
            cursor.executemany('''
                INSERT INTO media_items (
                    title, type, year, runtime, summary, genres, poster_url, tvdb_id, tmdb_id,
                    original_title, overview, popularity, vote_average, vote_count,
                    status, tagline, backdrop_path, release_date, content_rating,
                    network, credits, keywords, videos, language, production_companies,
                    reviews, episodes, season_count, episode_count, first_air_date, last_air_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', items)
            cursor.execute('COMMIT')
        except Exception as e:
            cursor.execute('ROLLBACK')
            self.logger.error(f"Database commit failed: {str(e)}")
            raise

    def _get_item_key(self, item: Any) -> str:
        return f"{item.title}_{getattr(item, 'year', '')}_{getattr(item, 'ratingKey', '')}"

    def _process_tmdb_metadata(self, data: Dict, media_type: str) -> Dict:
        metadata = {
            'tmdb_id': data.get('id'),
            'title': data.get('title' if media_type == 'movie' else 'name'),
            'original_title': data.get('original_title' if media_type == 'movie' else 'original_name'),
            'overview': data.get('overview'),
            'popularity': data.get('popularity'),
            'vote_average': data.get('vote_average'),
            'vote_count': data.get('vote_count'),
            'genres': [genre['name'] for genre in data.get('genres', [])],
            'keywords': [kw['name'] for kw in data.get('keywords', {}).get('results', [])],
            'production_companies': [
                {
                    'id': company['id'],
                    'name': company['name'],
                    'country': company.get('origin_country')
                }
                for company in data.get('production_companies', [])
            ],
            'production_countries': data.get('production_countries', []),
            'spoken_languages': data.get('spoken_languages', []),
            'credits': {
                'cast': [
                    {
                        'id': person['id'],
                        'name': person['name'],
                        'character': person.get('character'),
                        'order': person.get('order'),
                        'profile_path': person.get('profile_path')
                    }
                    for person in data.get('credits', {}).get('cast', [])
                ],
                'crew': [
                    {
                        'id': person['id'],
                        'name': person['name'],
                        'job': person.get('job'),
                        'department': person.get('department')
                    }
                    for person in data.get('credits', {}).get('crew', [])
                ]
            },
            'videos': [
                {
                    'key': video['key'],
                    'type': video['type'],
                    'site': video['site']
                }
                for video in data.get('videos', {}).get('results', [])
            ],
            'reviews': [
                {
                    'author': review['author'],
                    'content': review['content'],
                    'rating': review.get('author_details', {}).get('rating')
                }
                for review in data.get('reviews', {}).get('results', [])
            ],
            'similar': [
                {
                    'id': item['id'],
                    'title': item.get('title', item.get('name')),
                    'overview': item.get('overview')
                }
                for item in data.get('similar', {}).get('results', [])
            ],
            'recommendations': [
                {
                    'id': item['id'],
                    'title': item.get('title', item.get('name')),
                    'overview': item.get('overview')
                }
                for item in data.get('recommendations', {}).get('results', [])
            ],
            'watch_providers': data.get('watch/providers', {}).get('results', {}),
            'external_ids': data.get('external_ids', {}),
            'content_ratings': data.get('content_ratings', {}).get('results', []),
            'images': {
                'posters': [
                    {
                        'file_path': img['file_path'],
                        'width': img['width'],
                        'height': img['height'],
                        'vote_average': img.get('vote_average', 0)
                    }
                    for img in data.get('images', {}).get('posters', [])
                ],
                'backdrops': [
                    {
                        'file_path': img['file_path'],
                        'width': img['width'],
                        'height': img['height'],
                        'vote_average': img.get('vote_average', 0)
                    }
                    for img in data.get('images', {}).get('backdrops', [])
                ]
            }
        }

        if not metadata['title']:
            return None

        if media_type == 'tv':
            metadata.update({
                'created_by': data.get('created_by', []),
                'episode_run_time': data.get('episode_run_time', []),
                'first_air_date': data.get('first_air_date'),
                'last_air_date': data.get('last_air_date'),
                'networks': data.get('networks', []),
                'number_of_episodes': data.get('number_of_episodes'),
                'number_of_seasons': data.get('number_of_seasons'),
                'seasons': data.get('seasons_data', []),
                'status': data.get('status'),
                'type': data.get('type')
            })

        return metadata
    async def _guess_tvdb_id(self, title: str) -> Optional[str]:

        try:
            
            search_results = await self.tvdb_client.search(title)
            if search_results:
                self.logger.info(f"TVDB search found ID for '{title}': {search_results[0]['id']}")
                return search_results[0]['id']
            self.logger.warning(f"TVDB search returned no results for title: {title}")
        except Exception as e:
            self.logger.error(f"Error during TVDB title search for '{title}': {str(e)}")
        return None

    def _extract_tvdb_id(self, item) -> Optional[str]:
        try:
            guid = getattr(item, 'guid', '')
            if not guid:
                self.logger.warning(f"Missing GUID for item: {item.title}")
                return None

            tvdb_match = re.search(r'(?:thetvdb|tvdb)://(\d+)', guid)
            if tvdb_match:
                return tvdb_match.group(1)

            if guid.startswith('plex://'):
                self.logger.info(f"Plex GUID found for {item.title}, falling back to title search")
                return None

            self.logger.warning(f"No valid TVDB ID found in GUID: {guid} for item: {item.title}")
            return None

        except Exception as e:
            self.logger.error(f"Error extracting TVDB ID for {item.title}: {str(e)}")
            return None


    async def _fallback_to_title_search(self, title: str) -> Optional[str]:
        try:
            search_results = await self.tvdb_client.search(title)
            if search_results:
                self.logger.info(f"TVDB search found ID for '{title}': {search_results[0]['id']}")
                return search_results[0]['id']
            self.logger.warning(f"No results found for TVDB title search: {title}")
        except Exception as e:
            self.logger.error(f"Error during TVDB title search for '{title}': {str(e)}")
        return None


    def _sanitize_list(self, items: List[Any]) -> List[str]:
        if not isinstance(items, list):
            return []
        return list(set(str(item).strip() for item in items if item))

    def _sanitize_rating(self, rating: Any) -> Optional[float]:
        try:
            if rating is None:
                return None
            rating_float = float(rating)
            return rating_float if 0 <= rating_float <= 10 else None
        except (ValueError, TypeError):
            return None

    def _sanitize_runtime(self, runtime: Any) -> Optional[int]:
        try:
            if runtime is None:
                return None
            runtime_int = int(runtime)
            return runtime_int if runtime_int > 0 else None
        except (ValueError, TypeError):
            return None

    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        try:
            if len(date_str) >= 4 and date_str[:4].isdigit():
                year = int(date_str[:4])
                return year if 1900 <= year <= datetime.now().year + 1 else None
            return None
        except (ValueError, TypeError):
            return None

    def _sanitize_episodes(self, episodes: List[Dict]) -> List[Dict]:
        if not isinstance(episodes, list):
            return []

        sanitized = []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue

            sanitized_episode = {
                'season': episode.get('seasonNumber'),
                'episode': episode.get('episodeNumber'),
                'title': episode.get('name', '').strip(),
                'overview': episode.get('overview', '').strip(),
                'air_date': episode.get('firstAired')
            }

            if all(sanitized_episode.get(key) is not None for key in ['season', 'episode', 'title']):
                sanitized.append(sanitized_episode)

        return sanitized

    def _extract_image_urls(self, data: Dict) -> Dict[str, Optional[str]]:
        return {
            'poster': data.get('image'),
            'banner': data.get('banner'),
            'fanart': data.get('fanart')
        }

class ScanThread(QThread):
    progress = Signal(str)
    
    def __init__(self, scanner, library_names):
        super().__init__()
        self.scanner = scanner
        self.library_names = library_names
        self.loop = None
        self._is_running = True

    def run(self):
        try:
            
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            
            self.loop.run_until_complete(self._run_scan())
            
        except Exception as e:
            self.progress.emit(f"Error during scanning: {str(e)}")
        finally:
            
            pending = asyncio.all_tasks(self.loop)
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def stop(self):
        self._is_running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
                
    async def _run_scan(self):
            try:
                
                if self.scanner.tmdb_client:
                    await self.scanner.tmdb_client._initialize_session()
                if self.scanner.tvdb_client:
                    await self.scanner.tvdb_client.ensure_token()

                
                self.scanner.session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(
                        limit=10,
                        force_close=True,
                        enable_cleanup_closed=True
                    ),
                    timeout=aiohttp.ClientTimeout(total=30)
                )

                
                media_libraries = {}
                for library in self.scanner.plex.library.sections():
                    if library.title in self.library_names:
                        if library.type == 'movie':
                            media_libraries.setdefault('movie', []).append(library)
                        elif library.type == 'show':
                            media_libraries.setdefault('show', []).append(library)

                
                for media_type, libraries in media_libraries.items():
                    if not self._is_running:
                        break
                        
                    self.progress.emit(f"Processing {media_type} libraries...")
                    for library in libraries:
                        if not self._is_running:
                            break
                            
                        try:
                            await self.scanner.scan_library(library, self.progress)
                        except Exception as e:
                            self.progress.emit(f"Error processing library {library.title}: {str(e)}")
                            continue

            except Exception as e:
                self.progress.emit(f"Error during scanning: {str(e)}")
                
            finally:
                
                cleanup_tasks = []
                
                if hasattr(self.scanner, 'session') and self.scanner.session:
                    cleanup_tasks.append(self.scanner.session.close())
                    self.scanner.session = None
                    
                if self.scanner.tmdb_client:
                    cleanup_tasks.append(self.scanner.tmdb_client.close())
                    
                if self.scanner.tvdb_client:
                    cleanup_tasks.append(self.scanner.tvdb_client.close())
                
                
                if cleanup_tasks:
                    await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    def __del__(self):
        self.stop()
        if self.loop and not self.loop.is_closed():
            self.loop.close()

def setup_theme(app: QApplication):
    palette = QPalette()
    
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    
    app.setPalette(palette)
    
    app.setStyle('Fusion')
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
def main():
    app = QApplication([])
    
    setup_theme(app)
    
    try:
        window = MediaRecommenderApp()
        window.show()
        
        app.aboutToQuit.connect(lambda: cleanup_application(window))
        
        return app.exec()
        
    except Exception as e:
        QMessageBox.critical(
            None,
            "Application Error",
            f"An error occurred while starting the application:\n\n{str(e)}"
        )
        return 1

def cleanup_application(window):
    try:
        Database.get_instance().close()
        
        if hasattr(window, 'cleanup'):
            window.cleanup()
        
    except Exception as e:
        print(f"Error during application cleanup: {e}")

if __name__ == '__main__':
    sys.exit(main())
