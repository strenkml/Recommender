# Recommender

A sophisticated AI-powered recommendation system that integrates with Plex, TMDB, and TVDB to provide personalized movie and TV show recommendations. Built with PySide6 for the GUI and leveraging PyTorch for neural networks, it incorporates multiple recommendation strategies including content-based filtering, collaborative filtering, and knowledge graphs.

![image](https://github.com/user-attachments/assets/4dd8f567-0ac8-4e2b-83b7-b52b2566a8a9)

![image](https://github.com/user-attachments/assets/5b0fc52b-24b2-4a47-86a5-eef317ba0e16)

![image](https://github.com/user-attachments/assets/98be5b16-5870-420f-aace-2f1c8a19bef5)



- [Core Features](#core-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Technical Architecture](#technical-architecture)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Core Features

### Recommendation Engine
- Hybrid recommendation system combining:
  - Neural content encoders for media analysis
  - Graph-based recommendation using GCN, GAT, and GraphSAGE
  - Adaptive learning from user feedback
  - Temporal dynamics handling
  - Genre and metadata preference tracking

### Integration
- **Plex Integration**: 
  - Direct library scanning
  - Watchlist management
  - Collections management with auto-add capabilities
  - Real-time metadata synchronization
- **External APIs**:
  - [TMDB](https://www.themoviedb.org/documentation/api) integration for movie/show details
  - [TVDB](https://thetvdb.com/api-information) integration for extended TV show metadata

### Advanced Features
- **Neural Networks**:
  - Content encoding with transformers
  - Multi-modal feature fusion
  - Graph neural networks for relationship modeling
  - Temporal encoding for release patterns
  - Adaptive batch processing
- **Data Processing**:
  - Efficient SQLite database with WAL mode
  - Optimized embedding caching
  - Background training process
  - Automatic data cleanup
  - Real-time preference updates

### User Interface
- Rating system (1-10 scale)
- Skip functionality
- Content blocking
- History navigation
- Watchlist integration
- Collections management
- Detailed progress tracking
- Configurable auto-collection thresholds

## System Requirements

### Software Requirements
- Python 3.8 or higher
- Plex Media Server
- API Keys:
  - [TMDB API Key](https://www.themoviedb.org/documentation/api)
  - [TVDB API Key](https://thetvdb.com/api-information)
  - [Plex Token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/)

### Hardware Requirements
- CPU: Any modern multi-core processor
- RAM: Minimum 4GB (8GB+ recommended)
- Storage: 500MB for application + database
- GPU: Not required (CPU-optimized)

## Installation

### Using Pre-built Release
1. Download latest release
2. Run executable
3. Configure API settings
4. Select libraries to scan
5. Begin rating content

### Manual Installation
1. Clone repository:
```bash
git clone https://github.com/yourusername/Recommender.git
cd Recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run application:
```bash
python main.py
```

### Dependencies
Primary packages:
```
pyside6>=6.4.0
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0
sentence-transformers>=2.2.0
plexapi>=4.15.0
aiohttp>=3.8.0
nltk>=3.8.0
numpy>=1.24.0
```

## Configuration

### Initial Setup
1. Open Configuration tab
2. Enter required API keys:
   - Plex URL & Token
   - TVDB API Key
   - TMDB API Key
3. Click "Save Configuration"
4. Load and select libraries
5. Configure collection settings (optional)
6. Start library scan

### Collection Settings
- Set default collection name
- Configure auto-add threshold
- Customize collection behavior
- Enable/disable automatic organization

## Usage Guide

### Rating Content
- View recommendations in Movies/TV Shows tabs
- Rate items 1-10 using rating buttons
- Skip unwanted recommendations
- Block items to never see again
- Navigate history with Back button

### Collections Management
- Automatic collection based on ratings
- Manual collection additions
- Customizable collection naming
- Smart collection organization

### Watchlist Features
- Add items to Plex watchlist
- Bulk watchlist management
- Clear entire watchlist
- Synchronization with Plex

### Model Management
- Reset recommendation system
- Clear blocked items
- Regenerate embeddings
- View training status
- Monitor system performance

## Technical Architecture

### Components
1. **Neural Networks**:
   - Content Encoder
   - Genre Encoder
   - Temporal Encoder
   - Graph Neural Networks (GCN, GAT, GraphSAGE)
   - Multi-head Attention

2. **Database**:
   - SQLite with WAL mode
   - Optimized indices
   - Efficient caching
   - Background cleanup

3. **API Integration**:
   - Asynchronous clients
   - Rate limiting
   - Error handling
   - Auto-retry logic

4. **Processing Pipeline**:
   - Background training
   - Async metadata fetching
   - Batch processing
   - Memory management

### Database Schema
```sql
media_items:
  - id (PRIMARY KEY)
  - title, type, year
  - metadata fields
  - status tracking
  - recommendation data

genre_preferences:
  - genre (PRIMARY KEY)
  - rating statistics
  - preference tracking

user_feedback:
  - rating history
  - temporal data
  - user preferences

embedding_cache:
  - cached computations
  - update timestamps
  - optimization data

similarity_matrix:
  - item relationships
  - computed similarities
  - update tracking
```

## Performance Optimization

### Database
- WAL journaling
- Optimized indices
- Memory-mapped I/O
- Strategic caching
- Background cleanup

### Neural Processing
- Batch operations
- Memory efficiency
- Background training
- Cached embeddings
- Adaptive scheduling

### API Handling
- Connection pooling
- Request batching
- Rate limit awareness
- Error resilience
- Automatic retry

## Troubleshooting

### Common Issues
1. **Connection Problems**:
   - Verify Plex URL/token
   - Check API keys
   - Confirm network access
   - Review error logs

2. **Performance Issues**:
   - Monitor memory usage
   - Check database size
   - Review process timing
   - Optimize settings

3. **Recommendation Quality**:
   - Provide more ratings
   - Check library scan
   - Review blocked items
   - Reset if needed

### Error Recovery
- Automatic cleanup
- Graceful degradation
- Self-healing processes
- Data consistency checks

## License
GPL v3 - See [LICENSE](LICENSE) file for details
