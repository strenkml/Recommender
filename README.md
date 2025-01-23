
# Recommender

Get personalized movie and TV show recommendations based on your Plex libraries and ratings. The more you rate, the better it learns your preferences!



![image](https://github.com/user-attachments/assets/4dd8f567-0ac8-4e2b-83b7-b52b2566a8a9)

![image](https://github.com/user-attachments/assets/5b0fc52b-24b2-4a47-86a5-eef317ba0e16)

![image](https://github.com/user-attachments/assets/98be5b16-5870-420f-aace-2f1c8a19bef5)



## What Does It Do?

- Shows you movies and TV shows you might like based on your ratings
- Learns your preferences over time
- Adds recommendations to your Plex watchlist
- Creates and manages Plex collections
- Works with your existing Plex libraries

The app runs completely independently of your Plex server. It only interacts with Plex to:
- Read your library contents during initial scan
- Add items to your watchlist when requested
- Create/modify collections when requested
- Fetch poster images for display

All recommendation logic, learning, and data storage happens locally in the app. Your Plex server is never modified without explicit actions from you.

## Features

### Smart Recommendations
- Rate movies and shows from 1-10
- Skip things you're not interested in
- Block items you never want to see
- Go back to previous recommendations
- Gets better the more you use it

### Plex Integration
- Works directly with your Plex server
- Adds items to your Plex watchlist
- Creates collections in Plex
- Auto-adds highly rated movies to collections

## Getting Started

### What You'll Need
1. A Plex server
2. Free API keys from:
   - [TMDB](https://www.themoviedb.org/documentation/api)
   - [TVDB](https://thetvdb.com/api-information)
3. Your [Plex token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/)

### Installation

#### Windows
1. Download the latest Windows release
2. Run the executable
3. Enter your Plex and API information
4. Select which libraries to use
5. Start rating content!

#### MacOS
1. Download the latest MacOS release, or install from source:
```bash
# Install dependencies
brew install python@3.8
brew install sqlite3

# Clone repository
git clone https://github.com/yourusername/recommend-for-plex.git
cd recommend-for-plex

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the application
python main.py
```

#### Linux
1. Install dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.8 python3-pip python3-venv sqlite3 python3-pyqt6

# Fedora
sudo dnf install python3.8 python3-pip python3-virtualenv sqlite python3-pyqt6

# Clone repository
git clone https://github.com/yourusername/recommend-for-plex.git
cd recommend-for-plex

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the application
python main.py
```

### First-Time Setup
1. In the Configuration tab:
   - Enter your Plex URL (e.g., http://localhost:32400)
   - Add your Plex token
   - Add your TVDB and TMDB API keys
2. Click "Save Configuration"
3. Click "Load Libraries" to see your Plex libraries
4. Select which libraries to include
5. Optional: Set up collection preferences
   - Choose a collection name
   - Set rating threshold for auto-adding
6. Click "Scan Libraries" to begin

## How to Use

### Rating Content
- Open the Movies or TV Shows tab
- Rate items from 1-10
- Skip items you don't want to rate
- Block items you never want to see
- Use the Back button to return to previous items

### Using Collections
- Movies can be added to collections automatically or manually
- Set a minimum rating to auto-add movies to collections
- Configure your preferred collection name
- Collections appear directly in your Plex library

### Managing Your Watchlist
- Add any item to your Plex watchlist
- Clear your entire watchlist from the Configuration tab
- Items sync with your Plex server

## Need Help?

### Common Issues
1. Can't connect to Plex?
   - Check your Plex URL
   - Verify your token
   - Make sure Plex is running

2. Recommendations not improving?
   - Rate more items
   - Try using the full rating scale
   - Check if library scan completed

3. Features not working?
   - Verify API keys
   - Check error messages in Configuration tab
   - Try rescanning libraries


## License
This project is under the GPLv3 License - see [LICENSE](LICENSE) file.
