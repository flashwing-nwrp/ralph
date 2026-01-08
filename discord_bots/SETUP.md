# Discord Bot Setup Guide

Complete step-by-step instructions for creating and running the RALPH agent bots.

---

## Part 1: Create Discord Applications (Repeat 5 Times)

You need to create **5 separate bot applications** in the Discord Developer Portal.

### Step 1: Go to Discord Developer Portal

1. Open https://discord.com/developers/applications
2. Log in with your Discord account

### Step 2: Create Application

Click **"New Application"** button (top right)

### Step 3: Application Details (Copy-Paste These)

Create each application with these exact settings:

---

#### Bot 1: RALPH Tuning Agent

**Application Name:**
```
RALPH Tuning Agent
```

**Description:**
```
Parameter optimization agent for RALPH trading system. Handles hyperparameter tuning, learning rate scheduling, and model configuration optimization.
```

---

#### Bot 2: RALPH Backtest Agent

**Application Name:**
```
RALPH Backtest Agent
```

**Description:**
```
Simulation and metrics agent for RALPH trading system. Runs historical backtests, calculates performance metrics, and validates strategy performance.
```

---

#### Bot 3: RALPH Risk Agent

**Application Name:**
```
RALPH Risk Agent
```

**Description:**
```
Safety auditing agent for RALPH trading system. Monitors risk exposure, validates position limits, and enforces safety constraints.
```

---

#### Bot 4: RALPH Strategy Agent

**Application Name:**
```
RALPH Strategy Agent
```

**Description:**
```
Logic and features agent for RALPH trading system. Designs trading strategies, implements signal generation, and manages feature engineering.
```

---

#### Bot 5: RALPH Data Agent

**Application Name:**
```
RALPH Data Agent
```

**Description:**
```
Preprocessing and denoising agent for RALPH trading system. Handles data ingestion, cleaning, normalization, and feature extraction.
```

---

### Step 4: Configure Bot Settings

For **each** application:

1. Click **"Bot"** in the left sidebar
2. Click **"Add Bot"** → Confirm "Yes, do it!"
3. **Copy the Token** immediately (you can only see it once!)
   - Click "Reset Token" if you need to regenerate
   - Save to `.env` file (see Part 2)

4. Configure these settings:

**PUBLIC BOT:** `OFF` (uncheck)
- Only you should be able to add this bot

**REQUIRES OAUTH2 CODE GRANT:** `OFF` (uncheck)

**PRESENCE INTENT:** `OFF` (uncheck)
- Not needed

**SERVER MEMBERS INTENT:** `OFF` (uncheck)
- Not needed

**MESSAGE CONTENT INTENT:** `ON` (check)
- Required to read message content

### Step 5: Configure OAuth2 Permissions

For **each** application:

1. Click **"OAuth2"** → **"URL Generator"** in the left sidebar

2. **SCOPES** - Check these:
   - `bot`

3. **BOT PERMISSIONS** - Check these:
   - `Send Messages`
   - `Send Messages in Threads`
   - `Create Public Threads`
   - `Create Private Threads`
   - `Read Message History`
   - `Add Reactions`
   - `Use Slash Commands`

   **Permission Integer:** `397553213504`

4. **Copy the generated URL** at the bottom

### Step 6: Invite Bots to Your Server

1. Open each generated OAuth2 URL in your browser
2. Select your private RALPH server
3. Click "Authorize"
4. Complete the CAPTCHA

Repeat for all 5 bots.

---

## Part 2: Environment Configuration

### Create .env File

```bash
cd discord_bots
cp .env.example .env
```

Edit `.env` and paste your tokens:

```env
# Bot Tokens (from Discord Developer Portal)
TUNING_AGENT_TOKEN=your_tuning_agent_token_here
BACKTEST_AGENT_TOKEN=your_backtest_agent_token_here
RISK_AGENT_TOKEN=your_risk_agent_token_here
STRATEGY_AGENT_TOKEN=your_strategy_agent_token_here
DATA_AGENT_TOKEN=your_data_agent_token_here

# Server Configuration
DISCORD_GUILD_ID=your_server_id_here
```

### Get Your Server (Guild) ID

1. Open Discord
2. Go to User Settings → Advanced → Enable "Developer Mode"
3. Right-click your server name → "Copy Server ID"
4. Paste into `.env` as `DISCORD_GUILD_ID`

---

## Part 3: Install Dependencies

```bash
cd discord_bots

# Create virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Part 4: Create Discord Channels

In your private server, create these channels:

### Text Channels

| Channel Name | Purpose |
|-------------|---------|
| `#ralph-team` | Main collaboration channel |
| `#tuning` | Parameter optimization discussions |
| `#backtesting` | Simulation results and metrics |
| `#risk` | Safety audits and alerts |
| `#strategy` | Logic and feature design |
| `#data` | Data pipeline discussions |
| `#bot-logs` | System status messages |
| `#error-logs` | Error reporting |

### Recommended Category Structure

```
RALPH AGENTS
├── #ralph-team
├── #tuning
├── #backtesting
├── #risk
├── #strategy
└── #data

SYSTEM
├── #bot-logs
└── #error-logs
```

---

## Part 5: Run the Bots

### Option A: Run All Bots Together

```bash
python run_all.py
```

This starts all 5 agents concurrently using asyncio.

### Option B: Run Single Bot (for testing)

```bash
# Run specific agent
python run_single.py tuning
python run_single.py backtest
python run_single.py risk
python run_single.py strategy
python run_single.py data
```

### Option C: Run in Background (Production)

**Windows (PowerShell):**
```powershell
Start-Process python -ArgumentList "run_all.py" -WindowStyle Hidden
```

**Linux/macOS:**
```bash
nohup python run_all.py > logs/bots.log 2>&1 &
```

**With systemd (Linux):**
```bash
# Create service file at /etc/systemd/system/ralph-bots.service
sudo systemctl enable ralph-bots
sudo systemctl start ralph-bots
```

---

## Part 6: Verify Setup

After starting the bots:

1. Check `#bot-logs` channel - each bot should post "Online and ready!"
2. Type `!ping` in `#ralph-team` - all bots should respond
3. Type `!status` in any agent channel - that agent should report status

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Token invalid" | Regenerate token in Developer Portal |
| "Missing Access" | Re-invite bot with correct permissions |
| "Missing Intent" | Enable MESSAGE CONTENT intent in Developer Portal |
| Bot not responding | Check if bot is online in server member list |
| Rate limited | Wait 60 seconds, reduce message frequency |

---

## Part 7: Channel IDs (Optional)

If you want bots to post to specific channels automatically, add channel IDs to `.env`:

1. Enable Developer Mode in Discord
2. Right-click each channel → "Copy Channel ID"
3. Add to `.env`:

```env
# Channel IDs (optional, for auto-posting)
CHANNEL_RALPH_TEAM=123456789012345678
CHANNEL_TUNING=123456789012345679
CHANNEL_BACKTESTING=123456789012345680
CHANNEL_RISK=123456789012345681
CHANNEL_STRATEGY=123456789012345682
CHANNEL_DATA=123456789012345683
CHANNEL_BOT_LOGS=123456789012345684
CHANNEL_ERROR_LOGS=123456789012345685
```

---

## Quick Reference: Permission Integer

If asked for a permission integer, use: **397553213504**

This includes:
- Send Messages (2048)
- Send Messages in Threads (274877906944)
- Create Public Threads (34359738368)
- Create Private Threads (68719476736)
- Read Message History (65536)
- Add Reactions (64)
- Use Slash Commands (2147483648)
