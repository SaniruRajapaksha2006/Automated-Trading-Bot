# 🤖 Professional Trading Bot

An institutional-grade algorithmic trading system with machine learning, risk management, and real-time execution.


### Completed scope
- Professional project structure
- Alpaca API integration (paper trading)
- Broker connection class
- Order manager with position sizing
- Logging system
- Test suite

### Features Implemented
- ✅ Paper trading connection
- ✅ Account information retrieval
- ✅ Real-time stock quotes
- ✅ Market order placement
- ✅ Position size calculator (Kelly-based)
- ✅ Order cancellation

### Tech Stack
- **Broker API**: Alpaca (free paper trading)
- **Language**: Python 3.14
- **Key Libraries**: alpaca-py, pandas, numpy

### How to Run
```bash
# Clone the repository
git clone https://github.com/SaniruRajapaksha2006/Trading-Bot.git
cd Trading-Bot

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Alpaca API keys to .env
# Then run the test
python tests/test_broker.py