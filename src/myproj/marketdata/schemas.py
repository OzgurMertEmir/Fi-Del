"""Schema definitions for store ingestion."""

RAW_COLS = ["rec_type", "ts_ms", "bid_px", "bid_qty", "ask_px", "ask_qty", "vol_base"]
NUMERIC_COLS = ["ts_ms", "bid_px", "bid_qty", "ask_px", "ask_qty", "vol_base"]
PRICE_COLS = ["bid_px", "ask_px"]
QTY_COLS = ["bid_qty", "ask_qty"]
