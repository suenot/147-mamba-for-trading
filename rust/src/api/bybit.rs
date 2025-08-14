//! Bybit API client for fetching cryptocurrency market data.
//!
//! This module provides a client for interacting with the Bybit exchange API
//! to fetch historical kline (candlestick) data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::OhlcvBar;

/// Errors that can occur when interacting with the Bybit API.
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API returned error: {message} (code: {code})")]
    ApiError { code: i32, message: String },

    #[error("Failed to parse response: {0}")]
    ParseError(String),
}

/// Response wrapper from Bybit API.
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result from Bybit API.
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client.
///
/// # Example
///
/// ```rust,no_run
/// use mamba_trading::api::bybit::BybitClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = BybitClient::new();
///     let klines = client.fetch_klines("BTCUSDT", "60", 100).await?;
///     println!("Fetched {} klines", klines.len());
///     Ok(())
/// }
/// ```
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit API client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a client with a custom base URL (useful for testing).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of OHLCV bars sorted by timestamp (ascending).
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<OhlcvBar>, BybitError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()
            .await?;

        let api_response: BybitResponse<KlineResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let mut bars: Vec<OhlcvBar> = api_response
            .result
            .list
            .into_iter()
            .filter_map(|row| self.parse_kline_row(&row).ok())
            .collect();

        // Sort by timestamp (ascending) - Bybit returns descending
        bars.sort_by_key(|bar| bar.timestamp);

        Ok(bars)
    }

    /// Fetch extended historical data by making multiple API calls.
    pub async fn fetch_extended(
        &self,
        symbol: &str,
        interval: &str,
        total_bars: usize,
    ) -> Result<Vec<OhlcvBar>, BybitError> {
        let mut all_bars = Vec::with_capacity(total_bars);
        let mut end_time: Option<i64> = None;

        while all_bars.len() < total_bars {
            let remaining = total_bars - all_bars.len();
            let batch_size = remaining.min(1000) as u32;

            let url = format!("{}/v5/market/kline", self.base_url);
            let mut query = vec![
                ("category", "linear".to_string()),
                ("symbol", symbol.to_string()),
                ("interval", interval.to_string()),
                ("limit", batch_size.to_string()),
            ];

            if let Some(end) = end_time {
                query.push(("end", end.to_string()));
            }

            let response = self
                .client
                .get(&url)
                .query(&query)
                .send()
                .await?;

            let api_response: BybitResponse<KlineResult> = response.json().await?;

            if api_response.ret_code != 0 {
                return Err(BybitError::ApiError {
                    code: api_response.ret_code,
                    message: api_response.ret_msg,
                });
            }

            let bars: Vec<OhlcvBar> = api_response
                .result
                .list
                .into_iter()
                .filter_map(|row| self.parse_kline_row(&row).ok())
                .collect();

            if bars.is_empty() {
                break;
            }

            // Get the earliest timestamp for the next batch
            if let Some(earliest) = bars.iter().min_by_key(|b| b.timestamp) {
                end_time = Some(earliest.timestamp - 1);
            }

            all_bars.extend(bars);

            // Small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Sort by timestamp
        all_bars.sort_by_key(|bar| bar.timestamp);

        Ok(all_bars)
    }

    /// Parse a kline row from Bybit API response.
    fn parse_kline_row(&self, row: &[String]) -> Result<OhlcvBar, BybitError> {
        if row.len() < 7 {
            return Err(BybitError::ParseError(
                "Invalid kline row length".to_string(),
            ));
        }

        Ok(OhlcvBar {
            timestamp: row[0]
                .parse::<i64>()
                .map_err(|e| BybitError::ParseError(e.to_string()))?,
            open: row[1]
                .parse::<f64>()
                .map_err(|e| BybitError::ParseError(e.to_string()))?,
            high: row[2]
                .parse::<f64>()
                .map_err(|e| BybitError::ParseError(e.to_string()))?,
            low: row[3]
                .parse::<f64>()
                .map_err(|e| BybitError::ParseError(e.to_string()))?,
            close: row[4]
                .parse::<f64>()
                .map_err(|e| BybitError::ParseError(e.to_string()))?,
            volume: row[5]
                .parse::<f64>()
                .map_err(|e| BybitError::ParseError(e.to_string()))?,
        })
    }

    /// Fetch available trading pairs.
    pub async fn get_tickers(&self, category: &str) -> Result<Vec<TickerInfo>, BybitError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", category)])
            .send()
            .await?;

        let api_response: BybitResponse<TickerResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        Ok(api_response.result.list)
    }
}

/// Ticker result from Bybit API.
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerInfo>,
}

/// Information about a trading ticker.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kline_row() {
        let client = BybitClient::new();
        let row = vec![
            "1704067200000".to_string(),
            "42000.5".to_string(),
            "42500.0".to_string(),
            "41800.0".to_string(),
            "42300.0".to_string(),
            "1234.5".to_string(),
            "52000000.0".to_string(),
        ];

        let bar = client.parse_kline_row(&row).unwrap();
        assert_eq!(bar.timestamp, 1704067200000);
        assert!((bar.open - 42000.5).abs() < 0.01);
        assert!((bar.close - 42300.0).abs() < 0.01);
    }
}
