//! Example: Fetching market data from Bybit
//!
//! This example demonstrates how to use the Bybit API client
//! to fetch cryptocurrency market data.
//!
//! Run with:
//! ```bash
//! cargo run --example fetch_data
//! ```

use mamba_trading::api::bybit::BybitClient;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mamba Trading - Data Fetching Example");
    println!("=====================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch BTC/USDT klines
    println!("Fetching BTC/USDT hourly klines from Bybit...");
    let klines = client.fetch_klines("BTCUSDT", "60", 100).await?;

    println!("Fetched {} klines\n", klines.len());

    // Display first few bars
    println!("Recent data:");
    println!("{:-<80}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{:-<80}", "");

    for bar in klines.iter().rev().take(10) {
        let datetime = chrono::DateTime::from_timestamp_millis(bar.timestamp)
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            datetime, bar.open, bar.high, bar.low, bar.close, bar.volume
        );
    }

    // Calculate some basic statistics
    if !klines.is_empty() {
        let closes: Vec<f64> = klines.iter().map(|b| b.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|b| b.volume).collect();

        let avg_close = closes.iter().sum::<f64>() / closes.len() as f64;
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;

        let min_close = closes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_close = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n{:-<80}", "");
        println!("Statistics:");
        println!("  Average Close: ${:.2}", avg_close);
        println!("  Min Close:     ${:.2}", min_close);
        println!("  Max Close:     ${:.2}", max_close);
        println!("  Average Volume: {:.2}", avg_volume);
    }

    // Fetch ETH/USDT for comparison
    println!("\n\nFetching ETH/USDT data...");
    let eth_klines = client.fetch_klines("ETHUSDT", "60", 10).await?;

    if let Some(last) = eth_klines.last() {
        println!("Latest ETH/USDT price: ${:.2}", last.close);
    }

    // Get available tickers
    println!("\n\nFetching available tickers...");
    let tickers = client.get_tickers("linear").await?;

    println!("Found {} trading pairs", tickers.len());
    println!("\nTop 5 by 24h volume:");
    let mut sorted_tickers = tickers.clone();
    sorted_tickers.sort_by(|a, b| {
        b.volume_24h
            .parse::<f64>()
            .unwrap_or(0.0)
            .partial_cmp(&a.volume_24h.parse::<f64>().unwrap_or(0.0))
            .unwrap()
    });

    for ticker in sorted_tickers.iter().take(5) {
        println!(
            "  {} - Price: ${}, 24h Vol: ${}",
            ticker.symbol, ticker.last_price, ticker.volume_24h
        );
    }

    // Save data to file
    let save_path = Path::new("btc_data.csv");
    println!("\n\nSaving BTC data to {:?}...", save_path);
    mamba_trading::data::loader::DataLoader::save_csv(&klines, save_path)?;
    println!("Data saved successfully!");

    Ok(())
}
