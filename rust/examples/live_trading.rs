//! Example: Live Trading Simulation
//!
//! This example demonstrates how to use the Mamba model
//! in a simulated live trading environment with Bybit data.
//!
//! Run with:
//! ```bash
//! cargo run --example live_trading
//! ```

use mamba_trading::{
    api::bybit::BybitClient,
    data::{features::FeatureEngineer, OhlcvBar, TradingSignal},
    model::{trading::MambaTrading, MambaConfig},
};

/// Trading configuration
struct TradingConfig {
    symbol: String,
    interval: String,
    lookback: usize,
    initial_capital: f64,
    position_size: f64,
    buy_threshold: f64,
    sell_threshold: f64,
    stop_loss: f64,
    take_profit: f64,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            interval: "60".to_string(),
            lookback: 60,
            initial_capital: 10000.0,
            position_size: 0.5, // 50% of capital per trade
            buy_threshold: 0.55,
            sell_threshold: 0.55,
            stop_loss: 0.03,     // 3%
            take_profit: 0.05,  // 5%
        }
    }
}

/// Simple trading simulator
struct TradingSimulator {
    config: TradingConfig,
    model: MambaTrading,
    fe: FeatureEngineer,
    capital: f64,
    position: f64,
    entry_price: f64,
    trades: Vec<Trade>,
}

struct Trade {
    entry_price: f64,
    exit_price: f64,
    pnl: f64,
    signal: TradingSignal,
}

impl TradingSimulator {
    fn new(config: TradingConfig, model: MambaTrading) -> Self {
        Self {
            capital: config.initial_capital,
            config,
            model,
            fe: FeatureEngineer::new(true),
            position: 0.0,
            entry_price: 0.0,
            trades: Vec::new(),
        }
    }

    fn run(&mut self, bars: &[OhlcvBar]) {
        // Compute features for all bars
        let features = self.fe.compute_features(bars);

        for i in self.config.lookback..bars.len() {
            let current_price = bars[i].close;

            // Check stop-loss / take-profit
            if self.position > 0.0 {
                let pnl_pct = (current_price - self.entry_price) / self.entry_price;

                if pnl_pct <= -self.config.stop_loss {
                    self.close_position(current_price, "Stop Loss");
                } else if pnl_pct >= self.config.take_profit {
                    self.close_position(current_price, "Take Profit");
                }
            }

            // Get model prediction
            let start = i - self.config.lookback;
            let sequence: Vec<Vec<f64>> = features[start..i]
                .iter()
                .map(|f| f.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect())
                .collect();

            let prediction = self.model.predict(&sequence);

            // Execute trading logic
            match prediction.signal {
                TradingSignal::Buy if prediction.confidence > self.config.buy_threshold && self.position == 0.0 => {
                    self.open_position(current_price);
                }
                TradingSignal::Sell if prediction.confidence > self.config.sell_threshold && self.position > 0.0 => {
                    self.close_position(current_price, "Signal");
                }
                _ => {}
            }
        }

        // Close any remaining position
        if self.position > 0.0 {
            let final_price = bars.last().map(|b| b.close).unwrap_or(self.entry_price);
            self.close_position(final_price, "End of Simulation");
        }
    }

    fn open_position(&mut self, price: f64) {
        let trade_capital = self.capital * self.config.position_size;
        self.position = trade_capital / price;
        self.capital -= trade_capital;
        self.entry_price = price;
    }

    fn close_position(&mut self, price: f64, _reason: &str) {
        let proceeds = self.position * price;
        let pnl = proceeds - (self.position * self.entry_price);

        self.trades.push(Trade {
            entry_price: self.entry_price,
            exit_price: price,
            pnl,
            signal: TradingSignal::Sell,
        });

        self.capital += proceeds;
        self.position = 0.0;
        self.entry_price = 0.0;
    }

    fn get_results(&self) -> SimulationResults {
        let total_pnl: f64 = self.trades.iter().map(|t| t.pnl).sum();
        let winning_trades = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let total_trades = self.trades.len();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital * 100.0;

        SimulationResults {
            final_capital: self.capital,
            total_pnl,
            total_return,
            total_trades,
            winning_trades,
            win_rate,
        }
    }
}

struct SimulationResults {
    final_capital: f64,
    total_pnl: f64,
    total_return: f64,
    total_trades: usize,
    winning_trades: usize,
    win_rate: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mamba Trading - Live Trading Simulation");
    println!("========================================\n");

    // Trading configuration
    let config = TradingConfig::default();

    println!("Configuration:");
    println!("  Symbol:          {}", config.symbol);
    println!("  Interval:        {} minutes", config.interval);
    println!("  Initial Capital: ${:.2}", config.initial_capital);
    println!("  Position Size:   {:.0}%", config.position_size * 100.0);
    println!("  Buy Threshold:   {:.0}%", config.buy_threshold * 100.0);
    println!("  Sell Threshold:  {:.0}%", config.sell_threshold * 100.0);
    println!("  Stop Loss:       {:.1}%", config.stop_loss * 100.0);
    println!("  Take Profit:     {:.1}%", config.take_profit * 100.0);

    // Fetch market data
    println!("\nFetching market data from Bybit...");
    let client = BybitClient::new();

    let bars = match client.fetch_extended(&config.symbol, &config.interval, 500).await {
        Ok(b) => {
            println!("Fetched {} bars", b.len());
            b
        }
        Err(e) => {
            println!("Failed to fetch live data: {}", e);
            println!("Using synthetic data instead...\n");
            generate_synthetic_data(500)
        }
    };

    // Create feature engineer to get number of features
    let fe = FeatureEngineer::new(true);
    let n_features = fe.feature_names().len();

    // Create model
    println!("\nInitializing Mamba model...");
    let model_config = MambaConfig::small(n_features);
    let model = MambaTrading::new(model_config);
    println!("Model initialized with {} features", n_features);

    // Run simulation
    println!("\nRunning trading simulation...");
    let mut simulator = TradingSimulator::new(config, model);
    simulator.run(&bars);

    // Print results
    let results = simulator.get_results();
    println!("\n{:=<50}", "");
    println!("SIMULATION RESULTS");
    println!("{:=<50}", "");
    println!("Final Capital:    ${:.2}", results.final_capital);
    println!("Total P&L:        ${:.2}", results.total_pnl);
    println!("Total Return:     {:.2}%", results.total_return);
    println!("{:-<50}", "");
    println!("Total Trades:     {}", results.total_trades);
    println!("Winning Trades:   {}", results.winning_trades);
    println!("Win Rate:         {:.1}%", results.win_rate * 100.0);
    println!("{:=<50}", "");

    // Compare with buy-and-hold
    if let (Some(first), Some(last)) = (bars.first(), bars.last()) {
        let buy_hold_return = (last.close - first.close) / first.close * 100.0;
        println!("\nBuy & Hold Return: {:.2}%", buy_hold_return);
        println!(
            "Strategy vs B&H:   {:.2}%",
            results.total_return - buy_hold_return
        );
    }

    println!("\nNote: This is a demonstration with a randomly initialized model.");
    println!("In production, you would train the model on historical data first.");

    Ok(())
}

/// Generate synthetic market data for demonstration.
fn generate_synthetic_data(n_bars: usize) -> Vec<OhlcvBar> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut bars = Vec::with_capacity(n_bars);
    let mut price: f64 = 45000.0; // Starting price for BTC-like simulation
    let mut timestamp = 1704067200000_i64;

    for _ in 0..n_bars {
        let change: f64 = rng.gen_range(-200.0..250.0);
        price = (price + change).max(30000.0);

        let volatility: f64 = rng.gen_range(50.0..300.0);
        let open: f64 = price - volatility * rng.gen_range(-0.5_f64..0.5);
        let close: f64 = price + volatility * rng.gen_range(-0.5_f64..0.5);
        let high = open.max(close) + volatility * rng.gen_range(0.0_f64..1.0);
        let low = open.min(close) - volatility * rng.gen_range(0.0_f64..1.0);
        let volume: f64 = rng.gen_range(100.0..1000.0);

        bars.push(OhlcvBar {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });

        timestamp += 3600000;
        price = close;
    }

    bars
}
