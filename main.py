import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Import our modules
from models.bayesian_hmm import BayesianHMM
from features.feature_extractor import FeatureExtractor
from data.data_loader import load_csv_data
from data.feature_processor import normalize_features, combine_features
from analysis.regime_analyzer import analyze_market_regimes, analyze_cross_asset_regimes
from analysis.regime_forecast import forecast_next_regime
from trading.strategy_recommender import recommend_option_strategy
from utils.file_io import (
    save_results, save_cross_asset_analysis, 
    create_summary_file, add_to_summary
)

# Import new visualization modules
from visualization.regime_plots import (
    plot_regime_distribution,
    plot_regime_timeline,
    plot_regime_characteristics,
    plot_transition_heatmap,
    plot_performance_metrics
)
from visualization.dashboard import create_dashboard

# Import new analytics modules
from analytics.regime_statistics import (
    calculate_extended_regime_statistics,
    calculate_regime_correlations,
    analyze_risk_adjusted_metrics
)
from analytics.transition_analysis import (
    calculate_transition_probabilities,
    analyze_regime_duration,
    predict_regime_stability
)

# Import new IO modules
from data_io.export import (
    export_regime_data_to_json,
    export_regime_data_to_csv,
    export_transition_matrix
)
from data_io.report import (
    generate_regime_report,
    generate_comparative_report
)


def process_single_file(file_path, n_regimes, feature_config, feature_weights, output_dir):
    """
    Process a single file to analyze market regimes
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    n_regimes : int
        Number of regimes to identify
    feature_config : dict
        Configuration for feature extraction
    feature_weights : dict
        Weights for combining features
    output_dir : str
        Directory to save the results
        
    Returns:
    --------
    tuple
        (ticker, current_regime, recommendations, error)
    """
    # Extract ticker from filename
    file_name = os.path.basename(file_path)
    ticker_match = re.match(r'([A-Z]+)_', file_name)
    if ticker_match:
        ticker = ticker_match.group(1)
    else:
        ticker = file_name.split('_')[0]  # Fallback
    
    print(f"\nProcessing {ticker} from {file_name}...")
    
    try:
        # Load data
        dates, ohlcv = load_csv_data(file_path)
        
        if len(dates) < 100:
            raise ValueError(f"Not enough data points: {len(dates)}")
        
        # Extract features
        features = FeatureExtractor.extract_features(ohlcv, config=feature_config)
        
        # Normalize features
        normalized_features = normalize_features(features)
        
        # Combine features into a single series
        combined_feature = combine_features(normalized_features, weights=feature_weights)
        
        # Add combined feature to the features dict for future use
        features['combined'] = combined_feature
        
        # Fit Bayesian HMM
        hmm = BayesianHMM(n_regimes=n_regimes)
        hmm.fit(combined_feature, max_iter=100)
        
        # Analyze market regimes
        results = analyze_market_regimes(dates, ohlcv, features, hmm, n_regimes=n_regimes)
        
        # Calculate extended regime statistics
        extended_stats = calculate_extended_regime_statistics(
            results['regimes'], features, ohlcv, dates
        )
        
        # Add extended stats to results
        results['extended_stats'] = extended_stats
        
        # Analyze risk-adjusted metrics
        risk_metrics = analyze_risk_adjusted_metrics(extended_stats)
        results['risk_metrics'] = risk_metrics
        
        # Analyze regime durations
        duration_stats = analyze_regime_duration(results['regimes'], dates)
        results['duration_stats'] = duration_stats
        
        # Create ticker-specific output directory
        ticker_output_dir = os.path.join(output_dir, ticker)
        if not os.path.exists(ticker_output_dir):
            os.makedirs(ticker_output_dir)
        
        # Save results in various formats
        save_results(results, ticker_output_dir, ticker=ticker)
        
        # Export data to JSON and CSV
        export_regime_data_to_json(
            results, 
            os.path.join(ticker_output_dir, f"{ticker}_regime_data.json")
        )
        
        export_regime_data_to_csv(
            results,
            os.path.join(ticker_output_dir, f"{ticker}_regime_data.csv")
        )
        
        export_transition_matrix(
            results,
            os.path.join(ticker_output_dir, f"{ticker}_transition_matrix.csv"),
            format='csv'
        )
        
        # Generate report
        generate_regime_report(
            results,
            os.path.join(ticker_output_dir, f"{ticker}_regime_report.txt"),
            include_time_series=True
        )
        
        # Create visualizations
        plot_regime_distribution(
            results['regime_stats'],
            output_path=os.path.join(ticker_output_dir, f"{ticker}_regime_distribution.png")
        )
        
        plot_regime_timeline(
            dates, results['regimes'], results['regime_stats'], ohlcv,
            output_path=os.path.join(ticker_output_dir, f"{ticker}_regime_timeline.png")
        )
        
        plot_regime_characteristics(
            results['regime_stats'],
            output_path=os.path.join(ticker_output_dir, f"{ticker}_regime_characteristics.png")
        )
        
        plot_transition_heatmap(
            results['hmm_model'],
            output_path=os.path.join(ticker_output_dir, f"{ticker}_transition_heatmap.png")
        )
        
        plot_performance_metrics(
            results['regime_stats'], features, results['regimes'], dates,
            output_path=os.path.join(ticker_output_dir, f"{ticker}_performance_metrics.png")
        )
        
        # Create interactive dashboard
        create_dashboard(
            results, 
            ticker_output_dir, 
            filename=f"{ticker}_dashboard.html"
        )
        
        # Generate current regime forecast and strategy recommendation
        current_regime = results['regimes'][-1]
        forecasts = forecast_next_regime(results['hmm_model'], current_regime, horizon=5)
        
        # Get current regime duration
        current_duration = 1
        for i in range(len(results['regimes'])-2, -1, -1):
            if results['regimes'][i] == current_regime:
                current_duration += 1
            else:
                break
        
        # Predict regime stability
        stability = predict_regime_stability(
            current_regime, current_duration, results['regimes'], dates
        )
        
        # Keep forecasts as a list for recommend_option_strategy
        recommendations = recommend_option_strategy(
            regime_stats=results['regime_stats'],
            current_regime=current_regime,
            forecasts=forecasts
        )
        
        # Add stability to recommendations after the fact
        recommendations['stability'] = stability
        
        # Print recommendations
        print(f"\n=== {ticker} OPTION STRATEGY RECOMMENDATIONS ===")
        print(f"Current Regime: {recommendations['current_regime']}")
        print(f"Primary Strategy: {recommendations['primary_strategy']}")
        print(f"Alternative Strategy: {recommendations['alternative_strategy']}")
        print(f"Position Sizing: {recommendations['position_sizing']}")
        print(f"Wing Width: {recommendations['wing_width']}")
        print(f"Recommended Expiration: {recommendations['expiration']}")
        print(f"Confidence Level: {recommendations['confidence']}")
        print(f"Regime Transition Risk: {recommendations['transition_risk']['description']} ({recommendations['transition_risk']['probability']:.2f})")
        
        return ticker, current_regime, recommendations, None
    
    except Exception as e:
        import traceback
        print(f"Error analyzing {ticker}: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")
        return ticker, None, None, str(e)

def main():
    """Main function to run Bayesian regime detection analysis on multiple stock files"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Bayesian Market Regime Analysis')
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\cinco\Desktop\Cinco-Quant\00_raw_data\5.16",
                        #default='/Users/jazzhashzzz/Desktop/Cinco-Quant/00_raw_data/5.15',
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default=r"C:\Users\cinco\Desktop\Cinco-Quant\00_raw_data\regime_analysis5.16",                        
                        #default='C:\Users\cinco\Desktop\Cinco-Quant\00_raw_data\regime_analysis5.16',
                        help='Directory to save output files')
    parser.add_argument('--n_regimes', type=int, default=3,
                        help='Number of regimes to identify')
    parser.add_argument('--single_file', type=str, default=None,
                        help='Process a single file instead of all files in data_dir')
    args = parser.parse_args()
    
    # Set parameters
    data_dir = args.data_dir
    output_dir = args.output_dir
    n_regimes = args.n_regimes
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Feature configuration
    feature_config = {
        'volatility_window': 5,
        'atr_window': 14,
        'bb_window': 20,
        'momentum_short': 10,
        'momentum_long': 20,
        'rsi_window': 14
    }
    
    # Feature weights (which features to emphasize)
    feature_weights = {
        'log_returns': 0.15,
        'volatility': 0.30,
        'atr': 0.20,
        'bb_width': 0.15,
        'momentum': 0.10,
        'rsi': 0.10
    }
    
    # Create summary file
    summary_file = create_summary_file(output_dir)
    
    # If a single file is specified, process only that file
    if args.single_file:
        file_path = args.single_file
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        ticker, current_regime, recommendations, error = process_single_file(
            file_path, n_regimes, feature_config, feature_weights, output_dir
        )
        
        # Add to summary
        add_to_summary(summary_file, ticker, current_regime, recommendations, error)
        
        if error:
            print(f"Analysis failed: {error}")
        else:
            print("Analysis complete.")
        
        return
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*_1daily_*.csv"))
    
    # Process each file
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    successful = 0
    failed = 0
    
    # Dictionary to store ticker stats for comparative report
    ticker_stats = {}
    
    for file_path in csv_files:
        ticker, current_regime, recommendations, error = process_single_file(
            file_path, n_regimes, feature_config, feature_weights, output_dir
        )
        
        # Add to summary
        add_to_summary(summary_file, ticker, current_regime, recommendations, error)
        
        if error:
            failed += 1
        else:
            successful += 1
            
            # Add to ticker stats for comparative report
            if current_regime is not None:
                ticker_dir = os.path.join(output_dir, ticker)
                
                try:
                    # Load results for this ticker
                    result_path = os.path.join(ticker_dir, f"{ticker}_results.npz")
                    results = np.load(result_path, allow_pickle=True)
                    regime_stats = results['regime_stats'].item()
                    
                    # Get current regime stats
                    current_regime_stats = next(
                        (s for s in regime_stats if s['regime'] == current_regime),
                        None
                    )
                    
                    ticker_stats[ticker] = {
                        'current_regime': current_regime,
                        'current_regime_stats': current_regime_stats,
                        'recommendations': recommendations
                    }
                except:
                    # If there's an error, just skip this ticker for comparative analysis
                    pass
    
    print(f"\nAnalysis complete. Successfully analyzed {successful} stocks. Failed: {failed}")
    print(f"Results saved to {output_dir}")
    print(f"Summary file: {summary_file}")
    
    # Cross-asset analysis
    regime_groups = analyze_cross_asset_regimes(summary_file)
    if regime_groups:
        # Save cross-asset analysis
        save_cross_asset_analysis(regime_groups, output_dir)
        
        # Generate comparative report if we have ticker statistics
        if ticker_stats:
            comparative_report_path = os.path.join(output_dir, "cross_asset_report.txt")
            generate_comparative_report(regime_groups, ticker_stats, comparative_report_path)
            print(f"Comparative report generated: {comparative_report_path}")


if __name__ == "__main__":
    main()
